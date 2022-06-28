from zonopy.environments.wrappers.gym_wrapper import GymWrapper
from zonopy.environments.wrappers.utils import dict_torch2np
from gym import spaces
import numpy as np
import torch
# Custom Robosuite wrapper which creates a DictGym env instead of a regular one. For use with HER
class DictGymWrapper(GymWrapper):
    def __init__(self, env, goal_state_key=None, acheived_state_key=None, *args, **kwargs):
        # Run super method
        super().__init__(env=env, *args, **kwargs)
        self.goal_state_key = goal_state_key
        self.acheived_state_key = acheived_state_key
        obs = self.observation_spec()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                -np.inf, np.inf, shape=obs[goal_state_key].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                -np.inf, np.inf, shape=obs[acheived_state_key].shape, dtype="float32"
                ),
                observation=self.observation_space,
            )
        )

    def _create_obs_dict(self, obs):
        vec_obs = self._flatten_obs(obs)
        obs_dict = {
            "observation": vec_obs,
            "achieved_goal": obs[self.acheived_state_key].numpy().astype(float).flatten(),
            "desired_goal": obs[self.goal_state_key].numpy().astype(float).flatten(),
        }
        return obs_dict

    def reset(self):
        obs = self.env.reset()
        return self._create_obs_dict(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(torch.tensor(action,dtype=torch.get_default_dtype()))
        info['action_taken'] = action
        return self._create_obs_dict(obs), float(reward), done, dict_torch2np(info)

    # A lazy way to do this. if vectorizable (dependent on environment), this will dramatically speed up results.
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = []
        default_dtype = torch.get_default_dtype()
        for i,info_dict in enumerate(info):
            reward.append(
                float(self.env.reward(action = torch.tensor(info_dict['action_taken'],dtype=default_dtype),
                                qpos = torch.tensor(achieved_goal[i],dtype=default_dtype),
                                qgoal = torch.tensor(desired_goal[i],dtype=default_dtype)))
            )
        
        return np.array(reward)


if __name__ == '__main__':
    from zonopy.environments.wrappers.wrapper import Wrapper
    from zonopy.environments.arm_2d import Arm_2D 
    env = Arm_2D()
    #env = DictGymWrapper(env,goal_state_key='qpos', acheived_state_key='qgoal',keys=['qpos'])
    env = GymWrapper(env,keys=['qpos'])
    import pdb;pdb.set_trace()
