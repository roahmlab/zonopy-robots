from zonopy.environments.wrappers.gym_wrapper import GymWrapper
from zonopy.environments.wrappers.utils import dict_torch2np
from gym import spaces
import numpy as np
import torch
# Custom Robosuite wrapper which creates a DictGym env instead of a regular one. For use with HER
class DictGymWrapper(GymWrapper):
    def __init__(self, env, keys, goal_state_key, acheived_state_key):
        # Run super method
        temp_keys = keys+goal_state_key+acheived_state_key
        super().__init__(env=env, keys=temp_keys)
        self.goal_state_key = goal_state_key
        self.acheived_state_key = acheived_state_key
        self.keys = keys
        obs = self.env.get_observations()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs,goal_state_key).shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs,acheived_state_key).shape, dtype="float32"
                ),
                observation=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs).shape, dtype="float32"
                ),
            )
        )
    
    def _flatten_obs(self, obs_dict, keys=None):
        if keys is None:
            keys = self.keys
        ob_lst = []
        for key in keys:
            if key in obs_dict:
                ob_lst.append(obs_dict[key].numpy().astype(float).flatten())
        return np.concatenate(ob_lst)

    def _create_obs_dict(self, obs):
        obs_dict = {
            "observation": self._flatten_obs(obs),
            "achieved_goal": self._flatten_obs(obs,self.acheived_state_key),
            "desired_goal": self._flatten_obs(obs,self.goal_state_key),
        }
        return obs_dict

    def reset(self):
        obs = self.env.reset()
        return self._create_obs_dict(obs)

    def step(self, action, *args, **kwargs):
        obs, reward, done, info = self.env.step(torch.as_tensor(action,dtype=self.env.dtype), *args, **kwargs)
        info['action_taken'] = action
        return self._create_obs_dict(obs), reward, done, dict_torch2np(info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = []

        for i,info_dict in enumerate(info):
            reward.append(
                float(self.env.get_reward(action = torch.as_tensor(info_dict['action_taken'],dtype=self.env.dtype),
                                        qpos = torch.as_tensor(achieved_goal[i],dtype=self.env.dtype),
                                        qgoal = torch.as_tensor(desired_goal[i],dtype=self.env.dtype),
                                        collision = info_dict['collision'],
                                        )
                                )
            )
        
        return np.array(reward)

class ParallelDictGymWrapper(DictGymWrapper):
    def __init__(self, env, keys, goal_state_key, acheived_state_key):
        self.num_envs = env.n_envs 
        super().__init__(env=env,keys=keys,goal_state_key=goal_state_key,acheived_state_key=acheived_state_key)
        obs = self.env.get_observations()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs,goal_state_key).shape[1:], dtype="float32"
                ),
                achieved_goal=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs,acheived_state_key).shape[1:], dtype="float32"
                ),
                observation=spaces.Box(
                -np.inf, np.inf, shape=self._flatten_obs(obs).shape[1:], dtype="float32"
                ),
            )
        )
        

    def _setup_observation_space(self):
        obs = self.env.get_observations()
        self.modality_dims = {key: tuple(obs[key].shape[1:]) for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.shape[1]
        obs = self.observation_spec()

    def _flatten_obs(self, obs_dict, keys=None):
        if keys is None:
            keys = self.keys
        ob_lst = []
        for key in keys:
            if key in obs_dict:
                ob_lst.append(obs_dict[key].numpy().astype(float).reshape(self.n_envs,-1))
        return np.hstack(ob_lst)

    def step(self, action, *args, **kwargs):
        if len(args) > 0:
            flag = torch.tensor(args[0], dtype=int,device=self.env.device)
        else:
            flag = torch.zeros(self.num_envs,dtype=int,device=self.env.device)
        ob_dicts, rewards, dones, infos = self.env.step(torch.as_tensor(action,dtype=self.env.dtype), flag)
        for b in range(self.n_envs):
            infos[b]['action_taken'] = action[b]
            for key in infos[b].keys():
                if isinstance(infos[b][key],torch.Tensor):
                    infos[b][key] = infos[b][key].numpy().astype(float)
                elif isinstance(infos[b][key],torch.Tensor) and isinstance(infos[b][key][0],torch.Tensor):
                    infos[b][key] = [el.numpy().astype(float) for el in infos[b][key]]
        return self._create_obs_dict(ob_dicts), rewards.numpy(), dones.numpy(), infos


    def compute_reward(self, achieved_goal, desired_goal, infos):
        action_taken = []
        collision = []
        safe_flag = []
        for info in infos:
            action_taken.append(info['action_taken'])
            collision.append(info['collision'])
            safe_flag.append(info['safe_flag'])

        action_taken = torch.as_tensor(np.vstack(action_taken),dtype=self.env.dtype)
        collision = torch.as_tensor(np.hstack(collision),dtype=self.env.dtype)
        safe_flag = torch.as_tensor(np.hstack(safe_flag),dtype=self.env.dtype)

        rewards = self.env.get_reward(action = action_taken,
                            qpos = torch.as_tensor(achieved_goal,dtype=self.env.dtype),
                            qgoal = torch.as_tensor(desired_goal,dtype=self.env.dtype),
                            collision = collision,
                            safe = safe_flag,
                            )
        return rewards.numpy()
        
    def get_attr(self,attr_name,indices=None):
        indices = self._get_indices(indices) 
        attr = getattr(self.env,attr_name)
        if isinstance(attr,torch.Tensor) and attr.shape[0] == self.num_envs:
            return [attr[i] for i in indices]
        else:
            return [attr for _ in indices] 
    
    def set_attr(self,attr_name,value,indices=None):
        indices = self._get_indices(indices) 
        attr = getattr(self.env,attr_name)
        if isinstance(attr,torch.Tensor) and attr.shape[0] == self.num_envs:
            for i in indices:
                attr[i] = value[i]
        else:
            attr = value 
        setattr(self.env,attr_name,attr)

    def env_method(self,method_name,*method_args,indices=None,**method_kwargs):
        indices = self._get_indices(indices) 
        if method_name == 'compute_reward':
            output = self.compute_reward(*method_args, **method_kwargs)
        else:
            output = getattr(self.env,method_name)(*method_args, **method_kwargs)
            
        if (isinstance(output,torch.Tensor) and output.shape[0] == self.num_envs):
            return [output[i] for i in indices]
        else:
            return [output for _ in indices]

    def _get_indices(self,indices):
        if indices is None:
            indices - range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices] 
        return indices

if __name__ == '__main__':
    from zonopy.environments.wrappers.wrapper import Wrapper
    from zonopy.environments.arm_2d import Arm_2D 
    env = Arm_2D()
    #env = DictGymWrapper(env,goal_state_key='qpos', acheived_state_key='qgoal',keys=['qpos'])
    env = GymWrapper(env,keys=['qpos'])
    import pdb;pdb.set_trace()

