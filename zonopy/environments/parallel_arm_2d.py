import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Parallel_Arm_2D:
    def __init__(self,
            n_envs = 1, # number of environments
            n_links = 2, # number of links
            n_obs = 1, # number of obstacles
            T_len = 50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.05, # goal threshold
            hyp_effort = 1.0, # hyperpara
            hyp_dist_to_goal = 1.0,
            hyp_collision = -200,
            hyp_success = 50,
            hyp_fail_safe = - 1,
            reward_shaping=True,
            max_episode_steps = 100,
            dtype= None,
            device = None
            ):
        self.n_envs = n_envs

        self.dimension = 2
        self.n_links = n_links
        self.n_obs = n_obs
        link_Z = torch.tensor([[[0.5, 0, 0],[0.5,0,0],[0,0.01,0]]]).repeat(n_envs,1,1)
        self.link_zonos = [zp.batchZonotope(link_Z)]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.interpolate = interpolate
        self.PI = torch.tensor(torch.pi)

        if interpolate:
            self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1).reshape(-1,1,1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len):] - T_PLAN
        
        self.obs_buffer_length = torch.tensor([0.001,0.001])
        self.obstacle_config = {'side_length':0.1*torch.eye(2).unsqueeze(0).repeat(n_envs,1,1), 'zero_pad': torch.zeros(n_envs,3,1)}
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
        self.collision_threshold = collision_threshold
        
        self.goal_threshold = goal_threshold
        self.hyp_effort = hyp_effort
        self.hyp_dist_to_goal = hyp_dist_to_goal
        self.hyp_collision = hyp_collision
        self.hyp_success = hyp_success
        self.hyp_fail_safe = hyp_fail_safe
        self.reward_shaping = reward_shaping
        self.discount = 1

        self.fig = None
        self.render_flag = True

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = torch.zeros(self.n_envs)

        self.get_plot_grid_size()
        self.reset()
    

    def __reset(self,idx):
        n_envs = idx.sum()
        self.qpos[idx] = torch.rand(n_envs,self.n_links)*2*torch.pi - torch.pi
        self.qpos_int[idx] = self.qpos[idx]
        self.qvel[idx] = torch.zeros(n_envs,self.n_links)
        self.qpos_prev[idx] = self.qpos[idx]
        self.qvel_prev[idx] = self.qvel[idx]
        self.qgoal[idx] = torch.rand(n_envs,self.n_links)*2*torch.pi - torch.pi

        if self.interpolate:
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1 
            self.qpos_to_brake[:,idx] = self.qpos[idx].unsqueeze(0).repeat(T_len_to_brake,1,1) 
            self.qvel_to_brake[:,idx] = torch.zeros(T_len_to_brake,n_envs,self.n_links) 
        else:
            self.qpos_brake[idx] = self.qpos[idx] + 0.5*self.qvel[idx]*(T_FULL-T_PLAN)
            self.qvel_brake[idx] = torch.zeros(n_envs,self.n_links) 

        R_qi = self.rot(self.qpos[idx])
        R_qg = self.rot(self.qgoal[idx])
        Ri, Pi = torch.eye(3), torch.zeros(3) 
        Rg, Pg = torch.eye(3), torch.zeros(3)   
        link_init, link_goal = [], []
        for j in range(self.n_links):    
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            
            link_zonos_idx = self.link_zonos[j][idx]
            link = Ri@link_zonos_idx+Pi
            link_init.append(link)
            link = Rg@link_zonos_idx+Pg
            link_goal.append(link)

        idx_nonzero = idx.nonzero().reshape(-1)
        for o in range(self.n_obs):
            safe_flag = torch.zeros(n_envs,dtype=bool)
            while True:
                obs_z, safe_idx = self.obstacle_sample(link_init,link_goal,~safe_flag)
                self.obs_zonos[o].Z[idx_nonzero[safe_idx]] = obs_z 
                safe_flag += safe_idx 
                if safe_flag.all():
                    break
        
        self.fail_safe_count[idx] = 0
        #self.done[idx] = False
        #self.collision[idx] = False
        self._elapsed_steps[idx] = 0
        self.reward_com[idx] = 0

    def reset(self):
        self.qpos = torch.rand(self.n_envs,self.n_links)*2*torch.pi - torch.pi
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_envs,self.n_links)
        self.qvel_int = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = torch.rand(self.n_envs,self.n_links)*2*torch.pi - torch.pi

        if self.interpolate:
            T_len_to_peak = int(T_PLAN/T_FULL*self.T_len)+1
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
            self.qpos_to_peak = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1,1)
            self.qvel_to_peak = torch.zeros(T_len_to_peak,self.n_envs,self.n_links)
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1,1)
            self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_envs,self.n_links)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_envs,self.n_links)            

        self.obs_zonos = []
        
        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)
        Ri, Pi = torch.eye(3), torch.zeros(3)       
        Rg, Pg = torch.eye(3), torch.zeros(3)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[:,j]
            Rg = Rg@self.R0[j]@R_qg[:,j]
            
            link = Ri@self.link_zonos[j]+Pi
            link_init.append(link)
            link = Rg@self.link_zonos[j]+Pg
            link_goal.append(link)
        for _ in range(self.n_obs):
            safe_flag = torch.zeros(self.n_envs,dtype=bool)
            obs_Z = torch.zeros(self.n_envs,3,3)
            while True:
                obs_z, safe_idx = self.obstacle_sample(link_init,link_goal,~safe_flag)
                obs_Z[safe_idx] = obs_z 
                safe_flag += safe_idx 
                if safe_flag.all():
                    obs = zp.batchZonotope(obs_Z)
                    self.obs_zonos.append(obs)
                    break
        self.fail_safe_count = torch.zeros(self.n_envs)
        if self.render_flag == False:
            for b in range(self.n_plots):
                self.one_time_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
        self.render_flag = True
        self.done = torch.zeros(self.n_envs,dtype=bool)
        self.collision = torch.zeros(self.n_envs,dtype=bool)
        self._elapsed_steps = torch.zeros(self.n_envs)
        self.reward_com = torch.zeros(self.n_envs)
        return self.get_observations()
        
    def obstacle_sample(self,link_init,link_goal,idx):
        '''
        if idx is None:
            n_envs= self.n_envs
            idx = torch.ones(n_envs,dtype=bool)
        else:
        ''' 
        n_envs = idx.sum()
        r,th = torch.rand(2,n_envs)
        #obs_pos = torch.rand(n_envs,2)*2*self.n_links-self.n_links
        obs_pos = (3/4*self.n_links*r*torch.vstack((torch.cos(2*torch.pi*th),torch.sin(2*torch.pi*th)))).T
        obs_Z = torch.cat((torch.cat((obs_pos.unsqueeze(1),self.obstacle_config['side_length'][:n_envs]),1),self.obstacle_config['zero_pad'][:n_envs]),-1)
        obs = zp.batchZonotope(obs_Z)
        safe_flag = torch.zeros(len(idx),dtype=bool)
        safe_flag[idx] = True
        for j in range(self.n_links):            
            buff = link_init[j][idx]-obs
            _,bi = buff.project([0,1]).polytope()
            buff = link_goal[j][idx]-obs
            _,bg = buff.project([0,1]).polytope()   
            safe_flag[idx] *= ((bi.min(1).values < 1e-6) * (bg.min(1).values < 1e-6)) # Ture for safe envs, -1e-6: more conservative, 1e-6: less conservative

        return obs_Z[safe_flag[idx]], safe_flag

    def step(self,ka,flag=None):
        if flag is None:
            self.step_flag = torch.zeros(self.n_envs)
        else:
            self.step_flag = flag
        self.safe = self.step_flag <= 0
        # -torch.pi<qvel+k*T_PLAN < torch.pi
        # (-torch.pi-qvel)/T_PLAN < k < (torch.pi-qvel)/T_PLAN
        self.ka = ka.clamp((-torch.pi-self.qvel)/T_PLAN,(torch.pi-self.qvel)/T_PLAN) # velocity clamp
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        if self.interpolate:
            unsafe = ~self.safe
            self.fail_safe_count = (unsafe)*(self.fail_safe_count+1)

            safe_qpos = self.qpos[self.safe].unsqueeze(0)
            safe_qvel = self.qvel[self.safe].unsqueeze(0)
            safe_action = self.ka[self.safe].unsqueeze(0)
            self.qpos_to_peak[:,self.safe] = wrap_to_pi(safe_qpos + self.t_to_peak*safe_qvel + .5*(self.t_to_peak**2)*safe_action)
            self.qvel_to_peak[:,self.safe] = safe_qvel + self.t_to_peak*safe_action
            self.qpos_to_peak[:,unsafe] = self.qpos_to_brake[:,unsafe]
            self.qvel_to_peak[:,unsafe] = self.qvel_to_brake[:,unsafe]
    
            self.qpos = self.qpos_to_peak[-1]
            self.qvel = self.qvel_to_peak[-1]

            qpos = self.qpos.unsqueeze(0)
            qvel = self.qvel.unsqueeze(0)
            bracking_accel = (0 - qvel)/(T_FULL - T_PLAN)
            self.qpos_to_brake = wrap_to_pi(qpos + self.t_to_brake*qvel + .5*(self.t_to_brake**2)*bracking_accel)
            self.qvel_to_brake = qvel + self.t_to_brake*bracking_accel 
            self.collision = self.collision_check(torch.cat((self.qpos_to_peak,self.qpos_to_brake[1:]),0))

        else:
            unsafe = ~self.safe
            self.fail_safe_count = (unsafe)*(self.fail_safe_count+1)
            self.qpos[self.safe] += wrap_to_pi(self.qvel[self.safe]*T_PLAN + 0.5*self.ka[self.safe]*T_PLAN**2)
            self.qvel[self.safe] += self.ka[self.safe]*T_PLAN
            self.qpos_brake[self.safe] = wrap_to_pi(self.qpos[self.safe] + 0.5*self.qvel[self.safe]*(T_FULL-T_PLAN))
            self.qvel_brake[self.safe] = 0
            self.qpos[unsafe] = self.qpos_brake[unsafe]
            self.qvel[unsafe] = self.qvel_brake[unsafe] 
            self.collision_check(self.qpos)
            self.collision = self.collision_check(torch.cat((self.qpos.unsqueeze(0),self.qpos_brake.unsqueeze(0)),0))
                    
        self._elapsed_steps += 1
        
        self.reward = self.get_reward(ka) # NOTE: should it be ka or self.ka ??
        self.reward_com *= self.discount
        self.reward_com += self.reward
        self.done = self.success + self.collision
        observations = self.get_observations()
        infos = self.get_info()
        print(f'collision: {self.collision}')
        print(f'success: {self.success}')
        if self.done.sum()>0:
            self.__reset(self.done)
        return observations, self.reward, self.done, infos

    def get_info(self):
        
        infos = []
        for idx in range(self.n_envs):
            info = {
                'is_success':bool(self.success[idx]),
                'collision':bool(self.collision[idx]),
                'safe_flag':bool(self.safe[idx]),
                'step_flag':int(self.step_flag[idx])
                }
            if self.collision[idx]:
                collision_info = {
                    'qpos_init':self.qpos_int[idx],
                    'qvel_int':torch.zeros(self.n_links),
                    'obs_pos':[self.obs_zonos[o].center[idx,:2] for o in range(self.n_obs)],
                    'qgoal':self.qgoal[idx]
                }
                info['collision_info'] = collision_info
            if self._elapsed_steps[idx] >= self._max_episode_steps:
                info["TimeLimit.truncated"] = not self.done[idx]
                self.done[idx] = True       
            info['episode'] = {"r":float(self.reward_com[idx]),"l":int(self._elapsed_steps[idx])}
            infos.append(info)
        return tuple(infos)

    def get_observations(self):
        observation = {'qpos':torch.clone(self.qpos),'qvel':torch.clone(self.qvel),'qgoal':torch.clone(self.qgoal)}
        
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.cat([self.obs_zonos[o].center[:,:2].unsqueeze(1) for o in range(self.n_obs)],1)
            observation['obstacle_size'] = torch.cat([self.obs_zonos[o].generators[:,[0,1],[0,1]].unsqueeze(1) for o in range(self.n_obs)],1)
        return observation

    def collision_check(self,qs):
        unsafe = torch.zeros(self.n_envs,dtype=bool)
        if self.check_collision:
            R_q = self.rot(qs)
            if len(R_q.shape) == 5:
                time_steps = R_q.shape[0]
                R, P = torch.eye(3), torch.zeros(3)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,:,j]
                    link =zp.batchZonotope(self.link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1,1))
                    link = R@link+P
                    for o in range(self.n_obs):
                        buff = torch.cat(((link.center - self.obs_zonos[o].center).unsqueeze(-2),link.generators,self.obs_zonos[o].generators.unsqueeze(0).repeat(time_steps,1,1,1)),-2)
                        _,b = zp.batchZonotope(buff).project([0,1]).polytope()
                        unsafe += (b.min(dim=-1)[0]>1e-6).any(dim=0)
                
                        
            else:
                time_steps = 1
                R, P = torch.eye(3), torch.zeros(3)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,j]
                    link = R@self.link_zonos[j]+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.project([0,1]).polytope()
                        unsafe += b.min(dim=-1)[0] > 1e-6
        return unsafe    

    def get_reward(self, action, qpos=None, qgoal=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None or qgoal is None:
            qpos = self.qpos
            qgoal = self.qgoal
        
        self.goal_dist = torch.linalg.norm(wrap_to_pi(qpos-qgoal),dim=-1)
        self.success = self.goal_dist < self.goal_threshold 
        success = self.success.to(dtype=torch.get_default_dtype())
        
        reward = 0.0

        # Return the sparse reward if using sparse_rewards
        if not self.reward_shaping:
            reward += self.hyp_collision * self.collision
            reward += success - 1 + self.hyp_success * success
            return reward

        # otherwise continue to calculate the dense reward
        # reward for position term
        reward -= self.hyp_dist_to_goal * self.goal_dist
        # reward for effort
        reward -= self.hyp_effort * torch.linalg.norm(action,dim=-1)
        # Add collision if needed
        reward += self.hyp_collision * self.collision
        # Add fail-safe if needed
        reward += self.hyp_fail_safe * (1 - self.safe.to(dtype=torch.get_default_dtype()))
        # Add success if wanted
        reward += self.hyp_success * success

        return reward       

    def render(self,FO_link=None):
        if self.render_flag:
            if self.fig is None:
                plt.ion()
                self.fig, self.axs = plt.subplots(self.plot_grid_size[0],self.plot_grid_size[1],figsize=[self.plot_grid_size[1]*6.4/2,self.plot_grid_size[0]*4.8/2])
            self.render_flag = False
            self.one_time_patches, self.FO_patches, self.link_patches= [], [], []

            R_q = self.rot(self.qgoal[:self.n_plots])
            link_goal_patches = []
            R, P = torch.eye(3), torch.zeros(3)
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[:,j]
                link_goal_patches.append((R@self.link_zonos[j][:self.n_plots]+P).polygon(nan=False).cpu().numpy())

            obs_patches = []
            for o in range(self.n_obs):
                obs_patches.append(self.obs_zonos[o][:self.n_plots].polygon(nan=False).cpu().numpy())
                #patches.Polygon(p,alpha=alpha,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth)
            for b, ax in enumerate(self.axs.flat):
                one_time_patch = []
                for j in range(self.n_links):
                    one_time_patch.append(patches.Polygon(link_goal_patches[j][b],alpha = .5, facecolor='gray',edgecolor='gray',linewidth=.2))
                for o in range(self.n_obs):
                    one_time_patch.append(patches.Polygon(obs_patches[o][b],alpha = .5, facecolor='red',edgecolor='red',linewidth=.2))
                self.one_time_patches.append(PatchCollection(one_time_patch, match_original=True))
                ax.add_collection(self.one_time_patches[b])
                self.FO_patches.append(ax.add_collection(PatchCollection([])))
                self.link_patches.append(ax.add_collection(PatchCollection([])))
        
        '''
        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count != 1:
                g_ka = torch.maximum(self.PI/24,abs(self.qvel_prev/3)) # NOTE: is it correct?
                self.FO_patches.remove()
                for j in range(self.n_links):
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)) 
                    if self.check_collision_FO:
                        c_link_slc = FO_link[j].center_slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1))
                        for o,obs in enumerate(self.obs_zonos):
                            obs_Z = obs.Z[:,:self.dimension].unsqueeze(0).repeat(100,1,1)
                            A, b = zp.batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope()
                            cons, _ = torch.max((A@c_link_slc.unsqueeze(-1)).squeeze(-1) - b,-1)
                            for t in range(100):                            
                                if cons[t] < 1e-6:
                                    color = 'red'
                                else:
                                    color = 'green'
                                FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor=color)
                                FO_patches.append(FO_patch)
                    else:
                        for t in range(100): 
                            FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor='green')
                            FO_patches.append(FO_patch)
                self.FO_patches = PatchCollection(FO_patches, match_original=True)
                self.ax.add_collection(self.FO_patches)            
        '''
        if self.interpolate:
            R_q = self.rot(self.qpos_to_peak[:,:self.n_plots])
            R, P = torch.eye(3), torch.zeros(3)
            link_trace_patches = []
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,:,j]
                link_trace_patches.append((R@self.link_zonos[j][:self.n_plots]+P).polygon(nan=False).cpu().numpy())   
            time_steps = int(T_PLAN/T_FULL*self.T_len) # NOTE
            for t in range(time_steps):
                for b, ax in enumerate(self.axs.flat):
                    self.link_patches[b].remove()
                    link_patch = []
                    for j in range(self.n_links):
                        link_patch.append(patches.Polygon(link_trace_patches[j][t,b],alpha = .5, facecolor='blue',edgecolor='blue',linewidth=.2))
                    self.link_patches[b] = PatchCollection(link_patch, match_original=True)
                    ax.add_collection(self.link_patches[b])
                
                    ax_scale = 1.2
                    axis_lim = ax_scale*self.n_links
                    ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        else:
            R_q = self.rot(self.qpos[:self.n_plots])
            R, P = torch.eye(3), torch.zeros(3)
            link_trace_patches = []
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_trace_patches.append((R@self.link_zonos[j][:self.n_plots]+P).polygon(nan=False).cpu().numpy())            
            
            for b, ax in enumerate(self.axs.flat):
                self.link_patches[b].remove()
                link_patch = []
                for j in range(self.n_links):
                    link_patch.append(patches.Polygon(link_trace_patches[j][b],alpha = .5, facecolor='blue',edgecolor='blue',linewidth=.2))
                self.link_patches[b] = PatchCollection(link_patch, match_original=True)
                ax.add_collection(self.link_patches[b])
                ax_scale = 1.2
                axis_lim = ax_scale*self.n_links
                ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.done.any():
            reset_flag = self.done[:self.n_plots].nonzero().reshape(-1)
        
            R_q = self.rot(self.qgoal[reset_flag])
            link_goal = []
            R, P = torch.eye(3), torch.zeros(3)
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[:,j]
                link_goal.append(R@self.link_zonos[j][reset_flag]+P)

            for idx, b in enumerate(reset_flag.tolist()):
                self.one_time_patches[b].remove()
                self.FO_patches[b].remove()
                self.link_patches[b].remove()
                ax = self.axs.flat[b]
                one_time_patch = []
                for o in range(self.n_obs):
                    one_time_patch.append(self.obs_zonos[o][b].polygon_patch(edgecolor='red',facecolor='red'))
                for j in range(self.n_links):
                    link_patch = link_goal[j][idx].polygon_patch(edgecolor='gray',facecolor='gray')
                    one_time_patch.append(link_patch)
                self.one_time_patches[b] = PatchCollection(one_time_patch, match_original=True)
                ax.add_collection(self.one_time_patches[b])
                self.FO_patches[b] = ax.add_collection(PatchCollection([]))
                self.link_patches[b] = ax.add_collection(PatchCollection([]))  
        
    def get_plot_grid_size(self):
        if self.n_envs in (1,2,3):
            self.plot_grid_size = (1, self.n_envs)
        elif self.n_envs < 9:
            self.plot_grid_size = (2, min(self.n_envs//2,3))
        else:
            self.plot_grid_size = (3,3)
        self.n_plots = self.plot_grid_size[0]*self.plot_grid_size[1]
    def rot(self,q=None):
        if q is None:
            q = self.qpos
        w = torch.tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

    @property
    def action_spec(self):
        pass
    @property
    def action_dim(self):
        pass
    @property 
    def action_space(self):
        pass 
    @property 
    def observation_space(self):
        pass 
    @property 
    def obs_dim(self):
        pass

if __name__ == '__main__':
    import time
    from zonopy.environments.arm_2d import Arm_2D
    n_envs = 3
    env = Parallel_Arm_2D(n_envs=n_envs,interpolate=True)
    env1 = Arm_2D()

    ts = time.time()
    for _ in range(n_envs):
        env1.reset()
    print(f'serial reset: {time.time()-ts}')
    ts = time.time()
    env.reset()
    print(f'parallel reset : {time.time()-ts}')

    for i in range(20):
        observations, rewards, dones, infos = env.step(torch.ones(env.n_envs,env.n_links))
        env.render()
    
    import pdb;pdb.set_trace()