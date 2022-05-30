import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection

T_PLAN, T_FULL = 0.5, 1

class Arm_2D:
    def __init__(self,n_links=2,n_obs=1,T_len=50,intermediate = True):
        self.dimension = 2
        self.n_links = n_links
        self.n_obs = n_obs
        self.link_zonos = [zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]])).to_polyZonotope()]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.intermediate = intermediate
        self.PI = torch.tensor(torch.pi)
        if intermediate:
            self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len)+1:] - T_PLAN

        self.reset()
    def reset(self):
        self.qpos = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zeros(self.n_links)
        self.qgoal = torch.rand(self.n_links)*2*torch.pi - torch.pi
        self.fail_safe_count = 0
        if self.intermediate:
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
            self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links)            

        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3), torch.zeros(3)       
        Rg, Pg = torch.eye(3), torch.zeros(3)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link = (Ri@self.link_zonos[j]+Pi).to_zonotope()
            link_init.append(link)
            link = (Rg@self.link_zonos[j]+Pg).to_zonotope()
            link_goal.append(link)

        for _ in range(self.n_obs):
            while True:
                obs_pos = torch.rand(2)*2*self.n_links-self.n_links
                obs = torch.hstack((torch.vstack((obs_pos,0.1*torch.eye(2))),torch.zeros(3,1)))
                obs = zp.zonotope(obs)
                safe_flag = True
                for j in range(self.n_links):
                    buff = link_init[j]-obs
                    A,b = buff.project([0,1]).polytope()
                    if max(A@torch.zeros(2)-b) < 1e-6:
                        safe_flag = False
                        break
                    buff = link_goal[j]-obs
                    A,b = buff.project([0,1]).polytope()
                    if max(A@torch.zeros(2)-b) < 1e-6:
                        safe_flag = False
                        break

                if safe_flag:
                    self.obs_zonos.append(obs)
                    break
        self.render_flag = True
        self.done = False

    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos
        self.qvel = qvel
        self.qgoal = qgoal
        self.fail_safe_count = 0
        if self.intermediate:
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
            self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links)            
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3), torch.zeros(3)       
        Rg, Pg = torch.eye(3), torch.zeros(3)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link = (Ri@self.link_zonos[j]+Pi).to_zonotope()
            link_init.append(link)
            link = (Rg@self.link_zonos[j]+Pg).to_zonotope()
            link_goal.append(link)

        for pos in obs_pos:
            obs = torch.hstack((torch.vstack((pos,0.1*torch.eye(2))),torch.zeros(3,1)))
            obs = zp.zonotope(obs)
            for j in range(self.n_links):
                buff = link_init[j]-obs
                A,b = buff.project([0,1]).polytope()
                if max(A@torch.zeros(2)-b) < 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                A,b = buff.project([0,1]).polytope()
                if max(A@torch.zeros(2)-b) < 1e-6:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
            self.obs_zonos.append(obs)

        self.render_flag = True
        self.done = False

    def step(self,ka,safe=0):
        self.safe = safe == 0
        self.ka = ka
        if self.intermediate:
            if self.safe:
                self.fail_safe_count = 0
                # to peak
                self.qpos_to_peak = self.qpos + torch.outer(self.t_to_peak,self.qvel) + .5*torch.outer(self.t_to_peak**2,ka)
                self.qvel_to_peak = self.qvel + torch.outer(self.t_to_peak,ka)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                #to stop
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_to_brake = self.qpos + torch.outer(self.t_to_brake,self.qvel) + .5*torch.outer(self.t_to_brake**2,bracking_accel)
                self.qvel_to_brake = self.qvel + torch.outer(self.t_to_brake,bracking_accel)
            else:
                self.fail_safe_count +=1
                self.qpos_to_peak = torch.clone(self.qpos_to_brake)
                self.qvel_to_peak = torch.clone(self.qvel_to_brake)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(self.T_len,1)
                self.qvel_to_brake = torch.zeros(self.T_len,self.n_links)
        else:
            if self.safe:
                self.fail_safe_count = 0
                self.qpos += self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
                self.qvel += ka*T_PLAN
                self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
                self.qvel_brake = torch.zeros(self.n_links)
            else:
                self.fail_safe_count +=1
                self.qpos = torch.clone(self.qpos_brake)
                self.qvel = torch.clone(self.qvel_brake) 
        goal_distance = torch.linalg.norm(self.qpos_to_peak-self.qgoal,dim=1)
        self.done = goal_distance.min() < 0.05
        if self.done:
            self.until_goal = goal_distance.argmin()
        return self.done


    def render(self,FO_link=None):
        if self.render_flag:
            plt.ion()
            self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8])
            self.ax = self.fig.gca()
            self.render_flag = False
            self.FO_patches = self.ax.add_collection(PatchCollection([]))
            self.link_patches = self.ax.add_collection(PatchCollection([]))
            one_time_patches = []
            for o in range(self.n_obs):
                one_time_patches.append(self.obs_zonos[o].polygon_patch(edgecolor='red',facecolor='red'))
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3), torch.zeros(3)            
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='gray',facecolor='gray')
                one_time_patches.append(link_patch)
            self.ax.add_collection(PatchCollection(one_time_patches, match_original=True))
        
        
        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count != 1:
                g_ka = torch.minimum(torch.maximum(self.PI/24,abs(self.qvel/3)),self.PI/3)
                self.FO_patches.remove()
                for j in range(self.n_links): 
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1))
                    for t in range(100): 
                        FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor='green')
                        FO_patches.append(FO_patch)
                self.FO_patches = PatchCollection(FO_patches, match_original=True)
                self.ax.add_collection(self.FO_patches)            

        if self.intermediate:
            R_q = self.rot(self.qpos_to_peak)
            if not self.done:
                time_steps = int(T_PLAN/T_FULL*self.T_len)
            else:
                time_steps = self.until_goal
            for t in range(time_steps):
                R, P = torch.eye(3), torch.zeros(3)
                link_patches = []
                self.link_patches.remove()
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[t,j]
                    link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='blue',facecolor='blue')
                    link_patches.append(link_patch)            
                self.link_patches = PatchCollection(link_patches, match_original=True)
                self.ax.add_collection(self.link_patches)
                ax_scale = 1.2
                axis_lim = ax_scale*self.n_links
                plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        else:
            R_q = self.rot()
            R, P = torch.eye(3), torch.zeros(3)
            link_patches = []
            self.link_patches.remove()
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_patch = (R@self.link_zonos[j]+P).to_zonotope().polygon_patch(edgecolor='blue',facecolor='blue')
                link_patches.append(link_patch)
            self.link_patches = PatchCollection(link_patches, match_original=True)
            self.ax.add_collection(self.link_patches)
            ax_scale = 1.2
            axis_lim = ax_scale*self.n_links
            plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def rot(self,q=None):
        if q is None:
            q = self.qpos
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w


class Batch_Arm_2D:
    def __init__(self,n_links=2,n_obs=1,n_batches=4):
        self.n_batches = n_batches
        self.n_links = n_links
        self.n_obs = n_obs
        self.link_zonos = [zp.zonotope(torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]])).to_polyZonotope()]*n_links 
        self.P0 = [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1)
        self.R0 = [torch.eye(3)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links)
        self.fig_scale = 1
        self.reset()
        self.get_plot_grid_size()

    def reset(self):
        self.qpos = torch.rand(self.n_batches,self.n_links)*2*torch.pi - torch.pi
        self.qvel = torch.zeros(self.n_batches,self.n_links)
        self.qgoal = torch.rand(self.n_batches,self.n_links)*2*torch.pi - torch.pi
        self.break_qpos = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
        self.break_qvel = torch.zeros(self.n_batches,self.n_links)

        self.obs_zonos = []
        for _ in range(self.n_obs):
            obs = torch.tensor([[0.0,0,0],[0.1,0,0],[0,0.1,0]]).repeat(self.n_batches,1,1)
            obs[:,0,:2] = torch.rand(self.n_batches,2)*2*self.n_links-self.n_links
            obs = zp.batchZonotope(obs)
            #if SAFE:
            self.obs_zonos.append(obs)
        self.render_flag = True
        
    def step(self,ka):
        self.qpos += self.qvel*T_PLAN + 0.5*ka*T_PLAN**2
        self.qvel += ka*T_PLAN
        self.break_qpos = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
        self.break_qvel = 0

    def render(self,FO_link=None):
        if self.render_flag:
            plt.ion()
            self.fig, self.axs = plt.subplots(self.plot_grid_size[0],self.plot_grid_size[1],figsize=[self.plot_grid_size[1]*6.4/2,self.plot_grid_size[0]*4.8/2])
            self.render_flag = False
            for b, ax in enumerate(self.axs.flat):
                for o in range(self.n_obs):
                    self.obs_zonos[o][b].plot(ax=ax, edgecolor='red',facecolor='red')
            self.link_patches = [[[] for _ in range(self.n_links)] for _ in range(b+1)]
        
        R_q = self.rot 
        
        for b, ax in enumerate(self.axs.flat):
            R, P = torch.eye(3), torch.zeros(3)  
            #if len(self.link_patches[0]) > 0:
            #ax.clear()
            for j in range(self.n_links):
                if not isinstance(self.link_patches[b][j],list):
                    self.link_patches[b][j].remove()
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[b,j]
                self.link_patches[b][j] = (R@self.link_zonos[j]+P).to_zonotope().plot(ax=ax, edgecolor='blue',facecolor='blue')
            ax_scale = 1.2
            axis_lim = ax_scale*self.n_links
            ax.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()        
        
    def get_plot_grid_size(self):
        if self.n_batches in (1,2,3):
            self.plot_grid_size = (1, self.n_batches)
        elif self.n_batches < 9:
            self.plot_grid_size = (2, min(self.n_batches//2,3))
        else:
            self.plot_grid_size = (3,3)
            
    @property
    def rot(self):
        w = torch.tensor([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0.0]]])
        w = (w@self.joint_axes.T).transpose(0,-1)
        q = self.qpos.reshape(self.n_batches,self.n_links,1,1)
        return torch.eye(3) + torch.sin(q)*w + (1-torch.cos(q))*w@w

if __name__ == '__main__':

    env = Arm_2D()
    #from zonopy.optimize.armtd import ARMTD_planner
    #planner = ARMTD_planner(env)
    for _ in range(50):
        #ka, flag = planner.plan(env.qpos,env.qvel,env.qgoal,env.obs_zonos,torch.zeros(2))

        env.step(torch.rand(2))
        env.render()
    '''
    env = Batch_Arm_2D()
    for _ in range(50):
        env.step(torch.rand(env.n_batches,2))
        env.render()
    '''    