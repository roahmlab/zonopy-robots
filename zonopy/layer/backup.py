import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0

# batch

def gen_grad_RTS_2D_Layer(link_zonos,joint_axes,n_links,n_obs,params):
    jrs_tensor = preload_batch_JRS_trig()
    dimension = 2
    n_timesteps = 100
    ka_0 = torch.zeros(n_links)
    PI_vel = torch.tensor(torch.pi-1e-6)
    zono_order=40
    g_ka = torch.pi/24
    class grad_RTS_2D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx,lambd,observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            ctx.lambd =lambd.clone().reshape(-1,n_links).to(dtype=torch.get_default_dtype())             
            #observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=torch.get_default_dtype())
            ka = g_ka*ctx.lambd
            
            n_batches = observation.shape[0]
            ctx.qpos = observation[:,:n_links]
            ctx.qvel = observation[:,n_links:2*n_links]
            obstacle_pos = observation[:,-4*n_obs:-2*n_obs]
            obstacle_size = observation[:,-2*n_obs:]
            qgoal = ctx.qpos + ctx.qvel*T_PLAN + 0.5*ka*T_PLAN**2

            #g_ka = torch.maximum(PI/24,abs(qvel/3))

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor,ctx.qpos,ctx.qvel,joint_axes)
            FO_link,_,_ = forward_occupancy(R_trig,link_zonos,params)
            
            As = [[] for _ in range(n_links)]
            bs = [[] for _ in range(n_links)]

            lambda_to_slc = ctx.lambd.reshape(n_batches,1,dimension).repeat(1,n_timesteps,1)
            
            #unsafe_flag = torch.zeros(n_batches) 
            unsafe_flag = (abs(ctx.qvel+ctx.lambd*g_ka*T_PLAN)>PI_vel).any(-1)#NOTE: this might not work on gpu, velocity lim check
            for j in range(n_links):
                FO_link[j] = FO_link[j].project([0,1]) 
                c_k = FO_link[j].center_slice_all_dep(lambda_to_slc).unsqueeze(-1) # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:,2*o:2*(o+1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:,2*o:2*(o+1)])),-2).unsqueeze(-3).repeat(1,n_timesteps,1,1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension                     
                    As[j].append(A_temp)
                    bs[j].append(b_temp)
                    unsafe_flag += (torch.max((A_temp@c_k).squeeze(-1)-b_temp,-1)[0]<1e-6).any(-1)  #NOTE: this might not work on gpu FOR, safety check

            M_obs = n_timesteps*n_links*n_obs
            M = M_obs+2*n_links
            ctx.flags = -torch.ones(n_batches,dtype=int) # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            unsafe_flag += True
            for i in unsafe_flag.nonzero().reshape(-1):
                class nlp_setup():
                    x_prev = np.zeros(n_links)*np.nan
                    def objective(nlp,x):
                        qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*g_ka*x*T_PLAN**2
                        return torch.sum(wrap_to_pi(qplan-qgoal[i])**2)

                    def gradient(nlp,x):
                        qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*g_ka*x*T_PLAN**2
                        return (g_ka*T_PLAN**2*wrap_to_pi(qplan-qgoal[i])).numpy()

                    def constraints(nlp,x): 
                        ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                        if (nlp.x_prev!=x).any():      
                            cons_obs = torch.zeros(M)                   
                            grad_cons_obs = torch.zeros(M,n_links)
                            # velocity min max constraints
                            possible_max_min_q_dot = torch.vstack((ctx.qvel[i],ctx.qvel[i]+g_ka*x*T_PLAN,torch.zeros_like(ctx.qvel[i])))
                            q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                            q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                            grad_q_max = torch.diag(T_PLAN*(q_dot_max_idx%2))
                            grad_q_min = torch.diag(T_PLAN*(q_dot_min_idx%2))
                            cons_obs[-2*n_links:] = torch.hstack((q_dot_max,q_dot_min))
                            grad_cons_obs[-2*n_links:] = torch.vstack((grad_q_max,grad_q_min))
                            # velocity min max constraints 
                            for j in range(n_links):
                                c_k = FO_link[j][i].center_slice_all_dep(ka)
                                grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                                for o in range(n_obs):
                                    cons, ind = torch.max((As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i],-1) # shape: n_timsteps, SAFE if >=1e-6
                                    grad_cons = (As[j][o][i].gather(-2,ind.reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                    cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = cons
                                    grad_cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = grad_cons
                            nlp.cons_obs = cons_obs.numpy()
                            nlp.grad_cons_obs = grad_cons_obs.numpy()
                            nlp.x_prev = np.copy(x)                
                        return nlp.cons_obs

                    def jacobian(nlp,x):
                        ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                        if (nlp.x_prev!=x).any():                
                            cons_obs = torch.zeros(M)   
                            grad_cons_obs = torch.zeros(M,n_links)
                            # velocity min max constraints
                            possible_max_min_q_dot = torch.vstack((ctx.qvel[i],ctx.qvel[i]+g_ka*x*T_PLAN,torch.zeros_like(ctx.qvel[i])))
                            q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
                            q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
                            grad_q_max = torch.diag(T_PLAN*(q_dot_max_idx%2))
                            grad_q_min = torch.diag(T_PLAN*(q_dot_min_idx%2))
                            cons_obs[-2*n_links:] = torch.hstack((q_dot_max,q_dot_min))
                            grad_cons_obs[-2*n_links:] = torch.vstack((grad_q_max,grad_q_min))
                            # velocity min max constraints 
                            for j in range(n_links):
                                c_k = FO_link[j][i].center_slice_all_dep(ka)
                                grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                                for o in range(n_obs):
                                    cons, ind = torch.max((As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i],-1) # shape: n_timsteps, SAFE if >=1e-6
                                    grad_cons = (As[j][o][i].gather(-2,ind.reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                    cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = cons
                                    grad_cons_obs[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = grad_cons
                            nlp.cons_obs = cons_obs.numpy()
                            nlp.grad_cons_obs = grad_cons_obs.numpy()
                            nlp.x_prev = np.copy(x)                   
                        return nlp.grad_cons_obs
                    
                    def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                            d_norm, regularization_size, alpha_du, alpha_pr,
                            ls_trials):
                        pass
                
                NLP = cyipopt.problem(
                n = n_links,
                m = M,
                problem_obj=nlp_setup(),
                lb = [-1]*n_links,
                ub = [1]*n_links,
                cl = [1e-6]*M_obs+[-1e20]*n_links+[-torch.pi+1e-6]*n_links,
                cu = [1e20]*M_obs+[torch.pi-1e-6]*n_links+[1e20]*n_links,
                )
                NLP.addOption('sb', 'yes')
                NLP.addOption('print_level', 0)
                
                k_opt, info = NLP.solve(ka_0)


                NLP =nlp_setup()
                
                import pdb;pdb.set_trace()
                # NOTE: for training, dont care about fail-safe
                if info['status'] == 0:
                    ctx.lambd[i] = torch.tensor(k_opt,dtype = torch.get_default_dtype())
                    ctx.flags[i] = 0
                else:
                    ctx.flags[i] = 1

                '''
                # NOTE: for training, dont care about fail-safe
                if info['status'] != 0:
                    ka[i] = -qvel[i]/(T_FULL-T_PLAN)
                    flags[i]=1
                else:
                    ka[i] = torch.tensor(k_opt,dtype = torch.get_default_dtype())
                    flags[i]=0
                '''
            #print(f'rts pass: {flags}')
            return ctx.lambd, FO_link, ctx.flags, info

        @staticmethod
        def backward(ctx,*grad_ouput):
            direction = grad_ouput[0]
            rts_pass = (ctx.flags == 0).reshape(-1,1)
            k_lim = (abs(ctx.lambd)>=1-1e-6)
            vel_lim = (abs(ctx.qvel+ctx.lambd*g_ka*T_PLAN)>PI_vel-1e-6)
            strongly_active = rts_pass*(k_lim+vel_lim)
            grad_lambd = (direction*(~strongly_active)).reshape(ctx.lambd_shape)
            return (grad_lambd,torch.zeros(ctx.obs_shape))
    return grad_RTS_2D_Layer.apply
