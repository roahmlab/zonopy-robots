# TODO VALIDATE

import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt
#import os
from torch.multiprocessing import Pool
import zonopy as zp

from zonopy.layer.nlp_setup import NlpSetupLocked3D

# torch.multiprocessing.set_start_method('spawn', force=True)
# os.environ['OMP_NUM_THREADS'] = '2'
import time 

T_PLAN, T_FULL = 0.5, 1.0
NUM_PROCESSES = 40

EPS = 1e-6
TOL = 1e-4

def rtd_pass(A, b, FO_link, qpos, qvel, qgoal, n_timesteps, n_links, dof, n_obs_in_frs, n_pos_lim, actual_pos_lim, vel_lim, lim_flag, dimension, g_ka, ka_0, lambd_hat):
    '''
    if 'count_rtd' not in globals():
        global some_obs_solve, some_obs_unsolve, some_obs_limit, no_obs_solve, no_obs_unsolve, no_obs_limit
        global count_rtd
        some_obs_solve, some_obs_unsolve, some_obs_limit = 0, 0, 0
        no_obs_solve, no_obs_unsolve, no_obs_limit = 0, 0, 0
        count_rtd = 0
    '''
    M_obs = n_links * n_timesteps * int(n_obs_in_frs)
    M = M_obs+2*dof+6*n_pos_lim
    nlp_obj = NlpSetupLocked3D(A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,dof,int(n_obs_in_frs),n_pos_lim,actual_pos_lim,vel_lim,lim_flag,dimension,g_ka)
    NLP = cyipopt.Problem(
        n=dof,
        m=M,
        problem_obj=nlp_obj,
        lb = [-1]*dof,
        ub = [1]*dof,
        cl = [-1e20]*M,
        cu = [-EPS]*M,
        )
    NLP.add_option('sb', 'yes')
    NLP.add_option('print_level', 0)
    #NLP.add_option('max_cpu_time', 0.2)
    NLP.add_option('max_iter',15)
    #NLP.add_option('hessian_approximation','limited-memory')
    NLP.add_option('tol', TOL)
    NLP.add_option('linear_solver', 'ma27')
    k_opt, info = NLP.solve(ka_0)


    ###########################################################
    '''
    if info['status'] ==0:
        if n_obs_in_frs > 0:
            some_obs_solve += 1
        else:
            no_obs_solve += 1
    elif info['status'] ==1 or info['status'] ==2:
        if n_obs_in_frs > 0:
            some_obs_unsolve += 1
        else:
            no_obs_unsolve += 1    
    elif info['status'] == -4 or info['status'] == -1:
        if n_obs_in_frs > 0:
            some_obs_limit += 1
        else:
            no_obs_limit += 1
    else:
        import pdb;pdb.set_trace()
    count_rtd += 1 
    #if count_rtd % 10 == 0:
    if True:
        some_obs = some_obs_solve + some_obs_unsolve + some_obs_limit
        no_obs = no_obs_solve + no_obs_unsolve + no_obs_limit

        print('~'*60)
        if some_obs != 0:
            print(f'N of prob with obstacle nearby: {some_obs}')
            print(f'Solve Rate with obstacle nearby: {some_obs_solve/some_obs*100}')
            print(f'Infeasible Rate with obstacle nearby: {some_obs_unsolve/some_obs*100}')
            print(f'Limited Rate with obstacle nearby: {some_obs_limit/some_obs*100}')
        if no_obs != 0:
            print(f'N of prob without obstacle: {no_obs}')
            print(f'Solve Rate without obstacle: {no_obs_solve/no_obs*100}')
            print(f'Infeasible Rate without obstacle: {no_obs_unsolve/no_obs*100}')
            print(f'Limited Rate without obstacle: {no_obs_limit/no_obs*100}')
        print('~'*60)
        if some_obs == 1000 or no_obs == 1000:
            import pdb;pdb.set_trace()
    '''
    ###########################################################


    # NOTE: for training, dont care about fail-safe
    if info['status'] == 0:
        lambd_opt = k_opt.tolist()
        flag = 0
    else:
        lambd_opt = lambd_hat.tolist()
        flag = 1
    return lambd_opt, flag, info

def rot(q,joint_axes):
    dtype, device = q.dtype, q.device
    q = q.reshape(q.shape+(1,1))
    w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]],dtype=dtype,device=device)
    rot_skew_sym = (w@joint_axes.to(dtype=dtype,device=device).T).transpose(0,-1)
    return torch.eye(3,dtype=dtype,device=device) + torch.sin(q)*rot_skew_sym + (1-torch.cos(q))*rot_skew_sym@rot_skew_sym

def gen_RTS_star_Locked_3D_Layer(link_zonos, joint_axes, n_links, n_obs, pos_lim, vel_lim, lim_flag, locked_idx, locked_qpos, params, num_processes=NUM_PROCESSES, dtype = torch.float, device=torch.device('cpu'), multi_process=False):
    jrs_tensor = preload_batch_JRS_trig(dtype=dtype, device=device)
    dimension = 3
    n_timesteps = 100

    PI_vel = torch.tensor(torch.pi - EPS,dtype=dtype, device=device)
    g_ka = torch.pi / 24

    actual_pos_lim = pos_lim[lim_flag]
    n_pos_lim = int(lim_flag.sum().cpu())

    max_combs = 200 
    combs = [torch.combinations(torch.arange(i,device=device),2) for i in range(max_combs+1)]

    dof = n_links - len(locked_idx)
    unlocked_idx = torch.ones(n_links,dtype=bool,device=device)
    unlocked_idx[locked_idx] = False
    unlocked_idx = unlocked_idx.nonzero().reshape(-1)

    locked_joint_axes = []
    unlocked_joint_axes = []
    for j in range(n_links):
        if j in locked_idx:
            locked_joint_axes.append(joint_axes[j])
        else:
            unlocked_joint_axes.append(joint_axes[j])
    if len(locked_idx) > 0:
        locked_rot = list(rot(locked_qpos,torch.vstack(locked_joint_axes)).to(dtype=dtype,device=device).reshape(-1,1,1,3,3).repeat(1,1,n_timesteps,1,1))


    class RTS_star_Locked_3D_Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lambd, observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]

            ctx.lambd_shape = lambd.shape
            lambd = lambd.clone().reshape(-1, dof).to(dtype=dtype,device=device)
            ka = g_ka * lambd

            if isinstance(observation,dict):
                ctx.obs_type = 'dict'
                ctx.obs_shape = observation["observation"].shape 
                n_batches, obs_dim = ctx.obs_shape
                qpos = observation["achieved_goal"].to(dtype=dtype,device=device)
                #qgoal = observation["desired_goal"].to(dtype=dtype,device=device)
                observation_observation = observation["observation"].to(dtype=dtype,device=device)
                qvel = observation_observation[:,:dof]
                obstacle_center = observation_observation[:, obs_dim -6 * n_obs: obs_dim -3 * n_obs].reshape(n_batches,n_obs,1,dimension)
                obstacle_generators = torch.diag_embed(observation_observation[:, obs_dim -3 * n_obs:].reshape(n_batches,n_obs,dimension))


            else:
                ctx.obs_type = 'tensor'

                ctx.obs_shape = observation.shape
                
                # observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
                observation = observation.to(dtype=dtype,device=device)
                

                n_batches, obs_dim = observation.shape
                qpos = observation[:, :dof]
                qvel = observation[:, dof:2 * dof]
                obstacle_center = observation[:, obs_dim -6 * n_obs: obs_dim -3 * n_obs].reshape(n_batches,n_obs,1,dimension)
                obstacle_generators = torch.diag_embed(observation[:, obs_dim -3 * n_obs:].reshape(n_batches,n_obs,dimension))
            
            obs_Z = torch.cat((obstacle_center,obstacle_generators),-2).unsqueeze(-3).repeat(1,1,n_timesteps,1,1)
            
            qgoal = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN ** 2

            Rot = []
            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor, qpos, qvel, unlocked_joint_axes)
            lock_count, unlock_count = 0, 0
            for j in range(n_links):
                if j in locked_idx:
                    Rot.append(locked_rot[lock_count].repeat(n_batches,1,1,1))
                    lock_count += 1
                else:
                    Rot.append(R_trig[unlock_count]) 
                    unlock_count += 1
            batch_FO_link, _, _ = forward_occupancy(Rot, link_zonos, params)

            As = np.zeros((n_batches,n_links),dtype=object)
            bs = np.zeros((n_batches,n_links),dtype=object)

            FO_links_nlp = np.zeros((n_batches,n_links),dtype=object)
            FO_links = np.zeros((n_links,),dtype=object)
            lambda_to_slc = lambd.reshape(n_batches, 1, dof).repeat(1, n_timesteps, 1)

            # pos and vel lim

            # position and velocity constraints
            t_peak_optimum = -qvel/ka # time to optimum of first half traj.
            qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(qpos+qvel*t_peak_optimum+0.5*ka*t_peak_optimum**2).nan_to_num()
            qpos_peak = qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN**2
            qvel_peak = qvel + ka * T_PLAN

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = qpos + 0.5*qvel*(T_FULL+T_PLAN) + 0.5 * ka * T_PLAN * T_FULL

            # qpos+qvel*t_peak_optimum+0.5*ka*t_peak_optimum**2 < ub
            # qpos + qvel * T_PLAN + 0.5 * ka * T_PLAN**2 < ub
            # qpos + 0.5*qvel*(T_FULL+T_PLAN) + 0.5 * ka * T_PLAN * T_FULL < ub

            # ka < (ub - qpos - qvel*t_peak_optimum) / (0.5*t_peak_optimum**2)
            # ka  < (ub - qpos - qvel * T_PLAN) / (0.5 * T_PLAN**2)
            # ka  < (ub - qpos - 0.5*qvel*(T_FULL+T_PLAN)) / (0.5 * T_PLAN * T_FULL)

            qpos_possible_max_min = torch.cat((qpos_peak_optimum.unsqueeze(-2),qpos_peak.unsqueeze(-2),qpos_brake.unsqueeze(-2)),-2)[:,:,lim_flag] 
            qpos_ub = (qpos_possible_max_min - actual_pos_lim[:,0]).reshape(n_batches,-1)
            qpos_lb = (actual_pos_lim[:,1] - qpos_possible_max_min).reshape(n_batches,-1)
            
            unsafe_flag = (abs(qvel_peak)>vel_lim).any(-1) + (qpos_ub>0).any(-1) + (qpos_lb>0).any(-1)
            obs_in_reach_idx = torch.zeros(n_batches,n_obs, dtype=bool,device=device)
            
            lambd0 = lambd.clamp((-PI_vel-qvel)/(g_ka *T_PLAN),(PI_vel-qvel)/(g_ka *T_PLAN)).cpu().numpy()
            
            for j in range(n_links):
                FO_link_temp = batch_FO_link[j]
                c_k = FO_link_temp.center_slice_all_dep(lambda_to_slc).reshape(n_batches,1,n_timesteps,dimension,1)  # FOR, safety check
         
                obs_buff_Grest = zp.batchZonotope(torch.cat((obs_Z,FO_link_temp.Grest.unsqueeze(1).repeat(1,n_obs,1,1,1)),-2))
                A_Grest, b_Grest  = obs_buff_Grest.polytope(combs)
                h_obs = ((A_Grest @ c_k).squeeze(-1) - b_Grest).nan_to_num(-torch.inf)
                unsafe_flag += (torch.max(h_obs, -1)[0] < 1e-6).any(-1).any(1)  # NOTE: this might not work on gpu FOR, safety check
                                
                obs_buff = obs_buff_Grest - zp.batchZonotope(FO_link_temp.Z[FO_link_temp.batch_idx_all+(slice(FO_link_temp.n_dep_gens+1),)].unsqueeze(1).repeat(1,n_obs,1,1,1))
                _, b_obs = obs_buff.reduce(6).polytope(combs)
                
                obs_in_reach_idx += (torch.min(b_obs.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)

                As[-1,j] = A_Grest.cpu().numpy()
                bs[-1,j] = b_Grest.cpu().numpy()
                FO_links_nlp[:,j] = [fo for fo in FO_link_temp.cpu()]
                FO_links[j] = FO_link_temp

            #unsafe_flag = torch.ones(n_batches, dtype=torch.bool)  # NOTE: activate rtd always
            rtd_pass_indices = unsafe_flag.nonzero().reshape(-1).tolist()
            n_problems = len(rtd_pass_indices)
    
            flags = -torch.ones(n_batches, dtype=torch.int, device=device)  # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            infos = [None for _ in range(n_batches)]

            if n_problems > 0:
                qpos_np = qpos.cpu().numpy()
                qvel_np = qvel.cpu().numpy()
                qgoal_np = qgoal.cpu().numpy()
                lambd_np = lambd.cpu().numpy()
                
                actual_pos_lim_np = actual_pos_lim.cpu().numpy()
                vel_lim_np =  vel_lim.cpu().numpy()
                lim_flag_np = lim_flag.cpu().numpy()         

                obs_in_reach_idx_list = obs_in_reach_idx.tolist()
                N_obs_in_frs = obs_in_reach_idx.sum(-1).cpu().numpy()

                if multi_process:
                    for idx in rtd_pass_indices:
                        obs_idx = obs_in_reach_idx_list[idx]
                        for j in range(n_links):
                            As[idx,j] = As[-1,j][idx,obs_idx]
                            bs[idx,j] = bs[-1,j][idx,obs_idx]
                    with Pool(processes=min(num_processes, n_problems)) as pool:
                        results = pool.starmap(
                            rtd_pass,
                            [x for x in
                            zip(As[rtd_pass_indices],
                                bs[rtd_pass_indices],
                                FO_links_nlp[rtd_pass_indices],
                                qpos_np[rtd_pass_indices],
                                qvel_np[rtd_pass_indices],
                                qgoal_np[rtd_pass_indices],
                                [n_timesteps] * n_problems,
                                [n_links] * n_problems,
                                [dof] * n_problems,
                                N_obs_in_frs[rtd_pass_indices], 
                                [n_pos_lim] * n_problems,
                                [actual_pos_lim_np] * n_problems,
                                [vel_lim_np] * n_problems,
                                [lim_flag_np] * n_problems, 
                                [dimension] * n_problems,
                                [g_ka] * n_problems,
                                [lambd0[idx] for idx in rtd_pass_indices],  #[ka_0] * n_problems,
                                lambd_np[rtd_pass_indices]
                            )
                            ]
                        )
                    rtd_lambd_opts, rtd_flags = [], []
                    for idx, res in enumerate(results):
                        rtd_lambd_opts.append(res[0])
                        rtd_flags.append(res[1])
                        infos[rtd_pass_indices[idx]] = res[2]
                    lambd[rtd_pass_indices] = torch.tensor(rtd_lambd_opts,dtype=dtype,device=device)
                    flags[rtd_pass_indices] = torch.tensor(rtd_flags, dtype=flags.dtype, device=device)
                else:
                    rtd_lambd_opts, rtd_flags = [], []                    
                    for idx in rtd_pass_indices:

                        obs_idx = obs_in_reach_idx_list[idx]
                        for j in range(n_links):
                            As[idx,j] = As[-1,j][idx,obs_idx]
                            bs[idx,j] = bs[-1,j][idx,obs_idx]

                        rtd_lambd_opt, rtd_flag, info = rtd_pass(
                                                                As[idx],
                                                                bs[idx],
                                                                FO_links_nlp[idx],
                                                                qpos_np[idx],
                                                                qvel_np[idx],
                                                                qgoal_np[idx],
                                                                n_timesteps,
                                                                n_links,
                                                                dof,
                                                                N_obs_in_frs[idx],
                                                                n_pos_lim,
                                                                actual_pos_lim_np,
                                                                vel_lim_np,
                                                                lim_flag_np, 
                                                                dimension,
                                                                g_ka,
                                                                lambd0[idx],
                                                                lambd_np[idx])
                        infos[idx] = info
                        rtd_lambd_opts.append(rtd_lambd_opt)
                        rtd_flags.append(rtd_flag)
                    lambd[rtd_pass_indices] = torch.tensor(rtd_lambd_opts,dtype=dtype,device=device)
                    flags[rtd_pass_indices] = torch.tensor(rtd_flags, dtype=flags.dtype, device=device)
            return lambd, FO_links, flags, infos

        @staticmethod
        def backward(ctx, *grad_ouput):
            if ctx.obs_type == 'dict':
                return (torch.zeros(ctx.lambd_shape,dtype=dtype,device=device), None)
            else:
                return (torch.zeros(ctx.lambd_shape,dtype=dtype,device=device), torch.zeros(ctx.obs_shape,dtype=dtype,device=device))

    return RTS_star_Locked_3D_Layer.apply


if __name__ == '__main__':
    
    from zonopy.environments.parallel_arm_3d import Parallel_Locked_Arm_3D
    import time
    ##### 0. SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### 1. SET ENVIRONMENT #####
    n_batch = 9
    env = Parallel_Locked_Arm_3D(n_envs = n_batch, n_obs=1, n_plots = 1, locked_idx = [0, 2, 4], locked_qpos = [0, 0, 0], FO_render_freq=0)

    ##### 2. GENERATE RTS* LAYER #####    
    P,R,link_zonos = [],[],[]
    for p,r,l in zip(env.P0,env.R0,env.link_zonos):
        P.append(p.to(device=device,dtype=dtype))
        R.append(r.to(device=device,dtype=dtype))
        link_zonos.append(l.to(device=device,dtype=dtype))
    params = {'n_joints': env.n_links, 'P': P, 'R': R}
    joint_axes = [j for j in env.joint_axes.to(device=device,dtype=dtype)]
    rts_star = gen_RTS_star_Locked_3D_Layer(link_zonos, joint_axes, env.n_links, env.n_obs, env.pos_lim, env.vel_lim, env.lim_flag, env.locked_idx, env.locked_qpos, params,device=device,dtype=dtype)

    ##### 3. RUN RTS* #####
    t_forward = 0
    t_render = 0
    n_steps = 30
    
    print('='*90)
    observation = env.reset()
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].reshape(n_batch,-1) for key in observation.keys()])

        lam = torch.tensor([0.8]*env.dof,device=device,dtype=dtype)
        lam, FO_link, flag, nlp_info = rts_star(torch.vstack(([lam] * n_batch)), observ_temp)
              
        print(f'action: {lam[0]}')
        print(f'flag: {flag[0]}')

        t_elasped = time.time() - ts
        print(f'Time elasped for RTS* forward:{t_elasped}')
        print('='*90)
        t_forward += t_elasped
        observation, reward, done, info = env.step(lam.cpu().to(dtype=torch.get_default_dtype()) * torch.pi / 24, flag.cpu().to(dtype=torch.get_default_dtype()))

        ts = time.time()
        env.render(FO_link)
        #env.render()        
        t_render += time.time()-ts 



    print(f'Total time elasped for RTS* forward with {n_steps} steps: {t_forward}')
    print(f'Total time elasped for rendering with {n_steps} steps: {t_render}')