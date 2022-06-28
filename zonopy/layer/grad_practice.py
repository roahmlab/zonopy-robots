import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import cyipopt



import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.linalg import block_diag

import time 

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0


# NOTE: unbatch


class gradient_RTS_test:
    def __init__(ctx,link_zonos,joint_axes,n_links,n_obs,params):
        ctx.jrs_tensor = preload_batch_JRS_trig()
        ctx.dimension = 2
        ctx.n_timesteps = 100
        ctx.ka_0 = torch.zeros(n_links)
        ctx.PI_vel = torch.tensor(torch.pi-1e-6)
        ctx.g_ka = torch.pi/24
        
        ctx.link_zonos = link_zonos 
        ctx.joint_axes = joint_axes 
        ctx.n_links = n_links 
        ctx.n_obs = n_obs
        ctx.params = params 


    def forward(ctx,lambd,observation):
        # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
        
        ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
        ctx.lambd =lambd.clone().reshape(-1,ctx.n_links).to(dtype=torch.get_default_dtype())             
        observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
        ka = ctx.g_ka*ctx.lambd
        
        n_batches = observation.shape[0]
        ctx.qpos = observation[:,:ctx.n_links]
        ctx.qvel = observation[:,ctx.n_links:2*ctx.n_links]
        obstacle_pos = observation[:,-4*ctx.n_obs:-2*ctx.n_obs]
        obstacle_size = observation[:,-2*ctx.n_obs:]
        qgoal = ctx.qpos + ctx.qvel*T_PLAN + 0.5*ka*T_PLAN**2

        #g_ka = torch.maximum(PI/24,abs(qvel/3))

        _, R_trig = process_batch_JRS_trig_ic(ctx.jrs_tensor,ctx.qpos,ctx.qvel,ctx.joint_axes)
        FO_link,_,_ = forward_occupancy(R_trig,ctx.link_zonos,ctx.params)
        
        ctx.As = [[] for _ in range(n_links)]
        bs = [[] for _ in range(n_links)]
        lambda_to_slc = ctx.lambd.reshape(n_batches,1,ctx.dimension).repeat(1,ctx.n_timesteps,1)
        unsafe_flag = (abs(ctx.qvel+ctx.lambd*ctx.g_ka*T_PLAN)>ctx.PI_vel).any(-1)#NOTE: this might not work on gpu, velocity lim check
        for j in range(ctx.n_links):
            FO_link[j] = FO_link[j].project([0,1])
            c_k = FO_link[j].center_slice_all_dep(lambda_to_slc).unsqueeze(-1) # FOR, safety check
            for o in range(ctx.n_obs):
                obs_Z = torch.cat((obstacle_pos[:,2*o:2*(o+1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:,2*o:2*(o+1)])),-2).unsqueeze(-3).repeat(1,ctx.n_timesteps,1,1)
                A_temp, b_temp = batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension 
                ctx.As[j].append(A_temp)
                bs[j].append(b_temp)
                unsafe_flag += (torch.max((A_temp@c_k).squeeze(-1)-b_temp,-1)[0]<1e-6).any(-1)  #NOTE: this might not work on gpu FOR, safety check

        unsafe_flag = torch.ones(n_batches,dtype=bool) # NOTE: activate rts all ways

        M_obs = ctx.n_timesteps*ctx.n_links*ctx.n_obs
        M = M_obs+2*ctx.n_links
        ctx.flags = -torch.ones(n_batches,dtype=int) # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
        ctx.nlp_obj = [None for _ in range(n_batches)]
        ctx.nlp_info = [None for _ in range(n_batches)]
        for i in unsafe_flag.nonzero().reshape(-1):
            class nlp_setup():
                x_prev = np.zeros(ctx.n_links)*np.nan
                def objective(nlp,x):
                    qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*ctx.g_ka*x*T_PLAN**2
                    return torch.sum(wrap_to_pi(qplan-qgoal[i])**2)

                def gradient(nlp,x):
                    qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*ctx.g_ka*x*T_PLAN**2
                    return (ctx.g_ka*T_PLAN**2*wrap_to_pi(qplan-qgoal[i])).numpy()

                def constraints(nlp,x): 
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(ctx.n_timesteps,1)
                    if (nlp.x_prev!=x).any(): 
                        nlp.possible_obs_cons = []# NOTE
                        nlp.obs_cons_max_ind = torch.zeros(ctx.n_links,ctx.n_obs,ctx.n_timesteps,dtype=int)# NOTE
                        Cons = torch.zeros(M)   
                        Jac = torch.zeros(M,ctx.n_links)
                        # velocity constraints
                        q_peak = ctx.qvel[i]+ctx.g_ka*x*T_PLAN
                        grad_q_peak = ctx.g_ka*T_PLAN*torch.eye(ctx.n_links)
                        Cons[-2*ctx.n_links:] = torch.hstack((q_peak-torch.pi,-torch.pi-q_peak))
                        Jac[-2*ctx.n_links:] = torch.vstack((grad_q_peak,-grad_q_peak))
                        # velocity constraints 
                        for j in range(ctx.n_links):
                            c_k = FO_link[j][i].center_slice_all_dep(ka)
                            grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                            for o in range(ctx.n_obs):
                                pos_obs_cons = (ctx.As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i]
                                nlp.possible_obs_cons.extend(list(pos_obs_cons))
                                cons, nlp.obs_cons_max_ind[j,o]= torch.max(pos_obs_cons,-1) # shape: n_timsteps, SAFE if >=1e-6
                                jac = (ctx.As[j][o][i].gather(-2,nlp.obs_cons_max_ind[j,o].reshape(ctx.n_timesteps,1,1).repeat(1,1,ctx.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                Cons[(j+ctx.n_links*o)*ctx.n_timesteps:(j+ctx.n_links*o+1)*ctx.n_timesteps] = - cons
                                Jac[(j+ctx.n_links*o)*ctx.n_timesteps:(j+ctx.n_links*o+1)*ctx.n_timesteps] = - jac
                        nlp.cons = Cons.numpy()
                        nlp.jac = Jac.numpy()
                        nlp.x_prev = np.copy(x)
                    return nlp.cons

                def jacobian(nlp,x): 
                    ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(ctx.n_timesteps,1)
                    if (nlp.x_prev!=x).any(): 
                        nlp.possible_obs_cons = []# NOTE
                        nlp.obs_cons_max_ind = torch.zeros(ctx.n_links,ctx.n_obs,ctx.n_timesteps,dtype=int)# NOTE
                        Cons = torch.zeros(M)   
                        Jac = torch.zeros(M,ctx.n_links)
                        # velocity constraints
                        q_peak = ctx.qvel[i]+ctx.g_ka*x*T_PLAN
                        grad_q_peak = ctx.g_ka*T_PLAN*torch.eye(ctx.n_links)
                        Cons[-2*ctx.n_links:] = torch.hstack((q_peak-torch.pi,-torch.pi-q_peak))
                        Jac[-2*ctx.n_links:] = torch.vstack((grad_q_peak,-grad_q_peak))
                        # velocity constraints 
                        for j in range(ctx.n_links):
                            c_k = FO_link[j][i].center_slice_all_dep(ka)
                            grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                            for o in range(ctx.n_obs):
                                pos_obs_cons = (ctx.As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i]
                                nlp.possible_obs_cons.extend(list(pos_obs_cons))
                                cons, nlp.obs_cons_max_ind[j,o]= torch.max(pos_obs_cons,-1) # shape: n_timsteps, SAFE if >=1e-6
                                jac = (ctx.As[j][o][i].gather(-2,nlp.obs_cons_max_ind[j,o].reshape(ctx.n_timesteps,1,1).repeat(1,1,ctx.dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                Cons[(j+ctx.n_links*o)*ctx.n_timesteps:(j+ctx.n_links*o+1)*ctx.n_timesteps] = - cons
                                Jac[(j+ctx.n_links*o)*ctx.n_timesteps:(j+ctx.n_links*o+1)*ctx.n_timesteps] = - jac
                        nlp.cons = Cons.numpy()
                        nlp.jac = Jac.numpy()
                        nlp.x_prev = np.copy(x)
                    return nlp.jac
                
                def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                        d_norm, regularization_size, alpha_du, alpha_pr,
                        ls_trials):
                    pass
            

            ctx.nlp_obj[i] = nlp_setup()
            NLP = cyipopt.problem(
            n = ctx.n_links,
            m = M,
            problem_obj=ctx.nlp_obj[i],
            lb = [-1]*ctx.n_links,
            ub = [1]*ctx.n_links,
            cl = [-1e20]*M_obs+[-1e20]*2*ctx.n_links,
            cu = [-1e-6]*M_obs+[-1e-6]*2*ctx.n_links,
            )
            NLP.addOption('sb', 'yes')
            NLP.addOption('print_level', 0)
            
            k_opt, ctx.nlp_info[i] = NLP.solve(ctx.ka_0)

            # NOTE: for training, dont care about fail-safe
            if ctx.nlp_info[i]['status'] == 0:
                ctx.lambd[i] = torch.tensor(k_opt,dtype = torch.get_default_dtype())
                ctx.flags[i] = 0
            else:
                ctx.flags[i] = 1
        return ctx.lambd, FO_link, ctx.flags

    def backward_standard(ctx,direction):
        grad_input = torch.zeros(direction.shape).reshape(-1,ctx.n_links)
        direction = direction.reshape(-1,ctx.n_links)
        # COMPUTE GRADIENT
        tol = 1e-6
        M_obs = ctx.n_timesteps*ctx.n_links*ctx.n_obs
        for i, flag in enumerate(ctx.flags):
            if flag == -1: # direct pass
                grad_input[i] = torch.tensor(direction[i])
            elif flag == 0: # rts success path. solve QP   
                k_opt = ctx.lambd[i].numpy()
                scale = 2.0
                # compute jacobian of each smooth constraint which will be constraints for QP
                jac = ctx.nlp_obj[i].jacobian(k_opt)
                cons = ctx.nlp_obj[i].cons
                A_AT = []
                size_As = torch.zeros(ctx.n_links,ctx.n_obs,dtype=int)
                for j in range(ctx.n_links):
                    for o in range(ctx.n_obs):
                        A = ctx.As[j][o][i]
                        size_As[j,o] = A.shape[-2]
                        a_at = 2/scale*(A.gather(-2,ctx.nlp_obj[i].obs_cons_max_ind[j,o].reshape(ctx.n_timesteps,1,1).repeat(1,1,ctx.dimension))@A.transpose(-2,-1)).squeeze(-2)
                        A_AT.extend(list(a_at))
                size_As = size_As.unsqueeze(-1).repeat(1,1,ctx.n_timesteps).flatten()
                size_As = torch.hstack((torch.zeros(1,dtype=int),size_As)).cumsum(0)
                
                num_smooth_var = int(size_As[-1]) # full dimension of lambda
                num_a_var = ctx.n_links # number of decision var. in armtd
                num_b_var = num_a_var + num_smooth_var # number of decition var. in B-armtd
                Ac_b = torch.block_diag(*ctx.nlp_obj[i].possible_obs_cons).numpy().reshape(M_obs,-1)
                Ac_b = (Ac_b)*(Ac_b>1e-6+tol)*(Ac_b<-tol)
                qp_cons1 = np.hstack((1/scale*jac[:M_obs],- Ac_b)) # [A*c(k)-b].T*lambda # NOTE
                qp_cons2 = np.hstack((np.zeros((M_obs,num_a_var)),torch.block_diag(*A_AT).numpy())) # ||A.T*lambda||-1
                EYE = np.eye(num_b_var)
                #qp_cons3 = sp.csr_matrix(([1.]*size_As[-1],(range(qp_cons1.shape[-1]-n_links),range(n_links,qp_cons1.shape[-1]))))
                qp_cons3 = EYE[num_a_var:] # lambda
                qp_cons4 = -EYE[:num_a_var] # lb
                qp_cons5 = EYE[:num_a_var] # ub
                qp_cons6 =  np.hstack((jac[-2*ctx.n_links:],np.zeros((2*ctx.n_links,num_smooth_var))))
                qp_cons = np.vstack((qp_cons1,qp_cons2,qp_cons3,qp_cons4,qp_cons5,qp_cons6))

                # compute duals for smooth constraints                
                mult_smooth_cons1 = scale * ctx.nlp_info[i]['mult_g'][:M_obs]*(ctx.nlp_info[i]['mult_g'][:M_obs]>tol)
                mult_smooth_cons2 = np.zeros(M_obs)
                mult_smooth_cons3 = np.zeros(num_smooth_var)
                
                
                for idx in range(M_obs):
                    Ac_b = ctx.nlp_obj[i].possible_obs_cons[idx]
                    Ac_b = (Ac_b)*(Ac_b>1e-6+tol)*(Ac_b<-tol)
                    mult_smooth_cons3[size_As[idx]:size_As[idx+1]] = -mult_smooth_cons1[idx]*Ac_b # NOTE
                mult_smooth_cons4 = ctx.nlp_info[i]['mult_x_L']*(ctx.nlp_info[i]['mult_x_L']>tol)
                mult_smooth_cons5 = ctx.nlp_info[i]['mult_x_U']*(ctx.nlp_info[i]['mult_x_U']>tol)
                mult_smooth_cons6 = ctx.nlp_info[i]['mult_g'][-2*ctx.n_links:]*(ctx.nlp_info[i]['mult_g'][-2*ctx.n_links:]>tol)
            

                mult_smooth = np.hstack((mult_smooth_cons1,mult_smooth_cons2,mult_smooth_cons3,mult_smooth_cons4,mult_smooth_cons5,mult_smooth_cons6))
                
                # compute smooth constraints     
                smoother = np.zeros(num_smooth_var) # NOTE: we might wanna assign smoother value for inactive or weakly active as 1/2 instead of 1.
                obs_cons_max_inds = size_As[:-1]+ctx.nlp_obj[i].obs_cons_max_ind.flatten()
                smoother[obs_cons_max_inds] = 1/scale
                
                smooth_cons1 = 1/scale * cons[:M_obs]*(cons[:M_obs]<-1e-6-tol)
                smooth_cons2 = ((1/scale)**2-1)*np.ones(M_obs)
                '''
                # This will result in all (1/scale)**2-1 if all the nonzero element of smoother is 1/scale
                for j in range(n_links):
                    for o in range(n_obs):
                        A_smoother = As[j][o][i].gather(-2,ctx.nlp_obj[i].obs_cons_max_ind[j,o].reshape(n_timesteps,1,1).repeat(1,1,dimension)).squeeze(-2)
                        smooth_cons2[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = (torch.linalg.norm(A_smoother,dim=-1)/scale)**2-1        
                ''' 
                smooth_cons3 = -smoother
                smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt <-1e-6-tol)
                smooth_cons5 = (k_opt - 1) * (k_opt - 1 <-1e-6-tol)
                smooth_cons6 = cons[-2*ctx.n_links:]*(cons[-2*ctx.n_links:]<-1e-6-tol)
                smooth_cons = np.hstack((smooth_cons1,smooth_cons2,smooth_cons3,smooth_cons4,smooth_cons5,smooth_cons6))


                
                # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
                H = 0.5*sp.csr_matrix(([1.]*num_a_var,(range(num_a_var),range(num_a_var))),shape=(num_b_var,num_b_var))
                #f = sp.csr_matrix(([-1.]*num_a_var,(range(num_a_var),range(num_a_var))),shape=(num_b_var,num_b_var))
                f_d = sp.csr_matrix((-direction[i],([0.]*num_a_var,range(num_a_var))),shape=(1,num_b_var))

                strongly_active = (mult_smooth > tol) * (smooth_cons >= -1e-6-tol)
                weakly_active = (mult_smooth <= tol) * (smooth_cons >= -1e-6-tol)
                inactive = (smooth_cons < -1e-6-tol)



                # standard QP

                qp = gp.Model("back_prop")
                qp.Params.LogToConsole = 0
                z = qp.addMVar(shape=num_b_var, name="z",vtype=GRB.CONTINUOUS,ub=np.inf, lb=-np.inf)
                qp.setObjective(z@H@z+f_d@z, GRB.MINIMIZE)
                qp_eq_cons = sp.csr_matrix(qp_cons[strongly_active])
                rhs_eq = np.zeros(strongly_active.sum())
                qp_ineq_cons = sp.csr_matrix(qp_cons[weakly_active])
                rhs_ineq = -0*np.ones(weakly_active.sum())
                qp.addConstr( qp_eq_cons @ z == rhs_eq, name="eq")
                qp.addConstr(qp_ineq_cons @ z <= rhs_ineq, name="ineq")
                qp.optimize()
                grad_input[i] = torch.tensor(z.X[:num_a_var],dtype = torch.get_default_dtype())

        print(f'Solution of standard QP:{grad_input[0]}')
        return grad_input

    def backward_reduced(ctx,direction):
        grad_input = torch.zeros(direction.shape).reshape(-1,ctx.n_links)
        direction = direction.reshape(-1,ctx.n_links)
        # COMPUTE GRADIENT
        tol = 1e-6
        for i, flag in enumerate(ctx.flags):
            if flag == -1: # direct pass
                grad_input[i] = torch.tensor(direction[i])
            elif flag == 0: # rts success path. solve QP   
                k_opt = ctx.lambd[i].numpy()
                # compute jacobian of each smooth constraint which will be constraints for QP
                jac = ctx.nlp_obj[i].jacobian(k_opt)
                cons = ctx.nlp_obj[i].cons

                num_a_var = ctx.n_links # number of decision var. in armtd
                qp_cons1 = jac # [A*c(k)-b].T*lambda  and vel. lim # NOTE
                EYE = np.eye(num_a_var)
                qp_cons4 = -EYE# lb
                qp_cons5 = EYE # ub
                qp_cons = np.vstack((qp_cons1,qp_cons4,qp_cons5))

                # compute duals for smooth constraints                
                mult_smooth_cons1 = ctx.nlp_info[i]['mult_g']*(ctx.nlp_info[i]['mult_g']>tol)
                mult_smooth_cons4 = ctx.nlp_info[i]['mult_x_L']*(ctx.nlp_info[i]['mult_x_L']>tol)
                mult_smooth_cons5 = ctx.nlp_info[i]['mult_x_U']*(ctx.nlp_info[i]['mult_x_U']>tol)
                mult_smooth = np.hstack((mult_smooth_cons1,mult_smooth_cons4,mult_smooth_cons5))
                
                # compute smooth constraints
                smooth_cons1 = cons*(cons<-1e-6-tol)
                smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt <-1e-6-tol)
                smooth_cons5 = (k_opt - 1) * (k_opt - 1 <-1e-6-tol)
                smooth_cons = np.hstack((smooth_cons1,smooth_cons4,smooth_cons5))


                strongly_active = (mult_smooth > tol) * (smooth_cons >= -1e-6-tol)
                weakly_active = (mult_smooth <= tol) * (smooth_cons >= -1e-6-tol)
                inactive = (smooth_cons < -1e-6-tol)

                # reduced QP
                # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
                H = 0.5*sp.csr_matrix(([1.]*num_a_var,(range(num_a_var),range(num_a_var))),shape=(num_a_var,num_a_var))
                f_d = sp.csr_matrix((-direction[i],([0.]*num_a_var,range(num_a_var))),shape=(1,num_a_var))

                qp = gp.Model("back_prop_reduced")
                qp.Params.LogToConsole = 0
                z = qp.addMVar(shape=num_a_var, name="z",vtype=GRB.CONTINUOUS,ub=np.inf, lb=-np.inf)
                qp.setObjective(z@H@z+f_d@z, GRB.MINIMIZE)
                qp_eq_cons = sp.csr_matrix(qp_cons[strongly_active])
                rhs_eq = np.zeros(strongly_active.sum())
                qp_ineq_cons = sp.csr_matrix(qp_cons[weakly_active])
                rhs_ineq = -0*np.ones(weakly_active.sum())
                qp.addConstr( qp_eq_cons @ z == rhs_eq, name="eq")
                qp.addConstr(qp_ineq_cons @ z <= rhs_ineq, name="ineq")
                qp.optimize()
                grad_input[i] = torch.tensor(z.X,dtype = torch.get_default_dtype())
        print(f'Solution of reduced QP:{grad_input[0]}')
        return grad_input


    def backward_reduced_batch(ctx,direction):
        grad_input = torch.zeros(direction.shape).reshape(-1,ctx.n_links)
        direction = direction.reshape(-1,ctx.n_links)
        # COMPUTE GRADIENT
        tol = 1e-6
        # direct pass
        direct_pass = ctx.flags == -1
        grad_input[direct_pass] = torch.tensor(direction)[direct_pass]


        rts_success_pass = (ctx.flags == 0).nonzero().reshape(-1)
        n_batch = rts_success_pass.numel()
        if n_batch > 0:
            QP_EQ_CONS = []
            QP_INEQ_CONS = []
            for i in rts_success_pass:
                k_opt = ctx.lambd[i].numpy()
                # compute jacobian of each smooth constraint which will be constraints for QP
                jac = ctx.nlp_obj[i].jacobian(k_opt)
                cons = ctx.nlp_obj[i].cons

                num_a_var = ctx.n_links # number of decision var. in armtd
                qp_cons1 = jac # [A*c(k)-b].T*lambda  and vel. lim # NOTE
                EYE = np.eye(num_a_var)
                qp_cons4 = -EYE# lb
                qp_cons5 = EYE # ub
                qp_cons = np.vstack((qp_cons1,qp_cons4,qp_cons5))

                # compute duals for smooth constraints                
                mult_smooth_cons1 = ctx.nlp_info[i]['mult_g']*(ctx.nlp_info[i]['mult_g']>tol)
                mult_smooth_cons4 = ctx.nlp_info[i]['mult_x_L']*(ctx.nlp_info[i]['mult_x_L']>tol)
                mult_smooth_cons5 = ctx.nlp_info[i]['mult_x_U']*(ctx.nlp_info[i]['mult_x_U']>tol)
                mult_smooth = np.hstack((mult_smooth_cons1,mult_smooth_cons4,mult_smooth_cons5))
                
                # compute smooth constraints
                smooth_cons1 = cons*(cons<-1e-6-tol)
                smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt <-1e-6-tol)
                smooth_cons5 = (k_opt - 1) * (k_opt - 1 <-1e-6-tol)
                smooth_cons = np.hstack((smooth_cons1,smooth_cons4,smooth_cons5))


                strongly_active = (mult_smooth > tol) * (smooth_cons >= -1e-6-tol)
                weakly_active = (mult_smooth <= tol) * (smooth_cons >= -1e-6-tol)

                QP_EQ_CONS.append(qp_cons[strongly_active])
                QP_INEQ_CONS.append(qp_cons[weakly_active])
            
            
            
            # reduced batch QP
            # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
            qp_size = n_batch*ctx.n_links
            H = 0.5*sp.csr_matrix(([1.]*qp_size,(range(qp_size),range(qp_size))))            
            f_d = sp.csr_matrix((-direction[rts_success_pass].flatten(),([0]*qp_size,range(qp_size))))

            qp = gp.Model("back_prop_reduced")
            qp.Params.LogToConsole = 0
            z = qp.addMVar(shape=qp_size, name="z",vtype=GRB.CONTINUOUS,ub=np.inf, lb=-np.inf)
            qp.setObjective(z@H@z+f_d@z, GRB.MINIMIZE)
            qp_eq_cons = sp.csr_matrix(block_diag(*QP_EQ_CONS))
            rhs_eq = np.zeros(qp_eq_cons.shape[0])
            qp_ineq_cons = sp.csr_matrix(block_diag(*QP_INEQ_CONS))
            rhs_ineq = -0*np.ones(qp_ineq_cons.shape[0])
            qp.addConstr( qp_eq_cons @ z == rhs_eq, name="eq")
            qp.addConstr(qp_ineq_cons @ z <= rhs_ineq, name="ineq")
            qp.optimize()
            grad_input[rts_success_pass] = torch.tensor(z.X.reshape(n_batch,ctx.n_links))
        
        print(f'Solution of reduced batch QP:{grad_input[0]}')    
        return grad_input


if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    n_batch = 100
    eps = 1e-6
    n_links = 2
    E = eps*torch.eye(n_links)
    d = np.random.uniform(-1,1,n_links)
    d = np.array([.1,.2])
    print(f'directional input: {d}')
    #d = np.array([-1,0])

    torch.set_default_dtype(torch.float64)

    render = True
    env = Arm_2D(n_links=n_links,n_obs=1)
    observation = env.reset()

    observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.tensor([0.*torch.pi,0.*torch.pi]), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([1,0.6])])
    t_armtd = 0
    params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
    joint_axes = [j for j in env.joint_axes]
    RTS = gradient_RTS_test(env.link_zonos,joint_axes,env.n_links,env.n_obs,params)

    observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys()]).reshape(1,-1)
    #k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
    lam_hat = torch.tensor([[0.9,.8]])
    lam, FO_link, flag = RTS.forward(lam_hat.repeat(n_batch,1),observ_temp.repeat(n_batch,1)) 

    d = d.reshape(1,-1).repeat(n_batch,1)
    t1 = time.time()
    RTS.backward_standard(d)
    t2 = time.time()
    RTS.backward_reduced(d)
    t3 = time.time()
    RTS.backward_reduced_batch(d)
    t4 = time.time()
    print(f'Elasped time for standard QP: {t2-t1} sec.')
    print(f'Elasped time for reduced QP: {t3-t2} sec.')
    print(f'Elasped time for reduced batch QP: {t4-t3} sec.')

    import pdb;pdb.set_trace()
    assert flag>=0, 'fail safe'

    
    if render:
        observation, reward, done, info = env.step(lam[0]*torch.pi/24,flag[0])
        FO_link = [fo[0] for fo in FO_link]
        env.render(FO_link)

    print(f'action: {lam}')

    diff1 = torch.zeros(n_links,n_links)
    diff2 = torch.zeros(n_links,n_links)

    for i in range(n_links):
        lam_hat1 = lam_hat + E[:,i]
        lam1, _, flag1 = RTS(lam_hat1,observ_temp.reshape(1,-1),d) 

        lam_hat2 = lam_hat - E[:,i]
        lam2, _, flag2 = RTS(lam_hat2,observ_temp.reshape(1,-1),d) 


        diff1[:,i] = (lam1-lam)/eps
        diff2[:,i] = (lam1-lam2)/(2*eps)
    print(diff1)
    print(diff1@d)

