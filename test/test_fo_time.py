import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy

import time
#if torch.cuda.is_available():
#    zp.conSet.DEFAULT_OPTS.set(device='cuda:0')
zp.setup_cuda()
N_test = 100
jrs_t = torch.zeros(N_test)
fo_t = torch.zeros(N_test) 
for idx in range(N_test):
    qpos =  torch.tensor([0.0,0.0])
    qvel =  torch.tensor([torch.pi,torch.pi/2])
    params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*2, 
            'R': [torch.eye(3)]*2,
            'P': [torch.tensor([0.0,0.0,0.0]), torch.tensor([1.0,0.0,0.0])],
            'n_joints':2}
    link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]])).to_polyZonotope()]*2

    t_start = time.time()
    _, R_trig = zp.load_JRS_trig(qpos,qvel)
    t =  time.time()

    jrs_t[idx] = t-t_start
    t_start = t
    #_, R =zp.gen_JRS(qpos,qvel,params['joint_axes'],taylor_degree=1,make_gens_independent =True)
    n_time_steps = len(R_trig)
    t =  time.time()
    #print(t-t_start)
    t_start = t

    FO_link_trig, FO_link = [], []

    for t in range(n_time_steps):
        FO_link_temp,_,_ = forward_occupancy(R_trig[t],link_zonos,params)
        FO_link_trig.append(FO_link_temp)
        #FO_link_temp,_,_ = forward_occupancy(R[t],link_zonos,params)
        #FO_link.append(FO_link_temp)
    t =  time.time()

    fo_t[idx] = t-t_start
    t_start = t

print(torch.mean(jrs_t))
print(torch.mean(fo_t))