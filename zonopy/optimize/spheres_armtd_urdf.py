# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.SO import sphere_occupancy, make_spheres
import cyipopt

import time

T_PLAN, T_FULL = 0.5, 1.0

from typing import List
from zonopy.optimize.sphere_nlp_problem import OfflineArmtdSphereConstraints
from zonopy.optimize.armtd_nlp_problem import ArmtdNlpProblem
from zonopy.robots2.robot import ZonoArmRobot

from zonopy.optimize.distance_net.compute_vertices_from_generators import compute_edges_from_generators

class ARMTD_3D_planner():
    def __init__(self,
                 robot: ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 sphere_device: torch.device = torch.device('cpu'),
                 spheres_per_link: int = 5,
                 include_end_effector: bool = False,
                 ):
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS_tensor = zp.preload_batch_JRS_trig(dtype=self.dtype,device=self.device)
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)
        self.include_end_effector = include_end_effector

        self._setup_robot(robot)
        self.SFO_constraint = OfflineArmtdSphereConstraints(dtype=dtype, device=sphere_device)
        self.sphere_device = sphere_device
        self.spheres_per_link = spheres_per_link

        # Prepare the nlp
        self.g_ka = np.ones((self.dof),dtype=self.np_dtype) * np.pi/24                  # Hardcoded because it's preloaded...
        self.nlp_problem_obj = ArmtdNlpProblem(self.dof,
                                         self.g_ka,
                                         self.pos_lim, 
                                         self.vel_lim,
                                         self.continuous_joints,
                                         self.pos_lim_mask,
                                         self.dtype,
                                         T_PLAN,
                                         T_FULL)

        # self.wrap_env(env)
        # self.n_timesteps = 100
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def _setup_robot(self, robot: ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass

    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
        
    def _prepare_SO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        ### Process the obstacles
        dist_net_time = time.perf_counter()
        n_obs = len(obs_zono)
        # 5. Compute hyperplanes from buffered obstacles generators
        # TODO: this step might be able to be optimized
        hyperplanes_A, hyperplanes_b = obs_zono.to(device=self.sphere_device).polytope(self.combs)
        hyperplanes_b = hyperplanes_b.unsqueeze(-1)
        
        # 6. Compute vertices from buffered obstacles generators
        v1, v2 = compute_edges_from_generators(obs_zono.Z[...,0:1,:], obs_zono.Z[...,1:,:], hyperplanes_A, hyperplanes_b)

        # combine to one input for the NN
        obs_tuple = (hyperplanes_A, hyperplanes_b, v1, v2)

        ### get the forward occupancy
        SFO_gen_time = time.perf_counter()
        joint_occ, link_joint_pairs, _ = sphere_occupancy(JRS_R, self.robot, self.zono_order)
        
        # Flatten out all link joint pairs so we can just process that
        joint_pairs = []
        for pairs in link_joint_pairs.values():
            joint_pairs.extend(pairs)
        n_timesteps = JRS_R[0].batch_shape[0]

        ## Discard pairs that are too far to be a problem
        hyp_A = obs_tuple[0].expand(self.spheres_per_link*n_timesteps, -1, -1, -1).reshape(-1, *obs_tuple[0].shape[-2:])
        hyp_b = obs_tuple[1].expand(self.spheres_per_link*n_timesteps, -1, -1, -1).reshape(-1, *obs_tuple[1].shape[-2:])
        v1 = obs_tuple[2].expand(self.spheres_per_link*n_timesteps, -1, -1, -1).reshape(-1, *obs_tuple[2].shape[-2:])
        v2 = obs_tuple[3].expand(self.spheres_per_link*n_timesteps, -1, -1, -1).reshape(-1, *obs_tuple[3].shape[-2:])

        joints = {}
        for name, (pz, r) in joint_occ.items():
            joint_int = pz.to_interval()
            centers = joint_int.center()
            joints[name] = (centers, r + joint_int.rad().max(-1)[0]*np.sqrt(3))

        def keep_pair(pair):
            p1, r1 = joints[pair[0]]
            p2, r2 = joints[pair[1]]
            spheres = make_spheres(p1, p2, r1, r2, n_spheres=self.spheres_per_link)
            points = spheres[0].expand(n_obs, -1, -1, -1).transpose(0,1).reshape(-1, self.SFO_constraint.dimension)
            radii = spheres[1].expand(n_obs, -1, -1).transpose(0,1).reshape(-1)
            dist_out, _ = self.SFO_constraint.distance_net(points, hyp_A, hyp_b, v1, v2)
            return torch.any(dist_out < radii)
        joint_pairs = [pair for pair in joint_pairs if keep_pair(pair)]
        ## End discard pairs

        # output range
        out_g_ka = self.g_ka

        # Build the constraint
        self.SFO_constraint.set_params(joint_occ, joint_pairs, out_g_ka, obs_tuple, n_obs, self.spheres_per_link, n_timesteps, self.dof)

        final_time = time.perf_counter()
        out_times = {
            'SFO_gen': final_time - SFO_gen_time,
            'distance_prep_net': SFO_gen_time - dist_net_time,
        }
        return self.SFO_constraint, out_times

    def trajopt(self, qpos, qvel, qgoal, ka_0, SFO_constraint):
        # Moved to another file
        self.nlp_problem_obj.reset(qpos, qvel, qgoal, SFO_constraint, cons_val = -1e-6)
        n_constraints = self.nlp_problem_obj.M

        nlp = cyipopt.Problem(
        n = self.dof,
        m = n_constraints,
        problem_obj=self.nlp_problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*n_constraints,
        cu = [-1e-6]*n_constraints,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)                
        return SFO_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
    def plan(self,qpos, qvel, qgoal, obs, ka_0 = None):
        # prepare the JRS
        JRS_process_time = time.perf_counter()
        _, JRS_R = zp.process_batch_JRS_trig(self.JRS_tensor,
                                             torch.as_tensor(qpos, dtype=self.dtype, device=self.device),
                                             torch.as_tensor(qvel, dtype=self.dtype, device=self.device),
                                             self.joint_axis)
        JRS_process_time = time.perf_counter() - JRS_process_time

        # Create obs zonotopes
        obs_Z = torch.cat((
            torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
            torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2.
            ), dim=-2)
        obs_zono = zp.batchZonotope(obs_Z)

        # Compute FO
        SFO_constraint, SFO_times = self._prepare_SO_constraints(JRS_R, obs_zono)

        # preproc_time, FO_gen_time, constraint_time = self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)

        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, qgoal, ka_0, SFO_constraint)
        trajopt_time = time.perf_counter() - trajopt_time
        return k_opt, flag, trajopt_time, SFO_times['SFO_gen'], SFO_times['distance_prep_net'], JRS_process_time, constraint_times


if __name__ == '__main__':
    from zonopy.environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### LOAD ROBOT #####
    import os
    import zonopy as zp
    basedirname = os.path.dirname(zp.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    import zonopy.robots2.robot as robots2
    robots2.DEBUG_VIZ = False
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), device=device, dtype=dtype, create_joint_occupancy=True)
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    ##### SET ENVIRONMENT #####
    # Make a list of all links except the base and effector for collision checks (match SO)
    links = rob.urdf.links.copy()
    links.remove(rob.urdf.base_link)
    for el in rob.urdf.end_links:
        links.remove(el)
    links = [link.name for link in links]
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=False,
        collision_links=links,
        use_bb_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        n_obs=5,
        )
    # obs = env.reset()
    obs = env.reset(
        qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
        # qpos=np.array([0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428])-0.2,
        qvel=np.array([0,0,0,0,0,0,0.]),
        qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
        obs_pos=[
            np.array([0.65,-0.46,0.33]),
            np.array([0.5,-0.43,0.3]),
            np.array([0.47,-0.45,0.15]),
            np.array([-0.3,0.2,0.23]),
            np.array([0.3,0.2,0.31])
            ])
    # obs = env.reset(
    #     qpos=np.array([ 3.1098, -0.9964, -0.2729, -2.3615,  0.2724, -1.6465, -0.5739]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([-1.9472,  1.4003, -1.3683, -1.1298,  0.7062, -1.0147, -1.1896]),
    #     obs_pos=[
    #         np.array([ 0.3925, -0.7788,  0.2958]),
    #         np.array([0.3550, 0.3895, 0.3000]),
    #         np.array([-0.0475, -0.1682, -0.7190]),
    #         np.array([0.3896, 0.5005, 0.7413]),
    #         np.array([0.4406, 0.1859, 0.1840]),
    #         np.array([ 0.1462, -0.6461,  0.7416]),
    #         np.array([-0.4969, -0.5828,  0.1111]),
    #         np.array([-0.0275,  0.1137,  0.6985]),
    #         np.array([ 0.4139, -0.1155,  0.7733]),
    #         np.array([ 0.5243, -0.7838,  0.4781])
    #         ])

    ##### 2. RUN ARMTD #####    
    planner = ARMTD_3D_planner(rob, device=device, sphere_device=device, dtype=dtype, spheres_per_link=4)
    t_armtd = []
    T_NLP = []
    T_SFO = []
    T_NET_PREP = []
    T_PREPROC = []
    T_CONSTR_E = []
    N_EVALS = []
    n_steps = 100
    for _ in range(n_steps):
        ts = time.time()
        qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
        ka, flag, tnlp, tsfo, tdnp, tpreproc, tconstraint_evals = planner.plan(qpos, qvel, qgoal, obstacles)
        t_elasped = time.time()-ts
        #print(f'Time elasped for ARMTD-3d:{t_elasped}')
        T_NLP.append(tnlp)
        T_SFO.append(tsfo)
        T_NET_PREP.append(tdnp)
        T_PREPROC.append(tpreproc)
        T_CONSTR_E.extend(tconstraint_evals)
        N_EVALS.append(len(tconstraint_evals))
        t_armtd.append(t_elasped)
        if flag != 0:
            print("executing failsafe!")
            ka = (0 - qvel)/(T_FULL - T_PLAN)
        obs, rew, done, info = env.step(ka)
        # env.step(ka,flag)
        assert(not info['collision_info']['in_collision'])
        # env.render()
    from scipy import stats
    print(f'Total time elasped for ARMTD-3D with {n_steps} steps: {stats.describe(t_armtd)}')
    print("Per step")
    print(f'NLP: {stats.describe(T_NLP)}')
    print(f'constraint evals: {stats.describe(T_CONSTR_E)}')
    print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    print(f'Distance net prep: {stats.describe(T_NET_PREP)}')
    print(f'SFO generation: {stats.describe(T_SFO)}')
    print(f'JRS preprocessing: {stats.describe(T_PREPROC)}')