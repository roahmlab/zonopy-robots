Search.setIndex({"docnames": ["dynamics", "generated/zonopyrobots.ZonoArmRobot", "generated/zonopyrobots.dynamics.pzrnea", "generated/zonopyrobots.kinematics.forward_kinematics", "generated/zonopyrobots.make_cfg_dict", "generated/zonopyrobots.robots.armrobot.ZonoArmRobotJoint", "generated/zonopyrobots.robots.armrobot.ZonoArmRobotLink", "generated/zonopyrobots.trajectory.BernsteinArmTrajectory", "generated/zonopyrobots.trajectory.PiecewiseArmTrajectory", "index", "kinematics", "robots", "trajectories", "utils"], "filenames": ["dynamics.rst", "generated/zonopyrobots.ZonoArmRobot.rst", "generated/zonopyrobots.dynamics.pzrnea.rst", "generated/zonopyrobots.kinematics.forward_kinematics.rst", "generated/zonopyrobots.make_cfg_dict.rst", "generated/zonopyrobots.robots.armrobot.ZonoArmRobotJoint.rst", "generated/zonopyrobots.robots.armrobot.ZonoArmRobotLink.rst", "generated/zonopyrobots.trajectory.BernsteinArmTrajectory.rst", "generated/zonopyrobots.trajectory.PiecewiseArmTrajectory.rst", "index.rst", "kinematics.rst", "robots.rst", "trajectories.rst", "utils.rst"], "titles": ["Dynamics Module", "zonopyrobots.ZonoArmRobot", "zonopyrobots.dynamics.pzrnea", "zonopyrobots.kinematics.forward_kinematics", "zonopyrobots.make_cfg_dict", "zonopyrobots.robots.armrobot.ZonoArmRobotJoint", "zonopyrobots.robots.armrobot.ZonoArmRobotLink", "zonopyrobots.trajectory.BernsteinArmTrajectory", "zonopyrobots.trajectory.PiecewiseArmTrajectory", "Zonopy Robots - Extensions to Zonopy for Robotics", "Kinematics Module", "Robots Module", "Common Trajectory Types", "Utility Functions"], "terms": {"class": [1, 5, 6, 7, 8, 11], "sourc": [1, 2, 3, 4, 5, 6, 7, 8], "base": [1, 2, 5, 6, 7, 8], "object": [1, 5, 6], "arm": [1, 2, 5, 6, 9], "robot": [1, 2, 3, 4], "definit": [1, 5, 6], "us": [1, 2, 5, 6], "zonopi": [1, 5, 6, 7, 8], "thi": [1, 2, 5, 6, 9, 11], "i": [1, 2, 4, 5, 6, 9], "wrapper": 1, "around": 1, "urdf": [1, 2, 4], "from": 1, "urchin": 1, "It": [1, 5, 6], "also": [1, 5, 6], "comput": [1, 3, 5], "some": 1, "addit": [1, 5, 6], "inform": [1, 5, 6], "about": [1, 5, 6], "zonotop": [1, 2, 3, 5, 6], "includ": [1, 5, 6, 9], "joint": [1, 2, 5], "axi": [1, 5], "origin": [1, 5], "limit": 1, "link": [1, 2, 3, 6, 9], "data": [1, 5, 6], "each": [1, 2], "bound": [1, 5, 6], "box": [1, 5, 6], "mass": [1, 2, 6], "center": [1, 2, 6], "inertia": [1, 2, 6], "tensor": [1, 2, 5, 6, 7, 8], "occup": [1, 5], "help": 1, "later": 1, "forward": [1, 3], "kinemat": [1, 9], "The": [1, 2, 3, 4, 5, 6], "dof": 1, "number": 1, "actuat": [1, 2, 5], "np": [1, 2, 5, 6], "A": [1, 5, 6], "numpi": [1, 5, 6], "version": [1, 5, 6], "torch": [1, 2, 5, 6, 7, 8], "devic": [1, 5, 6], "primari": [1, 5, 6], "none": [1, 2, 3, 4, 5, 6], "dtype": 1, "pytorch": 1, "joint_axi": 1, "3": 1, "arrai": 1, "These": 1, "ar": [1, 9], "topolog": 1, "order": [1, 2, 3], "end": 1, "effector": 1, "joint_origin": 1, "joint_origins_al": 1, "fix": [1, 5], "n_joint": 1, "actuated_mask": 1, "mask": 1, "which": [1, 11], "an": [1, 2], "boolean": 1, "pos_lim": 1, "posit": 1, "actuatedjoint": 1, "2": [1, 2], "where": 1, "first": 1, "row": 1, "lower": [1, 2], "second": 1, "upper": 1, "continu": 1, "have": 1, "inf": 1, "vel_lim": 1, "veloc": [1, 2], "eff_lim": 1, "effort": 1, "continuous_joint": 1, "indic": 1, "n_continuous_joint": 1, "pos_lim_mask": 1, "finit": 1, "link_parent_joint": 1, "map": 1, "parent": 1, "ha": 1, "name": 1, "link_child_joint": 1, "child": 1, "set": 1, "empti": 1, "joint_data": 1, "zonoarmrobotjoint": 1, "link_data": 1, "zonoarmrobotlink": 1, "is_single_chain": 1, "true": 1, "singl": 1, "chain": 1, "has_closed_loop": 1, "close": 1, "loop": 1, "__init__": [1, 5, 6, 7, 8], "method": [1, 5, 6, 7, 8], "attribut": [1, 5, 6], "static": 1, "load": 1, "str": [1, 2, 3, 4], "ityp": 1, "create_joint_occup": 1, "fals": [1, 4], "file": 1, "paramet": [1, 2, 3, 4, 11], "put": 1, "default": [1, 2, 3, 4], "type": [1, 2, 3, 4, 9], "index": [1, 5, 9], "If": [1, 2], "creat": [1, 4], "those": 1, "valu": [1, 2], "popul": 1, "return": [1, 2, 3, 4, 5, 6], "convert": [1, 5, 6], "doe": [1, 5, 6], "copi": [1, 5, 6], "rotatotop": [2, 3], "dict": [2, 3, 4], "mpztype": [2, 3], "bmpztype": [2, 3], "list": [2, 3, 4], "qd": 2, "pztype": [2, 3], "bpztype": [2, 3], "qd_aux": 2, "qdd": 2, "zonoarmrobot": [2, 3], "zono_ord": [2, 3], "int": [2, 3], "40": 2, "graviti": 2, "ndarrai": [2, 7, 8], "link_mass_overrid": 2, "ani": [2, 4], "link_center_mass_overrid": 2, "link_inertia_overrid": 2, "ordereddicttyp": [2, 3], "rnea": 2, "polynomi": 2, "implement": 2, "descript": 2, "iter": 2, "newton": 2, "euler": 2, "algorithm": 2, "describ": 2, "john": 2, "j": 2, "craig": 2, "": 2, "introduct": 2, "mechan": 2, "control": 2, "3rd": 2, "edit": 2, "chapter": 2, "6": 2, "5": [2, 7, 8], "isbn": 2, "978": 2, "0": [2, 7, 8], "201": 2, "54361": 2, "union": [2, 3, 4], "matpolyzonotop": 2, "batchmatpolyzonotop": 2, "dictionari": [2, 4], "provid": [2, 4, 5, 6], "assum": 2, "same": 2, "ident": 2, "matrix": 2, "polyzonotop": 2, "batchpolyzonotop": [2, 7, 8], "auxillari": 2, "acceler": 2, "option": [2, 3, 4], "faster": 2, "less": 2, "accur": 2, "vector": 2, "9": 2, "81": 2, "overrid": 2, "entri": 2, "kei": 2, "forc": 2, "moment": [2, 9], "torqu": 2, "applic": 2, "20": 3, "tupl": 3, "config": 4, "allow_incomplet": 4, "bool": 4, "helper": 4, "function": [4, 9], "whether": 4, "allow": 4, "incomplet": 4, "actuated_idx": 5, "joint_occup": 5, "radiu": 5, "specifi": [5, 11], "idx": 5, "l_inf": 5, "norm": 5, "aabb": 5, "align": 5, "outer_pz": 5, "outer": 5, "bounding_pz": [5, 6], "properti": 5, "center_mass": 6, "frame": 6, "q0": [7, 8], "qd0": [7, 8], "qdd0": [7, 8], "kparam": [7, 8], "krang": [7, 8], "tbrake": [7, 8], "float": [7, 8], "tfinal": [7, 8], "1": [7, 8], "basearmtrajectori": [7, 8], "getrefer": [7, 8], "time": [7, 8], "extend": 9, "librari": 9, "dynam": 9, "trajectori": 9, "plan": 9, "system": 9, "At": 9, "current": 9, "focu": 9, "serial": 9, "manipul": 9, "support": 9, "other": 9, "project": 9, "still": 9, "earli": 9, "develop": 9, "so": 9, "much": 9, "api": 9, "subject": 9, "chang": 9, "modul": 9, "common": 9, "parameter": 9, "zonopyrobot": 9, "forward_kinemat": 9, "pzrnea": 9, "util": 9, "make_cfg_dict": 9, "search": 9, "page": 9, "contain": 11}, "objects": {"": [[11, 0, 0, "-", "zonopyrobots"]], "zonopyrobots": [[1, 1, 1, "", "ZonoArmRobot"], [4, 4, 1, "", "make_cfg_dict"]], "zonopyrobots.ZonoArmRobot": [[1, 2, 1, "", "__init__"], [1, 3, 1, "id0", "actuated_mask"], [1, 3, 1, "id1", "continuous_joints"], [1, 3, 1, "id2", "device"], [1, 3, 1, "id3", "dof"], [1, 3, 1, "id4", "dtype"], [1, 3, 1, "id5", "eff_lim"], [1, 3, 1, "id6", "has_closed_loop"], [1, 3, 1, "id7", "is_single_chain"], [1, 3, 1, "id8", "joint_axis"], [1, 3, 1, "id9", "joint_data"], [1, 3, 1, "id10", "joint_origins"], [1, 3, 1, "id11", "joint_origins_all"], [1, 3, 1, "id12", "link_child_joints"], [1, 3, 1, "id13", "link_data"], [1, 3, 1, "id14", "link_parent_joint"], [1, 2, 1, "", "load"], [1, 3, 1, "id15", "np"], [1, 2, 1, "", "numpy"], [1, 3, 1, "id16", "pos_lim"], [1, 3, 1, "id17", "pos_lim_mask"], [1, 3, 1, "id18", "tensor"], [1, 2, 1, "", "to"], [1, 3, 1, "id19", "urdf"], [1, 3, 1, "id20", "vel_lim"]], "zonopyrobots.dynamics": [[2, 4, 1, "", "pzrnea"]], "zonopyrobots.kinematics": [[3, 4, 1, "", "forward_kinematics"]], "zonopyrobots.robots.armrobot": [[5, 1, 1, "", "ZonoArmRobotJoint"], [6, 1, 1, "", "ZonoArmRobotLink"]], "zonopyrobots.robots.armrobot.ZonoArmRobotJoint": [[5, 2, 1, "", "__init__"], [5, 3, 1, "id0", "aabb"], [5, 3, 1, "id1", "axis"], [5, 3, 1, "id2", "bounding_pz"], [5, 3, 1, "id3", "device"], [5, 5, 1, "id4", "idx"], [5, 3, 1, "id5", "np"], [5, 2, 1, "", "numpy"], [5, 3, 1, "id6", "origin"], [5, 3, 1, "id7", "outer_pz"], [5, 3, 1, "id8", "radius"], [5, 3, 1, "id9", "tensor"], [5, 2, 1, "", "to"]], "zonopyrobots.robots.armrobot.ZonoArmRobotLink": [[6, 2, 1, "", "__init__"], [6, 3, 1, "id0", "bounding_pz"], [6, 3, 1, "id1", "center_mass"], [6, 3, 1, "id2", "device"], [6, 3, 1, "id3", "inertia"], [6, 3, 1, "id4", "mass"], [6, 3, 1, "id5", "np"], [6, 2, 1, "", "numpy"], [6, 3, 1, "id6", "tensor"], [6, 2, 1, "", "to"]], "zonopyrobots.trajectory": [[7, 1, 1, "", "BernsteinArmTrajectory"], [8, 1, 1, "", "PiecewiseArmTrajectory"]], "zonopyrobots.trajectory.BernsteinArmTrajectory": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "getReference"]], "zonopyrobots.trajectory.PiecewiseArmTrajectory": [[8, 2, 1, "", "__init__"], [8, 2, 1, "", "getReference"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:attribute", "4": "py:function", "5": "py:property"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "function", "Python function"], "5": ["py", "property", "Python property"]}, "titleterms": {"dynam": [0, 2], "modul": [0, 10, 11], "zonopyrobot": [1, 2, 3, 4, 5, 6, 7, 8], "zonoarmrobot": 1, "pzrnea": 2, "kinemat": [3, 10, 12], "forward_kinemat": 3, "make_cfg_dict": 4, "robot": [5, 6, 9, 11], "armrobot": [5, 6], "zonoarmrobotjoint": 5, "zonoarmrobotlink": 6, "trajectori": [7, 8, 12], "bernsteinarmtrajectori": 7, "piecewisearmtrajectori": 8, "zonopi": 9, "extens": 9, "content": 9, "indic": 9, "tabl": 9, "arm": 11, "common": 12, "type": 12, "parameter": 12, "util": 13, "function": 13}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 58}, "alltitles": {"Dynamics Module": [[0, "dynamics-module"]], "zonopyrobots.ZonoArmRobot": [[1, "zonopyrobots-zonoarmrobot"]], "zonopyrobots.dynamics.pzrnea": [[2, "zonopyrobots-dynamics-pzrnea"]], "zonopyrobots.kinematics.forward_kinematics": [[3, "zonopyrobots-kinematics-forward-kinematics"]], "zonopyrobots.make_cfg_dict": [[4, "zonopyrobots-make-cfg-dict"]], "zonopyrobots.robots.armrobot.ZonoArmRobotJoint": [[5, "zonopyrobots-robots-armrobot-zonoarmrobotjoint"]], "zonopyrobots.robots.armrobot.ZonoArmRobotLink": [[6, "zonopyrobots-robots-armrobot-zonoarmrobotlink"]], "zonopyrobots.trajectory.BernsteinArmTrajectory": [[7, "zonopyrobots-trajectory-bernsteinarmtrajectory"]], "zonopyrobots.trajectory.PiecewiseArmTrajectory": [[8, "zonopyrobots-trajectory-piecewisearmtrajectory"]], "Zonopy Robots - Extensions to Zonopy for Robotics": [[9, "zonopy-robots-extensions-to-zonopy-for-robotics"]], "Contents:": [[9, null]], "Indices and tables": [[9, "indices-and-tables"]], "Kinematics Module": [[10, "kinematics-module"]], "Robots Module": [[11, "module-zonopyrobots"]], "Arm Robots": [[11, "arm-robots"]], "Common Trajectory Types": [[12, "common-trajectory-types"]], "Kinematic Trajectory Parameterizations": [[12, "kinematic-trajectory-parameterizations"]], "Utility Functions": [[13, "utility-functions"]]}, "indexentries": {"zonoarmrobot (class in zonopyrobots)": [[1, "zonopyrobots.ZonoArmRobot"]], "__init__() (zonopyrobots.zonoarmrobot method)": [[1, "zonopyrobots.ZonoArmRobot.__init__"]], "actuated_mask (zonopyrobots.zonoarmrobot attribute)": [[1, "id0"], [1, "zonopyrobots.ZonoArmRobot.actuated_mask"]], "continuous_joints (zonopyrobots.zonoarmrobot attribute)": [[1, "id1"], [1, "zonopyrobots.ZonoArmRobot.continuous_joints"]], "device (zonopyrobots.zonoarmrobot attribute)": [[1, "id2"], [1, "zonopyrobots.ZonoArmRobot.device"]], "dof (zonopyrobots.zonoarmrobot attribute)": [[1, "id3"], [1, "zonopyrobots.ZonoArmRobot.dof"]], "dtype (zonopyrobots.zonoarmrobot attribute)": [[1, "id4"], [1, "zonopyrobots.ZonoArmRobot.dtype"]], "eff_lim (zonopyrobots.zonoarmrobot attribute)": [[1, "id5"], [1, "zonopyrobots.ZonoArmRobot.eff_lim"]], "has_closed_loop (zonopyrobots.zonoarmrobot attribute)": [[1, "id6"], [1, "zonopyrobots.ZonoArmRobot.has_closed_loop"]], "is_single_chain (zonopyrobots.zonoarmrobot attribute)": [[1, "id7"], [1, "zonopyrobots.ZonoArmRobot.is_single_chain"]], "joint_axis (zonopyrobots.zonoarmrobot attribute)": [[1, "id8"], [1, "zonopyrobots.ZonoArmRobot.joint_axis"]], "joint_data (zonopyrobots.zonoarmrobot attribute)": [[1, "id9"], [1, "zonopyrobots.ZonoArmRobot.joint_data"]], "joint_origins (zonopyrobots.zonoarmrobot attribute)": [[1, "id10"], [1, "zonopyrobots.ZonoArmRobot.joint_origins"]], "joint_origins_all (zonopyrobots.zonoarmrobot attribute)": [[1, "id11"], [1, "zonopyrobots.ZonoArmRobot.joint_origins_all"]], "link_child_joints (zonopyrobots.zonoarmrobot attribute)": [[1, "id12"], [1, "zonopyrobots.ZonoArmRobot.link_child_joints"]], "link_data (zonopyrobots.zonoarmrobot attribute)": [[1, "id13"], [1, "zonopyrobots.ZonoArmRobot.link_data"]], "link_parent_joint (zonopyrobots.zonoarmrobot attribute)": [[1, "id14"], [1, "zonopyrobots.ZonoArmRobot.link_parent_joint"]], "load() (zonopyrobots.zonoarmrobot static method)": [[1, "zonopyrobots.ZonoArmRobot.load"]], "np (zonopyrobots.zonoarmrobot attribute)": [[1, "id15"], [1, "zonopyrobots.ZonoArmRobot.np"]], "numpy() (zonopyrobots.zonoarmrobot method)": [[1, "zonopyrobots.ZonoArmRobot.numpy"]], "pos_lim (zonopyrobots.zonoarmrobot attribute)": [[1, "id16"], [1, "zonopyrobots.ZonoArmRobot.pos_lim"]], "pos_lim_mask (zonopyrobots.zonoarmrobot attribute)": [[1, "id17"], [1, "zonopyrobots.ZonoArmRobot.pos_lim_mask"]], "tensor (zonopyrobots.zonoarmrobot attribute)": [[1, "id18"], [1, "zonopyrobots.ZonoArmRobot.tensor"]], "to() (zonopyrobots.zonoarmrobot method)": [[1, "zonopyrobots.ZonoArmRobot.to"]], "urdf (zonopyrobots.zonoarmrobot attribute)": [[1, "id19"], [1, "zonopyrobots.ZonoArmRobot.urdf"]], "vel_lim (zonopyrobots.zonoarmrobot attribute)": [[1, "id20"], [1, "zonopyrobots.ZonoArmRobot.vel_lim"]], "pzrnea() (in module zonopyrobots.dynamics)": [[2, "zonopyrobots.dynamics.pzrnea"]], "forward_kinematics() (in module zonopyrobots.kinematics)": [[3, "zonopyrobots.kinematics.forward_kinematics"]], "make_cfg_dict() (in module zonopyrobots)": [[4, "zonopyrobots.make_cfg_dict"]], "zonoarmrobotjoint (class in zonopyrobots.robots.armrobot)": [[5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint"]], "__init__() (zonopyrobots.robots.armrobot.zonoarmrobotjoint method)": [[5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.__init__"]], "aabb (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id0"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.aabb"]], "axis (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id1"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.axis"]], "bounding_pz (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id2"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.bounding_pz"]], "device (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id3"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.device"]], "idx (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.idx"]], "idx (zonopyrobots.robots.armrobot.zonoarmrobotjoint property)": [[5, "id4"]], "np (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id5"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.np"]], "numpy() (zonopyrobots.robots.armrobot.zonoarmrobotjoint method)": [[5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.numpy"]], "origin (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id6"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.origin"]], "outer_pz (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id7"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.outer_pz"]], "radius (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id8"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.radius"]], "tensor (zonopyrobots.robots.armrobot.zonoarmrobotjoint attribute)": [[5, "id9"], [5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.tensor"]], "to() (zonopyrobots.robots.armrobot.zonoarmrobotjoint method)": [[5, "zonopyrobots.robots.armrobot.ZonoArmRobotJoint.to"]], "zonoarmrobotlink (class in zonopyrobots.robots.armrobot)": [[6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink"]], "__init__() (zonopyrobots.robots.armrobot.zonoarmrobotlink method)": [[6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.__init__"]], "bounding_pz (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id0"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.bounding_pz"]], "center_mass (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id1"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.center_mass"]], "device (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id2"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.device"]], "inertia (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id3"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.inertia"]], "mass (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id4"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.mass"]], "np (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id5"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.np"]], "numpy() (zonopyrobots.robots.armrobot.zonoarmrobotlink method)": [[6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.numpy"]], "tensor (zonopyrobots.robots.armrobot.zonoarmrobotlink attribute)": [[6, "id6"], [6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.tensor"]], "to() (zonopyrobots.robots.armrobot.zonoarmrobotlink method)": [[6, "zonopyrobots.robots.armrobot.ZonoArmRobotLink.to"]], "bernsteinarmtrajectory (class in zonopyrobots.trajectory)": [[7, "zonopyrobots.trajectory.BernsteinArmTrajectory"]], "__init__() (zonopyrobots.trajectory.bernsteinarmtrajectory method)": [[7, "zonopyrobots.trajectory.BernsteinArmTrajectory.__init__"]], "getreference() (zonopyrobots.trajectory.bernsteinarmtrajectory method)": [[7, "zonopyrobots.trajectory.BernsteinArmTrajectory.getReference"]], "piecewisearmtrajectory (class in zonopyrobots.trajectory)": [[8, "zonopyrobots.trajectory.PiecewiseArmTrajectory"]], "__init__() (zonopyrobots.trajectory.piecewisearmtrajectory method)": [[8, "zonopyrobots.trajectory.PiecewiseArmTrajectory.__init__"]], "getreference() (zonopyrobots.trajectory.piecewisearmtrajectory method)": [[8, "zonopyrobots.trajectory.PiecewiseArmTrajectory.getReference"]], "module": [[11, "module-zonopyrobots"]], "zonopyrobots": [[11, "module-zonopyrobots"]]}})