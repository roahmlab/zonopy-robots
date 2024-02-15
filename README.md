# zonopy-robots

Zonopy-robots is a Python package that extends the capabilities of the [zonopy](https://github.com/roahmlab/zonopy) library to facilitate robotics research.
Where zonopy introduces continuous set types such as zonotopes into Python, this package provides additional tools for kinematics, dynamics, and loading of URDF's.
For now, this package is particularly focused on the manipulation of arm robots.
It integrates with [urchin](https://github.com/fishbotics/urchin) for loading and working with URDF (Unified Robot Description Format) models of articulated robots.

## Features

- **Continuous Set Types**: Support for zonotopes and other continuous set representations for geometric operations.
- **Robot Kinematics and Dynamics**: Functions for calculating the kinematics and dynamics of robotic mechanisms.
- **URDF Loader**: Integration with `urchin` to load and manipulate URDF models of articulated arm robots.

## Requirements

- zonopy: The base library for continuous set operations.
- urchin: Required for URDF model loading and manipulation.

## Installation

Either clone the latest version or a specific tag of this repo and inside of the repo path run:

```
pip install -e .
```

## Quick Start

Here's a simple example to get started with `zonopy-robots`:

```python
# Example: Load a robot model and build the forward kinematics of the robot joints following the ARMTD piecewise trajectory
import numpy as np
import zonopyrobots as zpr

# There is an example URDF in this repo - kinova from kinova robotics' ros_kortex
import os
basedirname = os.path.dirname(zpr.__file__)
urdfpath = os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf')

# Load the robot with zonopy-robots. we can access the underlying urchin representation with
# `robot_model.urdf`, and we visualize it with urchin here.
robot_model = zpr.ZonoArmRobot.load(urdfpath, device="cpu")
robot_model.urdf.show()
# Note that this line sometimes stops the script for some reason

# Prepare the JRS generator
traj_class = zpr.trajectory.PiecewiseArmTrajectory
jrs_gen = zpr.JrsGenerator(robot_model, traj_class=traj_class, param_range=np.pi/6, batched=True)

# Initial configuration for the online JRS
q = np.zeros(robot_model.dof)
qd = q
qdd = q

# Generate the JRS for that configuration
rotatotopes = jrs_gen.gen_JRS(q, qd, qdd, only_R=True)

# Get the position PZ's and rotation MPZ's
fk = zpr.kinematics.forward_kinematics(rotatotopes, robot_model)
# The result is a dictionary of link names and their position and rotation PZ's.
```

<!-- ## Contributing

Contributions to `zonopy-robots` are welcome! Whether it's adding new features, improving existing ones, or reporting bugs, your input helps make this tool better for the research community.

## License

`zonopy-robots` is released under the MIT License. See the LICENSE file for more details. #still need to figure this out -->

## How to Cite

If you use zonopy or zonopy-robots in your research, please cite one or more of the following papers:

Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres. J. Michaux, A. Li, Q. Chen, C. Chen, B. Zhang, and R. Vasudevan. ArXiv, 2024. (https://arxiv.org/abs/2402.08857)
```bibtex
@article{michaux2024sparrows,
  title={Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres},
  author={Jonathan Michaux and Adam Li and Qingyi Chen and Che Chen and Bohao Zhang and Ram Vasudevan},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.08857},
  url={https://arxiv.org/abs/2402.08857}}
```

Reachability-based Trajectory Design with Neural Implicit Safety Constraints. J. B. Michaux, Y. S. Kwon, Q. Chen, and R. Vasudevan. Robotics: Science and Systems, 2023. (https://www.roboticsproceedings.org/rss19/p062.pdf)
```bibtex
@inproceedings{michaux2023rdf,
  title={{Reachability-based Trajectory Design with Neural Implicit Safety Constraints}},
  author={Jonathan B Michaux AND Yong Seok Kwon AND Qingyi Chen AND Ram Vasudevan},
  booktitle={Proceedings of Robotics: Science and Systems},
  year={2023},
  address={Daegu, Republic of Korea},
  doi={10.15607/RSS.2023.XIX.062}}
```
