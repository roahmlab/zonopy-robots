from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import zonopy.internal as zpi
import functools

if TYPE_CHECKING:
    from urchin import URDF
    from typing import Dict, Any, Union, List
    from typing import OrderedDict as OrderedDictType
    from zonopy import polyZonotope as PZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import batchMatPolyZonotope as BMPZType


def make_cfg_dict(
        configs: Union[Dict[str, Any], List[Any], None],
        robot: URDF,
        allow_incomplete: bool = False,
        ) -> Dict[str, Any]: 
    """Helper function to create a config dictionary if a list is provided.

    Args:
        configs (Union[Dict[str, Any], List[Any], None]): The configs.
        robot (URDF): The URDF.
        allow_incomplete (bool, optional): Whether to allow incomplete list configs. Defaults to False.
    
    Returns:
        Dict[str, Any]: The config dictionary.
    """
    if isinstance(configs, dict):
        if zpi.__debug_extra__:
            assert all(isinstance(x, str) for x in configs.keys()), "Keys for the config dict are not all strings!"
    elif isinstance(configs, (list, np.ndarray)):
        # Assume that this is for all actuated joints
        joint_names = __robot_actuated_joint_names(robot)
        if allow_incomplete:
            joint_names = joint_names[:len(configs)]
        assert len(joint_names) == len(configs), "Unexpected number of configs!"
        configs = {name: cfgs for name, cfgs in zip(joint_names, configs)}
    elif configs is None:
        configs = {}
    else:
        raise TypeError
    return configs


@functools.lru_cache
def __robot_actuated_joint_names(
        robot: URDF,
    ) -> List[str]:
    """Helper function to get the joint names of a robot.

    Args:
        robot (URDF): The URDF.
    
    Returns:
        List[str]: The joint names.
    """
    return [joint.name for joint in robot.actuated_joints]