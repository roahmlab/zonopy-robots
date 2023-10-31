# TODO CLEANUP IMPORTS & DOCUMENT
from __future__ import annotations
from typing import TYPE_CHECKING

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from zonopyrobots.robots import ZonoArmRobot
import numpy as np
import torch

if TYPE_CHECKING:
    from typing import Union, Dict, List, Tuple
    from typing import OrderedDict as OrderedDictType
    from zonopy import polyZonotope as PZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import batchMatPolyZonotope as BMPZType

from zonopyrobots.util.make_config_dict import make_cfg_dict

# This is based on the Urchin's FK source
def forward_kinematics(
        rotatotopes: Union[Dict[str, Union[MPZType, BMPZType]],
                           List[Union[MPZType, BMPZType]]],
        robot: ZonoArmRobot,
        zono_order: int = 20,
        links: List[str] = None,
        ) -> OrderedDictType[str, Union[Tuple[PZType, MPZType],
                                        Tuple[BPZType, BMPZType]]]:
    """Computes the forward kinematics of a robot.

    Args:
        rotatotopes (Union[Dict[str, Union[MPZType, BMPZType]], List[Union[MPZType, BMPZType]]]): The rotatotopes.
        robot (ZonoArmRobot): The robot.
        zono_order (int, optional): The zonotope order. Defaults to 20.
        links (List[str], optional): The links. Defaults to None.

    Returns:
        OrderedDictType[str, Union[Tuple[PZType, MPZType], Tuple[BPZType, BMPZType]]]: The forward kinematics.
    """
    # Create the rotato config dictionary
    cfg_map = make_cfg_dict(rotatotopes, robot.urdf)
    urdf = robot.urdf

    # Get our output link set, assume it's all links if unspecified
    link_set = set()
    if links is not None:
        for lnk in links:
            link_set.add(urdf._link_map[lnk])
    else:
        link_set = urdf.links

    # Compute forward kinematics in reverse topological order, base to end
    fk = OrderedDict()
    for lnk in urdf._reverse_topo:
        if lnk not in link_set:
            continue
        # Get the path back to the base and build with that
        path = urdf._paths_to_base[lnk]
        pos = polyZonotope.zeros(3, dtype=robot.dtype, device=robot.device)
        rot = matPolyZonotope.eye(3, dtype=robot.dtype, device=robot.device)
        for i in range(len(path) - 1):
            child = path[i]
            parent = path[i + 1]
            joint = urdf._G.get_edge_data(child, parent)["joint"]

            rotato_cfg = None
            if joint.mimic is not None:
                mimic_joint = urdf._joint_map[joint.mimic.joint]
                if mimic_joint.name in cfg_map:
                    rotato_cfg = cfg_map[mimic_joint.name]
                    rotato_cfg = joint.mimic.multiplier * rotato_cfg + joint.mimic.offset
                    import warnings
                    warnings.warn('Mimic joint may not work')
            elif joint.name in cfg_map:
                rotato_cfg = cfg_map[joint.name]
            else:
                rotato_cfg = torch.eye(3, dtype=robot.dtype, device=robot.device)
            
            # Get the transform for the joint
            # joint_rot = torch.as_tensor(joint.origin[0:3,0:3], dtype=torch.float64)
            # joint_pos = torch.as_tensor(joint.origin[0:3,3], dtype=torch.float64)
            joint_rot = robot.joint_data[joint].origin[0:3,0:3]
            joint_pos = robot.joint_data[joint].origin[0:3,3]
            joint_rot = joint_rot@rotato_cfg
            
            # We are moving from child to parent, so apply in reverse
            pos = joint_rot@pos + joint_pos
            rot = joint_rot@rot

            # Check existing FK to see if we can exit early
            if parent.name in fk:
                parent_pos, parent_rot = fk[parent.name]
                pos = parent_rot@pos + parent_pos
                rot = parent_rot@rot
                break

        # Save the values & reduce
        fk[lnk.name] = (pos.reduce_indep(zono_order), rot.reduce_indep(zono_order))

    return fk