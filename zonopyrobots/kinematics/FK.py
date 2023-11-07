# TODO CLEANUP IMPORTS & DOCUMENT
from __future__ import annotations
from typing import TYPE_CHECKING

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from zonopyrobots.robots import ZonoArmRobot
import numpy as np
import torch
import functools

if TYPE_CHECKING:
    from typing import Union, Dict, List, Tuple
    from typing import OrderedDict as OrderedDictType
    from zonopy import polyZonotope as PZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import batchMatPolyZonotope as BMPZType
    from urchin import URDF

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
    if robot.is_single_chain and len(rotatotopes) == robot.dof:
        return forward_kinematics_single(rotatotopes, robot, zono_order, links)
    # Create the rotato config dictionary
    cfg_map = make_cfg_dict(rotatotopes, robot.urdf)
    urdf = robot.urdf

    # Get our output link set, assume it's all links if unspecified
    if links is not None:
        link_set = set()
        for lnk in links:
            link_set.add(urdf._link_map[lnk])
    else:
        link_set = __all_links_as_set(urdf)

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

def forward_kinematics_single(
        rotatotopes: List[Union[MPZType, BMPZType]],
        robot: ZonoArmRobot,
        zono_order: int = 20,
        links: List[str] = None,
        ) -> OrderedDictType[str, Union[Tuple[PZType, MPZType],
                                        Tuple[BPZType, BMPZType]]]:
    urdf = robot.urdf

    # Get our output link set, assume it's all links if unspecified
    if links is not None:
        link_set = set()
        for lnk in links:
            link_set.add(urdf._link_map[lnk])
    else:
        link_set = __all_links_as_set(urdf)

    # Do the first pass
    n_all_joints = len(urdf.joints)
    rot = [None]*n_all_joints
    idx_in = 0
    for idx_out, actuated in enumerate(robot.np.actuated_mask):
        if actuated:
            rot[idx_out] = rotatotopes[idx_in]
            idx_in += 1
    all_rot = [robot.joint_origins_all[i,0:3,0:3]@rot[i] if robot.np.actuated_mask[i] else robot.joint_origins_all[i,0:3,0:3] for i in range(n_all_joints)]
    all_pos = list(robot.joint_origins_all[:,0:3,3])

    # Setup initial condition
    bpz_mask = np.array([isinstance(rotato, batchMatPolyZonotope) for rotato in rotatotopes])
    if np.any(bpz_mask):
        for idx, tf in enumerate(bpz_mask):
            if tf:
                batch_shape = rotatotopes[idx].batch_shape
                break
        link_rot = [batchMatPolyZonotope.eye(batch_shape, 3, dtype=robot.dtype, device=robot.device)]
        link_pos = [batchPolyZonotope.zeros(batch_shape, 3, dtype=robot.dtype, device=robot.device)]
    else:
        link_rot = [matPolyZonotope.eye(3, dtype=robot.dtype, device=robot.device)]
        link_pos = [polyZonotope.zeros(3, dtype=robot.dtype, device=robot.device)]
    link_rot.extend(all_rot)
    link_pos.extend(all_pos)
    
    # Forward kinematics
    for i in range(n_all_joints):
        parent_pos = link_pos[i]
        parent_rot = link_rot[i]
        link_pos[i+1] = (parent_rot@link_pos[i+1] + parent_pos).reduce_indep(zono_order)
        link_rot[i+1] = (parent_rot@link_rot[i+1]).reduce_indep(zono_order)

    # Generator to make the fk
    fk = OrderedDict((link.name, (link_pos[idx], link_rot[idx])) for idx, link in enumerate(urdf._reverse_topo) if link in link_set)
    return fk


@functools.lru_cache
def __all_links_as_set(robot: URDF) -> set:
    return set(robot.links)