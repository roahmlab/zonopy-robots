# TODO CLEANUP IMPORTS & DOCUMENT
from __future__ import annotations
from typing import TYPE_CHECKING

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from zonopyrobots.robots import ZonoArmRobot
from enum import Enum
import numpy as np
import torch
import functools

from zonopyrobots.util.make_config_dict import make_cfg_dict


class FKOrigin(str, Enum):
    """The origin of the forward kinematics."""
    BASE = "base"
    WORLD = "world"


if TYPE_CHECKING:
    from typing import Union, Dict, List, Tuple
    from typing import OrderedDict as OrderedDictType
    from zonopy import polyZonotope as PZType
    from zonopy import matPolyZonotope as MPZType
    from zonopy import batchPolyZonotope as BPZType
    from zonopy import batchMatPolyZonotope as BMPZType
    from urchin import URDF
    FKOriginType = Union[str, FKOrigin, Tuple[PZType, MPZType], Tuple[BPZType, BMPZType], Tuple[torch.Tensor, torch.Tensor]]


# This is based on the Urchin's FK source
def forward_kinematics(
        rotatotopes: Union[Dict[str, Union[MPZType, BMPZType]],
                           List[Union[MPZType, BMPZType]]],
        robot: ZonoArmRobot,
        zono_order: int = 20,
        links: List[str] = None,
        origin: FKOriginType = FKOrigin.BASE,
        ) -> OrderedDictType[str, Union[Tuple[PZType, MPZType],
                                        Tuple[BPZType, BMPZType]]]:
    """Computes the forward kinematics of a robot.

    Args:
        rotatotopes (Union[Dict[str, Union[MPZType, BMPZType]], List[Union[MPZType, BMPZType]]]): The rotatotopes.
        robot (ZonoArmRobot): The robot.
        zono_order (int, optional): The zonotope order. Defaults to 20.
        links (List[str], optional): The links. Defaults to None.
        origin (FKOriginType, optional): The origin of the forward kinematics. Defaults to FKOrigin.BASE. Can be a string, FKOrigin, or a tuple of position and rotation.

    Returns:
        OrderedDictType[str, Union[Tuple[PZType, MPZType], Tuple[BPZType, BMPZType]]]: The forward kinematics.
            Each value is the position and rotation of the link in the base frame.
    """
    if robot.is_single_chain and isinstance(rotatotopes, list) and len(rotatotopes) == robot.dof:
        return forward_kinematics_single(rotatotopes, robot, zono_order, links, origin)
    # Create the rotato config dictionary
    cfg_map = make_cfg_dict(rotatotopes, robot.urdf)
    urdf = robot.urdf

    # Get our output link set, assume it's all links if unspecified
    if links is not None:
        ext_link_set, link_set = __get_ext_link_set(urdf, links)
    else:
        link_set = __all_links_as_set(urdf)
        ext_link_set = set()
    
    # Setup initial condition
    bpz_mask = np.array([isinstance(rotato, batchMatPolyZonotope) for rotato in rotatotopes])
    batch_shape = None
    if np.any(bpz_mask):
        for idx, tf in enumerate(bpz_mask):
            if tf:
                batch_shape = rotatotopes[idx].batch_shape
                break
        base_pos = batchPolyZonotope.zeros(batch_shape, 3, dtype=robot.dtype, device=robot.device)
        base_rot = batchMatPolyZonotope.eye(batch_shape, 3, dtype=robot.dtype, device=robot.device)
    else:
        base_pos = polyZonotope.zeros(3, dtype=robot.dtype, device=robot.device)
        base_rot = matPolyZonotope.eye(3, dtype=robot.dtype, device=robot.device)

    # Compute forward kinematics in reverse topological order, base to end
    fk = OrderedDict()

    # Add the base link
    topo_itr = iter(urdf._reverse_topo)
    base_link = next(topo_itr)
    if origin == FKOrigin.BASE:
        fk[base_link] = (base_pos, base_rot)
    elif origin == FKOrigin.WORLD:
        pos = robot.origin_rot@base_pos + robot.origin_pos
        rot = robot.origin_rot@base_rot
        fk[base_link] = (pos, rot)
    else:
        origin_pos, origin_rot = origin
        pos = origin_rot@base_pos + origin_pos
        rot = origin_rot@base_rot
        fk[base_link] =(pos, rot)

    # do the rest
    for lnk in topo_itr:
        if lnk in ext_link_set:
            continue
        # Get the path back to the base and build with that
        path = urdf._paths_to_base[lnk]
        pos = base_pos
        rot = base_rot
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
            if parent in fk:
                parent_pos, parent_rot = fk[parent]
                pos = parent_rot@pos + parent_pos
                rot = parent_rot@rot
                break

        # Save the values & reduce
        fk[lnk] = (pos.reduce_indep(zono_order), rot.reduce_indep(zono_order))

    # get the appropriate subset for the ordered dict
    fk = OrderedDict((lnk.name, fk[lnk]) for lnk in fk if lnk in link_set)
    return fk

def forward_kinematics_single(
        rotatotopes: List[Union[MPZType, BMPZType]],
        robot: ZonoArmRobot,
        zono_order: int = 20,
        links: List[str] = None,
        origin: FKOriginType = FKOrigin.BASE,
        ) -> OrderedDictType[str, Union[Tuple[PZType, MPZType],
                                        Tuple[BPZType, BMPZType]]]:
    urdf = robot.urdf

    # Get our output link set, assume it's all links if unspecified
    if links is not None:
        link_set = __make_link_set(urdf, links)
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
    
    # Consider origin option
    if origin == FKOrigin.BASE:
        pass
    elif origin == FKOrigin.WORLD:
        link_pos[0] = robot.origin_rot@link_pos[0] + robot.origin_pos
        link_rot[0] = robot.origin_rot@link_rot[0]
    else:
        origin_pos, origin_rot = origin
        link_pos[0] = origin_rot@link_pos[0] + origin_pos
        link_rot[0] = origin_rot@link_rot[0]
    
    # Create the initial link set
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


@functools.lru_cache
def __make_link_set(robot: URDF, incl_link_names: List[str]) -> set:
    if incl_link_names is None:
        return __all_links_as_set(robot)
    
    incl_links = set()
    for lnk in incl_link_names:
        incl_links.add(robot._link_map[lnk])
    return incl_links


@functools.lru_cache
def __get_ext_link_set(robot: URDF, incl_link_names: List[str]) -> Tuple[set, set]:
    import networkx as nx
    incl_links = __make_link_set(robot, incl_link_names)
    all_links = __all_links_as_set(robot)

    if incl_link_names is None:
        return set(), all_links
    
    all_int_nodes = set()
    for target in incl_links:
      for path in nx.all_simple_paths(robot._G, source=target, target=robot.base_link):
        all_int_nodes.update(path)
    return all_links - all_int_nodes, incl_links
