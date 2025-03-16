from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
from collections import namedtuple
from functools import lru_cache

if TYPE_CHECKING:
    from urchin import URDF

DeviceDtype = namedtuple('DeviceDtype', ['device', 'dtype', 'np_dtype', 'itype', 'np_itype', 'basetype', 'baseitype'])
@lru_cache()
def resolve_device_type(device=None, dtype=None, itype=None):
    # Resolve the dtype and device to use
    temp_device = torch.empty(0, device=device)
    device = temp_device.device
    if isinstance(dtype, np.dtype):
        temp_dtype = torch.from_numpy(np.empty(0, dtype=dtype))
    else:
        temp_dtype = torch.empty(0, device='cpu', dtype=dtype)
        temp_dtype = torch.empty(0, device='cpu', dtype=dtype)
    dtype = temp_dtype.dtype
    np_dtype = temp_dtype.numpy().dtype
    if itype is not None:
        if isinstance(itype, np.dtype):
            temp_itype = torch.from_numpy(np.empty(0, dtype=itype))
        else:
            temp_itype = torch.empty(0, dtype=itype, device='cpu')
    else:
        temp_itype = torch.tensor([0], device='cpu')
    itype = temp_itype.dtype
    np_itype = temp_itype.numpy().dtype
    assert np.issubdtype(np_itype, np.integer), "Index type must be an integer type"
    basetype = temp_dtype
    baseitype = temp_itype
    return DeviceDtype(device, dtype, np_dtype, itype, np_itype, basetype, baseitype)
    

# Function to create a 3D basis for any defining normal vector (to an arbitrary hyperplane)
# Returns the basis as column vectors
def normal_vec_to_basis(norm_vec: np.ndarray) -> np.ndarray:
    """Creates a 3D basis for any defining normal vector (to an arbitrary hyperplane).
    
    Args:
        norm_vec (np.ndarray): The normal vector.

    Returns:
        np.ndarray: The basis as column vectors.
    """
    # first normalize the vector
    norm_vec = np.array(norm_vec, dtype=float).squeeze()
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    # Helper function for simple basis with unitary elements
    def simple_basis(order):
        ret = np.eye(3)
        idx = (np.arange(3) + order) % 3
        return ret[:,idx]

    # Try to project [1, 0, 0]
    if (proj := np.dot([1.0, 0, 0], norm_vec)):
        # Use this vector to create an orthogonal component
        rej = np.array([1.0, 0, 0]) - (norm_vec * proj)
        # Case for normal vector of [1, 0, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(1)
    # If not, try to project [0, 1, 0] and do the same
    elif (proj := np.dot([0, 1.0, 0], norm_vec)):
        rej = np.array([0, 1.0, 0]) - (norm_vec * proj)
        # Case for normal vector of [0, 1, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(2)
    else:
        # Otherwise, we are dealing with normal vector of [0, 0, 1],
        # so just create the identity as the basis
        return simple_basis(3)
    
    # Find a third orthogonal vector
    cross = np.cross(rej, norm_vec)
    # Just for simplicity, we treat the cross as x, the rej as y, and the vec as z
    # in order to keep a properly left-handed basis
    cross = cross / np.linalg.norm(cross)
    rej = rej / np.linalg.norm(rej)
    return np.column_stack((cross, rej, norm_vec))


def make_urdf_fixed(urdf: URDF):
    """ Convert a given URDF to a fixed joint URDF """
    urdf = urdf.copy()
    for joint in urdf.joints:
        joint.joint_type = 'fixed'
    return urdf.copy()


def make_baselink_urdf(name: str = None, link_name: str = None) -> URDF:
    """ Create a URDF with a single base link """
    import urchin
    if name is None:
        name = "robot"
    if link_name is None:
        link_name = "base_link"
    base_link = urchin.Link(link_name, None, [], [])
    base_urdf = urchin.URDF(
        name = name,
        links = [base_link],
    )
    return base_urdf
