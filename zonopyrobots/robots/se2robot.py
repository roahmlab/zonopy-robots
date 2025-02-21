from __future__ import annotations
from typing import TYPE_CHECKING

from urchin import URDF
from zonopyrobots.robots.baserobot import BaseZonoRobot
from zonopyrobots.robots.utils import make_urdf_fixed

if TYPE_CHECKING:
    from typing import Self
    from torch import device as torch_device, dtype as torch_dtype
    from numpy import dtype as np_dtype
    from numpy.typing import ArrayLike

class SE2ZonoRobot(BaseZonoRobot):
    """ A robot with SE(2) kinematics for use with zonopy
    
    This robot is a 3 DOF robot with a planar joint and a continuous joint.
    The planar joint is a planar joint that moves in the x-y plane.
    The continuous joint is a continuous joint that rotates around the z-axis.
    A URDF can be provided for geometry or collision checking, but all joints
    will be converted to fixed joints and the robot will be placed in an SE(2)
    base frame.

    Any FK using the SE(2) base frame will be performed with the first joint as
    the x-y position and the second joint as the rotation around the z-axis.
    
    Attributes:
        fixed_urdf: The URDF with all joints set to fixed
        original_urdf: The original URDF provided

    Inherited Attributes:
        name: The name of the robot. The URDF name is used if not specified. If
            specified, the URDF name is overwritten.
        urdf: The urchin URDF object for the robot
        dof: The degrees of freedom of the robot
        np: A numpy version of the object
        tensor: A torch version of the object
        device: The device the primary data is on. None if numpy.
        dtype: The pytorch or numpy dtype of the class
        itype: The pytorch or numpy integer type of the class
        origin_pos: The origin position of the robot in the world frame
        origin_rot: The origin rotation of the robot in the world frame as a
            3x3 rotation matrix
        _basetype: The base type of the robot dtype
        _baseitype: The base type of the robot itype
    """
    __slots__ = [
        "fixed_urdf",
        "original_urdf",
    ]

    fixed_urdf: URDF | None
    original_urdf: URDF | None

    def __init__(
            self,
            urdf: URDF | str | None = None,
            name: str | None = None,
            se2_prefix: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ):
        
        # Create a se2 base urdf programatically, then attach and lock the provided URDF
        base_urdf = _create_se2_base_urdf(name=name, prefix=se2_prefix)
        if urdf is not None:
            locked_urdf = make_urdf_fixed(urdf)
            combined_urdf = base_urdf.join(locked_urdf, base_urdf.end_links[0])
        else:
            locked_urdf = None
            combined_urdf = base_urdf

        # Call the parent constructor
        super().__init__(
            urdf=combined_urdf,
            name=name,
            origin_pos=origin_pos,
            origin_rot=origin_rot,
            device=device,
            dtype=dtype,
            itype=itype,
        )
        
        # correct the DOF and store extra URDFs
        self.dof = 3
        self.fixed_urdf = locked_urdf
        self.original_urdf = urdf

        # update references
        self.np = self.numpy()
        self.tensor = self.to(device='cpu')

    @staticmethod
    def load(
            urdf: URDF | str | None = None,
            name: str | None = None,
            se2_prefix: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ):
        """ Load a robot from a URDF file or URDF object

        Args:
            urdf: The URDF file or URDF object to load and lock. If not provided,
                a default SE(2) robot with no geometry is created.
            name: The name of the robot. "se2_robot" is used if not specified.
            se2_prefix: The prefix to use for the SE(2) base URDF. "se2_" is used
                if not specified.
            origin_pos: The origin position of the robot in the world frame
            origin_rot: The origin rotation of the robot in the world frame as a
                3x3 rotation matrix
            device: The device to put the tensors on. None for pytorch default.
            dtype: The data type to use for the zonotopes. None for pytorch default.
            itype: The index type to use for the zonotopes. None for pytorch default.
        """
        return SE2ZonoRobot(
            urdf=urdf,
            name=name,
            se2_prefix=se2_prefix,
            origin_pos=origin_pos,
            origin_rot=origin_rot,
            device=device,
            dtype=dtype,
            itype=itype,
        )
    
    def numpy(self) -> Self:
        """ Convert the robot to a numpy version """
        ret = super().numpy()
        ret.fixed_urdf = self.fixed_urdf
        ret.original_urdf = self.original_urdf
        return ret
    
    def to(self, device: torch_device = None, inplace: bool = False) -> Self:
        """ Convert the robot to a torch version """
        ret = super().to(device=device, inplace=inplace)
        ret.fixed_urdf = self.fixed_urdf
        ret.original_urdf = self.original_urdf
        return ret


def _create_se2_base_urdf(name: str | None = None, prefix: str | None = None):
    import urchin
    if name is None:
        name = "se2_robot"
    if prefix is None:
        prefix = "se2_"
    se2_base = urchin.Link(f"{prefix}base", None, [], [])
    se2_rot = urchin.Link(f"{prefix}rot", None, [], [])
    se2_link = urchin.Link(f"{prefix}link", None, [], [])
    se2_planar_joint = urchin.Joint(f"{prefix}planar_joint", 'planar', se2_base.name, se2_rot.name)
    se2_rot_joint = urchin.Joint(f"{prefix}rot_joint", 'continuous', se2_rot.name, se2_link.name)
    base_urdf = urchin.URDF(
        name = name,
        links = [se2_base, se2_rot, se2_link],
        joints = [se2_planar_joint, se2_rot_joint],
    )
    return base_urdf

if __name__ == '__main__':
    import os
    import zonopyrobots as zpr
    basedirname = os.path.dirname(zpr.__file__)
    a = SE2ZonoRobot(URDF.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf')))
    print("test")
    pass
