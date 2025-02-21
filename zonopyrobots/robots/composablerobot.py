from __future__ import annotations
from typing import TYPE_CHECKING

from urchin import URDF
import numpy as np
import torch
import copy

from zonopyrobots.robots.baserobot import BaseZonoRobot
from zonopyrobots.robots.utils import resolve_device_type

if TYPE_CHECKING:
    from urchin import Link
    from typing import Self
    from torch import device as torch_device, dtype as torch_dtype, Tensor
    from numpy import dtype as np_dtype, ndarray
    from numpy.typing import ArrayLike

class ComposedZonoRobot(BaseZonoRobot):
    """ A class for a robot composed of multiple zonotopic robots.
    
    This class is a wrapper around the URDF class from urchin, with extended functionality
    to merge together multiple Zono-type robots. It takes in the base robot which will be
    used as the base of the composed robot, and then additional robots which will be
    merged together with the base robot at specific links. It also helps manage which
    device various values are on.

    Attributes:
        base_robot: The base robot
        robots: Dictionary of all the robots in the composed robot as {robot_object: (link_obj, link_frame)}
        robot_map: Dictionary of robot names to robot objects

    Inherited Attributes:
        name: The name of the robot. The URDF name is used if not specified. If
            specified, the URDF name is overwritten.
        urdf: The urchin URDF object for the whole robot (subrobot URDF's are preserved)
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

    Methods:
        add_robot: Add a robot to the composed robot
        __getitem__: Get a robot by name
    """
    __slots__ = [
        'base_robot',
        'robots',
        'robot_map',
        '__robots',
        '__robot_map',
        '__robots_np',
        '__robot_map_np',
    ]

    base_robot: BaseZonoRobot
    robots: dict[BaseZonoRobot, tuple[str, ArrayLike]]
    robot_map: dict[str, BaseZonoRobot]
    __robots: dict[BaseZonoRobot, tuple[str, Tensor]]
    __robot_map: dict[str, BaseZonoRobot]
    __robots_np: dict[BaseZonoRobot, tuple[str, ndarray]]
    __robot_map_np: dict[str, BaseZonoRobot]

    def __init__(
            self,
            base_robot: BaseZonoRobot,
            robots: dict[BaseZonoRobot, tuple[Link | str, ArrayLike | None] | Link | str],
            name: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ):

        # Resolve the device and types
        device_dtypes = resolve_device_type(device, dtype, itype)

        # Create the merged URDF and robots dictionary
        self.base_robot = base_robot.copy(device=device, dtype=dtype, itype=itype)
        combined_urdf = base_robot.urdf.copy()
        if name is None:
            name = base_robot.urdf.name
        # initialize the robots dictionary with the base robot at the base link with the identity pose
        base_origin = np.eye(4, dtype=device_dtypes.np_dtype)
        self.__robots_np = {self.base_robot.np: (combined_urdf.base_link.name, base_origin)}
        self.__robots = {self.base_robot.tensor: (combined_urdf.base_link.name, torch.from_numpy(base_origin))}
        self.__robot_map_np = {self.base_robot.name: self.base_robot.np}
        self.__robot_map = {self.base_robot.name: self.base_robot.tensor}
        total_dof = base_robot.dof
        # Merge the robots together
        for rob in robots:
            # unpack the connection data
            link = robots[rob]
            if isinstance(link, (tuple, list)):
                link, link_frame = link
            else:
                link_frame = None
            # get a 4x4 pose matrix from None, a 6x1 pose vector of xyzrpy, or a 4x4 pose matrix
            from urchin.utils import configure_origin
            link_frame = configure_origin(link_frame).astype(device_dtypes.np_dtype)
            # resolve the link object if it's a string
            if not isinstance(link, str):
                link = link.name
            # Copy robot and avoid name collisions
            save_rob = rob.copy(device=device, dtype=dtype, itype=itype)
            if save_rob.name in self.__robot_map:
                i = 1
                while f"{save_rob.name}-{i}" in self.__robot_map:
                    i += 1
                save_rob.name = f"{save_rob.name}-{i}"
            # join the URDFs with a prefix for the added robot
            prefix = f"{link}_{save_rob.name}_"
            combined_urdf = combined_urdf.join(save_rob.urdf, link, link_frame, prefix=prefix)
            # save the robot and connection data
            self.__robots_np[save_rob.np] = (link, link_frame)
            self.__robots[save_rob.tensor] = (link, torch.from_numpy(link_frame))
            self.__robot_map_np[save_rob.name] = save_rob.np
            self.__robot_map[save_rob.name] = save_rob.tensor
            total_dof += rob.dof
        combined_urdf.name = name

        # Initialize the base class with the combined URDF
        super().__init__(combined_urdf, name=name, origin_pos=origin_pos, origin_rot=origin_rot, device=device, dtype=dtype, itype=itype)
        self.dof = total_dof

        # Assign aliases
        self.np = self.numpy()
        self.tensor = self.to(device='cpu')
        self.to(device=device_dtypes.device, inplace=True)
    
    def copy(self, name=None, device=None, dtype=None, itype=None):
        """ Copy the robot """
        robots_in = copy.copy(self.np.robots)
        robots_in.pop(self.base_robot.np)
        return ComposedZonoRobot(
            self.base_robot,
            robots_in,
            name=name,
            origin_pos=self.origin_pos,
            origin_rot=self.origin_rot,
            device=device,
            dtype=dtype,
            itype=itype,
        )

    @property
    def name(self):
        return self.urdf.name

    @name.setter
    def name(self, value):
        self.urdf.name = str(value)
    
    @staticmethod
    def load(
            base_robot: BaseZonoRobot,
            robots: dict[BaseZonoRobot, tuple[Link | str, ArrayLike | None] | Link | str],
            name: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ):
        return ComposedZonoRobot(
            base_robot,
            robots,
            name=name,
            origin_pos=origin_pos,
            origin_rot=origin_rot,
            device=device,
            dtype=dtype,
            itype=itype,
        )
    
    def add_robot(self, robot: BaseZonoRobot, link: Link | str, link_frame: ArrayLike | None = None):
        """ Add a robot to the composed robot """
        # get a 4x4 pose matrix from None, a 6x1 pose vector of xyzrpy, or a 4x4 pose matrix
        from urchin.utils import configure_origin
        np_dtype = self._basetype.numpy().dtype
        link_frame = configure_origin(link_frame).astype(np_dtype)
        link_frame_tensor = torch.from_numpy(link_frame)
        # resolve the link object if it's a string
        if not isinstance(link, str):
            link = link.name
        # Copy robot and avoid name collisions
        save_rob = robot.copy(device=self.device, dtype=self.dtype, itype=self.itype)
        if save_rob.name in self.__robot_map:
            i = 1
            while f"{save_rob.name}-{i}" in self.__robot_map:
                i += 1
            save_rob.name = f"{save_rob.name}-{i}"
        # join the URDFs with a prefix for the added robot
        prefix = f"{link}_{save_rob.name}_"
        new_urdf = self.urdf.join(save_rob.urdf, link, link_frame, prefix=prefix)
        # save the robot and connection data
        self.__robots_np[save_rob.np] = (link, link_frame)
        self.__robots[save_rob.tensor] = (link, link_frame_tensor)
        self.__robot_map_np[save_rob.name] = save_rob.np
        self.__robot_map[save_rob.name] = save_rob.tensor
        if self.device is not None and self.device != torch.device('cpu'):
            self.robots[save_rob] = (link, link_frame_tensor.to(device=self.device))
            self.robot_map[save_rob.name] = save_rob
        # update values (because we manually update the np and tensor mapped version, update their values)
        self.urdf = new_urdf
        self.np.urdf = new_urdf
        self.tensor.urdf = new_urdf
        self.dof += robot.dof
        self.np.dof = self.dof
        self.tensor.dof = self.dof

    def numpy(self) -> Self:
        """ Convert the robot to numpy """
        ret = super().numpy()
        ret.robots = self.__robots_np
        ret.robot_map = self.__robot_map_np
        return ret
    
    def to(self, device=None, inplace=False):
        """ Convert the robot to a specific device """
        ret = super().to(device=device, inplace=inplace)
        if inplace:
            ret.robots = self.__robots
            ret.robot_map = self.__robot_map
            for rob_name in ret.robot_map:
                ret.robot_map[rob_name] = ret.robot_map[rob_name].to(device=device, inplace=inplace)
        else:
            ret.robots = self.__robots.copy()
            ret.robot_map = self.__robot_map.copy()
            for rob_name in ret.robot_map:
                mapped_robot_data = ret.robots.pop(ret.robot_map[rob_name])
                ret.robot_map[rob_name] = ret.robot_map[rob_name].to(device=device)
                ret.robots[ret.robot_map[rob_name]] = mapped_robot_data
        return ret

    def __getitem__(self, key: str) -> BaseZonoRobot:
        return self.robot_map[key]

if __name__ == '__main__':
    import os
    import zonopyrobots as zpr
    basedirname = os.path.dirname(zpr.__file__)
    a = zpr.ZonoArmRobot(URDF.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf')))
    b = ComposedZonoRobot(a, {a:["base_link", [0,1,0,0,0,0]]})
    b.add_robot(a, "base_link", [0,-1,0,0,0,0])
    print("test")
    pass
