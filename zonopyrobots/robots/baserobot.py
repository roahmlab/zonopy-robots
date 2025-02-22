from __future__ import annotations
from typing import TYPE_CHECKING

from urchin import URDF
import copy
import torch
import numpy as np

from zonopyrobots.robots.utils import resolve_device_type

if TYPE_CHECKING:
    from typing import Self
    from torch import device as torch_device, dtype as torch_dtype, Tensor
    from numpy import dtype as np_dtype, ndarray
    from numpy.typing import ArrayLike


class BaseZonoRobot:
    """ Base robot definition for use with zonopy
    
    This class is meant to be subclassed and extended to provide
    the extra caching functionality useful for zonopy. It provides
    the basic structure and slots for the robot.

    """
    __slots__ = [
        'urdf',
        'dof',
        'np',
        'tensor',
        'device',
        'dtype',
        'itype',
        'origin_pos',
        'origin_rot',
        '_basetype',
        '_baseitype',
        '__origin_pos',
        '__origin_rot',
        '__origin_pos_np',
        '__origin_rot_np',
    ]

    urdf: URDF
    """The urchin URDF object for the robot"""

    dof: int
    """The degrees of freedom of the robot"""

    np: Self
    """A numpy version of the object"""

    tensor: Self
    """A CPU torch version of the object"""

    device: torch_device | None
    """The device the primary data is on. None if numpy."""
    
    dtype: torch_dtype | np_dtype
    """The pytorch or numpy dtype of the class"""

    itype: torch_dtype | np_dtype
    """The pytorch or numpy integer dtype of the class"""

    origin_pos: Tensor | ndarray
    """The origin position of the robot in the world frame"""

    origin_rot: Tensor | ndarray
    """The origin rotation of the robot in the world frame as a 3x3 rotation matrix"""

    _basetype: Tensor
    """The base type of the robot dtype"""

    _baseitype: Tensor
    """The base type of the robot itype"""

    __origin_pos: Tensor
    """The underlying origin position cpu tensor reference"""

    __origin_rot: Tensor
    """The underlying origin rotation cpu tensor reference"""

    __origin_pos_np: ndarray
    """The underlying origin position numpy reference and data"""

    __origin_rot_np: ndarray
    """The underlying origin rotation numpy reference and data"""

    def __init__(
            self,
            urdf: URDF | str,
            name: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ):
        """ Create a new BaseZonoRobot object

        Args:
            urdf: The URDF file or URDF object to load
            name: The name of the robot
            origin_pos: The origin position of the robot in the world frame
            origin_rot: The origin rotation of the robot in the world frame as a 3x3 rotation matrix
            device: The device to put the tensors on. None for pytorch default.
            dtype: The data type to use for the zonotopes. None for pytorch default.
            itype: The index type to use for the zonotopes. None for pytorch default.
        
        Raises:
            AssertionError: If the origin position is not a 3D vector or the origin rotation is not a 3x3 matrix
            AssertionError: If the device is not a torch device or the dtype is not a torch or numpy dtype
        """

        # Resolve the device and dtype to use
        device_dtypes = resolve_device_type(device, dtype, itype)
        self._basetype = device_dtypes.basetype
        self._baseitype = device_dtypes.baseitype

        # Save the URDF and DOF
        if type(urdf) == str:
            robot = URDF.load(urdf)
            if name is not None:
                robot.name = name
        else:
            robot = urdf.copy(name=name)
        self.urdf = robot
        self.dof = int(len(self.urdf.actuated_joints))

        # Save the origin position and rotation
        origin_pos = np.zeros(3) if origin_pos is None else origin_pos
        origin_rot = np.eye(3) if origin_rot is None else origin_rot
        self.__origin_pos_np = np.array(origin_pos, dtype=device_dtypes.np_dtype)
        self.__origin_rot_np = np.array(origin_rot, dtype=device_dtypes.np_dtype)
        self.__origin_pos = torch.from_numpy(self.__origin_pos_np)
        self.__origin_rot = torch.from_numpy(self.__origin_rot_np)
        assert self.__origin_pos.shape == (3,), "Origin position must be a 3D vector"
        assert self.__origin_rot.shape == (3, 3), "Origin rotation must be a 3x3 matrix"

        # Create the np and tensor versions
        self.np = BaseZonoRobot.numpy(self)
        self.tensor = BaseZonoRobot.to(self, device='cpu')
        BaseZonoRobot.to(self, device=device_dtypes.device, inplace=True)

    @property
    def name(self) -> str:
        """ The name of the robot. The URDF name is used if not specified. If
        specified, the URDF name is overwritten.
        """
        return self.urdf.name

    @name.setter
    def name(self, value: str) -> None:
        self.urdf.name = str(value)
    
    def update_origin(
            self,
            origin_pos: ArrayLike = None,
            origin_rot: ArrayLike = None,
        ) -> None:
        """ Update the origin position and rotation of the robot """
        # update origin_pos
        if origin_pos is not None:
            self.__origin_pos_np = np.array(origin_pos, dtype=self._basetype.numpy().dtype)
            self.__origin_pos = torch.from_numpy(self.__origin_pos_np)
            assert self.__origin_pos.shape == (3,), "Origin position must be a 3D vector"
            if self.device is not None:
                self.origin_pos = self.__origin_pos.to(device=self.device)
            else:    
                self.origin_pos = self.__origin_pos_np
        # update origin_rot
        if origin_rot is not None:
            self.__origin_rot_np = np.array(origin_rot, dtype=self._basetype.numpy().dtype)
            self.__origin_rot = torch.from_numpy(self.__origin_rot_np)
            assert self.__origin_rot.shape == (3, 3), "Origin rotation must be a 3x3 matrix"
            if self.device is not None:
                self.origin_rot = self.__origin_rot.to(device=self.device)
            else:
                self.origin_rot = self.__origin_rot_np
        # refresh np and tensor versions
        self.np = self.numpy()
        self.tensor = self.to(device='cpu')

    def copy(
            self,
            name: str | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ) -> Self:
        """ Create a deep copy of the robot by copying all data while respecting overlapping memory
        
        Args:
            name: The name of the robot. None to keep the same name.
            device: The device to put the tensors on. None to keep the same device.
            dtype: The data type to use for the zonotopes. None to keep the same dtype.
            itype: The index type to use for the zonotopes. None to keep the same itype.
        """
        # create a shallow copy of self for the base container
        ret = copy.copy(self)

        # Resolve the device and dtype to use
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if itype is None:
            itype = self.itype
        device_dtypes = resolve_device_type(device, dtype, itype)
        ret._basetype = device_dtypes.basetype
        ret._baseitype = device_dtypes.baseitype

        # Copy what needs to be copied
        ret.urdf = self.urdf.copy(name=name)
        ret.__origin_pos_np = np.array(self.__origin_pos_np, dtype=device_dtypes.np_dtype)
        ret.__origin_rot_np = np.array(self.__origin_rot_np, dtype=device_dtypes.np_dtype)
        ret.__origin_pos = torch.from_numpy(ret.__origin_pos_np)
        ret.__origin_rot = torch.from_numpy(ret.__origin_rot_np)

        # Update associations
        ret.np = BaseZonoRobot.numpy(ret)
        ret.tensor = BaseZonoRobot.to(ret, device='cpu')
        BaseZonoRobot.to(ret, device=device_dtypes.device, inplace=True)
        return ret
    
    @staticmethod
    def load(
            robot: URDF | str,
            name: str | None = None,
            origin_pos: ArrayLike | None = None,
            origin_rot: ArrayLike | None = None,
            device: torch_device | None = None,
            dtype: torch_dtype | np_dtype | None = None,
            itype: torch_dtype | np_dtype | None = None,
        ) -> Self:
        """ Load a robot from a URDF file or URDF object

        Alias for the constructor for backwards compatibility and to allow for
        a more consistent API with urchin.

        Args:
            robot: The URDF file or URDF object to load
            name: The name of the robot
            origin_pos: The origin position of the robot in the world frame
            origin_rot: The origin rotation of the robot in the world frame as a 3x3 rotation matrix
            device: The device to put the tensors on. None for pytorch default.
            dtype: The data type to use for the zonotopes. None for pytorch default.
            itype: The index type to use for the zonotopes. None for pytorch default.

        Returns:
            A BaseZonoRobot object
        """
        return BaseZonoRobot(
            robot,
            name=name,
            origin_pos=origin_pos,
            origin_rot=origin_rot,
            device=device,
            dtype=dtype,
            itype=itype,
        )

    def numpy(self) -> Self:
        """ Convert and return a numpy version of the object. Does not copy underlying array data.
        
        Returns:
            A numpy version of the object
        """
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to the origin data
        ret.origin_pos = self.__origin_pos_np
        ret.origin_rot = self.__origin_rot_np

        # save the device parameter
        ret.device = None
        ret.dtype = self._basetype.numpy().dtype
        ret.itype = self._baseitype.numpy().dtype
        return ret

    def to(self, device: torch_device = None, inplace: bool = False) -> Self:
        """ Convert and return a torch version of the object. Does not copy data if device is the same.
        
        Args:
            device: The device to put the tensors on. None for pytorch default.
            inplace: If true, update the current object in place. If false, return a new object.
        
        Returns:
            A torch version of the object
        """
        if inplace:
            ret = self
        else:
            # create a shallow copy of self
            ret = copy.copy(self)
        # update references to the origin data
        ret.origin_pos = self.__origin_pos.to(device=device)
        ret.origin_rot = self.__origin_rot.to(device=device)

        # save the device parameter
        ret.device = device
        ret.dtype = self._basetype.dtype
        ret.itype = self._baseitype.dtype
        return ret

if __name__ == '__main__':
    import os
    import zonopyrobots as zpr
    basedirname = os.path.dirname(zpr.__file__)
    a = BaseZonoRobot(URDF.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf')))
    b = a.copy()
    b.update_origin(origin_pos=[1, 2, 3], origin_rot=np.eye(3))
    print("test")
    pass
