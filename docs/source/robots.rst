Robots Module
=============
.. automodule:: zonopyrobots
    :members:
    :show-inheritance:
.. currentmodule:: zonopyrobots

This module contains classes which specify robot parameters.
This module also contains quick access to provided URDF's (currently just the Kinova Gen3).

Included Robot URDFS
--------------------
Paths to the URDF's are stored in ``zonopyrobots.robots.files``.
URDF's can also be opened with urchin through using ``zonopyrobots.robots.urdfs``.
URDF's opened in this way are only loaded at first access, and all available robots mirror the URDF's in ``zonopyrobots.robots.files``.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    robots.files
    robots.urdfs

Arm Robots
----------
ZonoArmRobot is also aliased as ArmZonoRobot.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ZonoArmRobot
    robots.armrobot.ZonoArmRobotLink
    robots.armrobot.ZonoArmRobotJoint

SE2 and SE3 Robots
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    SE2ZonoRobot
    SE3ZonoRobot

Composable Robots
-----------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:

    ComposedZonoRobot

Robot Base Class
----------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseZonoRobot

Robot Utility Functions
-----------------------
.. currentmodule:: zonopyrobots.robots
.. autosummary::
    :toctree: generated
    :nosignatures:

    make_urdf_fixed
    make_baselink_urdf

