import typing
from dataclasses import dataclass
from .types import Pressures, Joints


@dataclass
class JointsState:

    positions : Joints = None
    velocities : Joints = None

@dataclass
class RobotState:

    desired_pressures : typing.Tuple[Pressures,Pressures] = None
    observed_pressures : typing.Tuple[Pressures,Pressures] = None
    joints : JointsState = JointsState()

    

