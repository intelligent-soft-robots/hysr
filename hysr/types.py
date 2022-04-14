import typing
import context

StampedTrajectory = context.ball_trajectories.StampedTrajectory
""" 
A tuple which first entry is a sequence of time stamps (in microseconds),
and the second entry a numpy array which lines are 3d float positions.   
"""

StampedTrajectories = context.ball_trajectories.StampedTrajectories
"""
A sequence of StampedTrajectory.
"""

DurationPoint = context.ball_trajectories.DurationPoint
"""
tuple [time stamp (microseconds), o80.Item3dState],
o80.Item3dState has set/get_positon() and set/get_velocity
methods. 
"""

ListOrIndex = typing.Union[int, typing.Sequence[int]]
""" For functions accepting either an int or a list of int as arguments"""

AcceptedNbOfBalls = typing.Literal[3, 10, 20, 50, 100]
""" The underlying C++ API does not support any number of extra
  balls per ExtraBallsSet. Here the only accepted values. 
"""

Point3D = typing.Tuple[float, float, float]
""" For 3d position or 3d velocities  """

ExtraBall = typing.Tuple[Point3D, Point3D, bool]
""" Position 3d, Velocity 3d, and contact info
  (True if the ball ever had a contact with the racket, since
  reset was called) """

JointPressures = typing.Tuple[int, int]
""" Pressures of a joint (agonist, antagonist)"""

RobotPressures = typing.Tuple[
    JointPressures, JointPressures, JointPressures, JointPressures
]
""" Pressures of a robot. Can be used for observed / desired pressures or pressure commands"""
