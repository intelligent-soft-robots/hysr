from dataclasses import dataclass
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

Orientation3D = typing.Tuple[
    float, float, float, float, float, float, float, float, float
]
""" 3d orientation as 9d float tuple (flatted matrix) """

CartesianPose = typing.Tuple[Point3D, Orientation3D]
""" for the 3d cartesian position and orientation of the racket """

ExtraBall = typing.Tuple[Point3D, Point3D, bool]
""" Position 3d, Velocity 3d, and contact info
  (True if the ball ever had a contact with the racket, since
  reset was called) """

JointStates = typing.Tuple[float, float, float, float]
""" For 4d (one per joint) angular positions or angular velocities of joints"""

JointPressures = typing.Tuple[int, int]
""" Pressures of a joint (agonist, antagonist)"""

RobotPressures = typing.Tuple[
    JointPressures, JointPressures, JointPressures, JointPressures
]
""" Pressures of a robot. Can be used for observed / desired pressures or pressure commands"""


@dataclass
class MainSimState:
    """
    Snapshot state of a main simulation
    (ball + joint controlled robot)

    Attributes
    ----------
    ball_position: 3d float
      position of the ball
    ball_velocity: 3d float
      velocity of the ball
    joint_positions: 
       tuple of joint positions of the robot, in radian
    joint_velocities:
       tuple of joint velocities of the robot, in radian per seconds
    contact:
       instance with these attributes:
        - contact_occured : if true, at least one contact has occured
        - position: if contact occured, the 3d position of the first contact
        - time_stamp: if contact occured, the time stamp of the fist contact
        - minimal_distance: if contact did not occure, the minimal distance
                            between the two items
        - disabled : true if contact detection has been disabled
        Note that once a contact occured, the ball is no longer controlled by
        o80 (i.e. the load method of this class will have no effect for the
        corresponding ball), but by mujoco engine (until the method
        reset_contacts of this class is called)
    iteration: 
      iteration of the mujoco simulation
    time_stamp: 
      time stamp of the mujoco simulation (nanoseconds)
    """

    ball_position: Point3D = None
    ball_velocity: Point3D = None
    joint_positions: JointStates = None
    joint_velocities: JointStates = None
    racket_cartesian: CartesianPose = None
    contact: context.ContactInformation = None
    iteration: int = None
    time_stamp: int = None


@dataclass
class ExtraBallsState:
    """
    Snapshot state of an ExtraBallsState.

    Attributes
    ----------
    ball_positions: list of 3d positions
      positions of the balls
    ball_velocities: list of 3d positions 
      velocities of the balls
    contacts: bool
       if True, the corresponding ball had a 
       contact with the racket since the last call to reset
    joint_positions: 
       tuple of joint positions of the robot, in radian
    joint_velocities:
       tuple of joint velocities of the robot, in radian per seconds
    racket_cartesian: 3d position
        position of the racket
    iteration: int
      iteration of the mujoco simulation
    time_stamp: int
      time stamp of the mujoco simulation (nanoseconds)
    """

    ball_positions: typing.Sequence[Point3D] = None
    ball_velocities: typing.Sequence[Point3D] = None
    contacts: typing.Sequence[bool] = None
    joint_positions: JointStates = None
    joint_velocities: JointStates = None
    racket_cartesian: Point3D = None
    iteration: int = None
    time_stamp: int = None


@dataclass
class PressureRobotState:
    """ Snapshot state of a pressure controlled robot

    Attributes
    ----------
    positions: 
      for each joint, in radian
    velocities:
      for each joint, in radian per second
    desired_pressures:
      for each joint, agonist and antagonist pressures
    observed_pressures:
      for each joint, agonist and antagonist pressures
    iteration:
      iteration of the backend
    time_stamp: 
      time_stamp of the backend, in nanoseconds
    """

    joint_positions: JointStates = None
    joint_velocities: JointStates = None
    desired_pressures: typing.Tuple[
        JointPressures, JointPressures, JointPressures, JointPressures
    ] = None
    observed_pressures: typing.Tuple[
        JointPressures, JointPressures, JointPressures, JointPressures
    ] = None
    iteration: int = None
    time_stamp: int = None


@dataclass
class States:
    """ Snapshot states of a full HYSR setup,
    i.e. state of the pressure robot, of the 
    main simulation and of the extra balls.

    Attributes
    ----------
    pressure_robot:
      state of the pressure robot
    main_sim:
      state of the main simulation
    extra_balls:
      states of the ExtraBallsSet(s)
    """

    pressure_robot: PressureRobotState
    main_sim: MainSimState
    extra_balls: typing.Sequence[ExtraBallsState]
