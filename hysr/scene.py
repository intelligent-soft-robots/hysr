import typing
import pam_mujoco
from .defaults import Defaults


class Pose:
    """
    Encapsulate a 3d position and an orientation in a format suitable
    to Mujoco xml model file, i.e. 3d float for position and a 3d float for
    orientation.
    """

    def __init__(self, position: typing.Sequence[float], orientation: str):
        self.position = position
        self.orientation = orientation


class Scene:
    """
    Encapsulate all attributes required to fully describe
    an experiment scene (and related mujoco simulation).
    For the moment: robot type (pamy1 or pamy2), position and orientation of the table
    and of the robot.
    """

    def __init__(
        self, robot_type: pam_mujoco.RobotType, robot_pose: Pose, table_pose: Pose
    ):
        self.robot_type = robot_type
        self.robot = robot_pose
        self.table = table_pose

    @classmethod
    def get_defaults(cls):
        """
        Returns an instance of Scene using default values
        for the position and the orientation of the robot
        and of the table.
        """
        return cls(
            pam_mujoco.RobotType.PAMY2,
            Pose(Defaults.position_robot, Defaults.orientation_robot),
            Pose(Defaults.position_table, Defaults.orientation_table),
        )
