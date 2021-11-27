import typing

from .defaults import Defaults


class Pose:
    """
    Encapsulate a 3d position and an orientation in a format suitable
    to Mujoco xml model file, i.e. 3d float for position and
    string for orientation (e.g. '-1 0 0 0 -1 0')
    """

    def __init__(self, position: typing.Sequence[float], orientation: str):
        self.position = position
        self.orientation = orientation


class Scene:
    """
    Encapsulate all attributes required to fully describe
    an experiment scene (and related mujoco simulation).
    For the moment: position and orientation of the table
    and of the robot.
    """

    def __init__(self, robot: Pose, table: Pose):

        self.robot = robot
        self.table = table

    @classmethod
    def get_defaults(cls):
        """
        Returns an instance of Scene using default values
        for the position and the orientation of the robot
        and of the table.
        """
        return cls(
            Pose(Defaults.position_robot, Defaults.orientation_robot),
            Pose(Defaults.position_table, Defaults.orientation_table),
        )
