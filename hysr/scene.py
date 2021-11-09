import typing


class Pos:
    """
    Encapsulate a 3d position and an orientation in a format suitable
    to Mujoco xml model file, i.e. 3d float for position and
    string for orientation (e.g.
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

    def __init__(self, robot: Pos, table: Pos):

        self.robot = robot
        self.table = table
