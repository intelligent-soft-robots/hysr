import typing
import pam_mujoco
from .scene import Scene


class ExtraBallsSet:

    """
    Class for managing extra balls as well as their mujoco simulation.
    A set of extra balls is a set of balls which are all simulated by the same mujoco simulation,
    which is expected to mirror the real (or pseudo real) robot.
    The mujoco simulation as configured by this class will run in accelerated time and in
    bursting mode.

    Args:
         setid: id of the extra ball set (arbitrary, but must be different of all sets)
         graphics: if the mujoco simulation should run graphics
         scene : position and orientation of the table and robot
         contact: which contact between the enviromnent and the balls should be
                  monitored (default: the racket of the robot)
    """

    def __init__(
        self,
        setid: int,
        nb_balls: int,
        graphics: bool,
        scene: Scene,
        contact: pam_mujoco.ContactTypes = pam_mujoco.ContactTypes.racket1,
    ):

        # the mujoco simulation this constructor will configure, i.e it is assumed
        # that in a terminal ```pam_mujoco {mujoco_id}``` was called
        self._mujoco_id = self.get_mujoco_id(setid)

        # for creating o80 frontends pointings to the correct shared memory
        # (the corresponding o80 backends are hosted by the mujoco simulation)
        self._segment_id_table = str(setid) + "_table"
        self._segment_id_robot = str(setid) + "_robot"
        self._segment_id_balls_set = str(setid) + "_balls"
        self._segment_id_balls = [
            "{}_{}_ball".format(setid, index) for index in nb_balls
        ]

        # configuring the table
        table = pam_mujoco.MujocoTable(
            self._segment_id_table,
            position=scene.table.position,
            orientation=scene.table.orientation,
        )

        # configuring the robot
        robot = pam_mujoco.MujocoRobot(
            self._segment_id_robot,
            position=scene.robot.position,
            orientation=scene.robot.orientation,
            control=pam_mujoco.MujocoRobot.JOINT_CONTROL,
        )

        # configuring the balls
        self._balls = pam_mujoco.MujocoItems(extra_balls_segment_id)
        for index, ball_segment_id in enumerate(self._segment_id_balls):
            ball = pam_mujoco.MujocoItem(
                ball_segment_id,
                control=pam_mujoco.MujocoItem.CONSTANT_CONTROL,
                contact_type=contact,
            )
            self._balls.add_ball(ball)

        # configuring the mujoco simulation and
        # getting an handle to the mujoco simulation
        self._handle = pam_mujoco.MujocoHandle(
            self._mujoco_id,
            graphics=graphics,
            accelerated_time=True,
            burst_mode=True,
            table=table,
            robot1=robot,
            combined=balls,
        )

    @staticmethod
    def get_mujoco_id(setid: int) -> str:
        """
        returns the mujoco id corresponding to a
        ball set id
        """
        return "extra_balls_" + str(setid)
