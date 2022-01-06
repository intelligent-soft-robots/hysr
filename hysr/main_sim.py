import typing
import o80, pam_mujoco, context
from .scene import Scene
from .ball_trajectories import TrajectoryGetter, RandomRecordedTrajectory
from . import types

# mujoco_id, the same value has to be used for
# the pam_mujoco instance of the instance of MainSim.
_mujoco_id_g: str = "hysr_main_sim"

# (o80) segment_id of the items managed by the
# MainSim.
_table_segment_id: str = "hysr_main_sim_table"
_robot_segment_id: str = "hysr_main_sim_robot"
_ball_segment_id: str = "hysr_main_sim_ball"
_hit_point_segment_id: str = "hysr_main_hit_point"
_goal_segment_id: str = "hysr_main_goal"


class MainSim:
    """
    Code for managing the (joint controlled) robot, the ball, the hit point
    and the goal of a pam simulation. A instance of MainSim will configure
    pam mujoco to run in accelerated time. 

    Arguments
    ---------
    graphics:
      If true, the pam_mujoco instance will be configured to display graphics.
    scene:
      For setting the position and orientation of the table and of the robot.
    trajectory_getter:
      Will be used to set trajectories to the ball.
    """

    def __init__(
        self,
        graphics: bool,
        scene: Scene,
        contact: pam_mujoco.ContactTypes = pam_mujoco.ContactTypes.racket1,
        trajectory_getter: TrajectoryGetter = RandomRecordedTrajectory(),
    ):

        table = pam_mujoco.MujocoTable(
            _table_segment_id,
            position=scene.table.position,
            orientation=scene.table.orientation,
        )
        robot = pam_mujoco.MujocoRobot(
            _robot_segment_id,
            position=scene.robot.position,
            orientation=scene.robot.orientation,
            control=pam_mujoco.MujocoRobot.JOINT_CONTROL,
        )
        ball = pam_mujoco.MujocoItem(
            _ball_segment_id,
            control=pam_mujoco.MujocoItem.COMMAND_ACTIVE_CONTROL,
            contact_type=pam_mujoco.ContactTypes.racket1,
        )
        hit_point = pam_mujoco.MujocoItem(
            _hit_point_segment_id, control=pam_mujoco.MujocoItem.CONSTANT_CONTROL
        )
        goal = pam_mujoco.MujocoItem(
            _goal_segment_id, control=pam_mujoco.MujocoItem.CONSTANT_CONTROL
        )

        self._handle = pam_mujoco.MujocoHandle(
            self.get_mujoco_id(),
            graphics=graphics,
            accelerated_time=True,
            burst_mode=True,
            table=table,
            robot1=robot,
            balls=(ball,),
            hit_points=(hit_point,),
            goals=(goal,),
        )

        self._frontend_robot = self._handle.frontends[_robot_segment_id]
        self._frontend_ball = self._handle.frontends[_ball_segment_id]
        self._trajectory_getter = trajectory_getter

    def set_trajectory_getter(self, trajectory_getter: TrajectoryGetter) -> None:
        """
        Overwrite the current instance of trajectory_getter
        (that will be used by the method load_trajectories).
        """
        self._trajectory_getter = trajectory_getter

    def burst(self, nb_iterations: int):
        """
        Requests the corresponding pam_mujoco instance to burst
        """
        self._handle.burst(nb_iterations)

    def load_trajectory(self) -> None:
        """
        Generate a trajectory using the trajectory_getter (cf constructor)
        and load this trajecty to the mujoco backend. Note that
        as the mujoco backend is running in bursting mode, the trajectory
        will not start playing until the burst method of the handle is
        called.
        """
        trajectory = self._trajectory_getter.get_one()
        rate = self._trajectory_getter.get_sample_rate()
        rate_ns = int(rate * 1e9)
        duration = o80.Duration_us.nanoseconds(rate_ns)

        # going to first trajectory point
        self._frontend_ball.add_command(
            trajectory[0].position, trajectory[0].velocity, o80.Mode.OVERWRITE
        )
        # loading full trajectory
        for item in trajectory[1:]:
            self._frontend_ball.add_command(
                item.position, item.velocity, duration, o80.Mode.QUEUE
            )
        self._frontend_ball.pulse()

    def reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the 
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        """
        self._handle.reset(_ball_segment_id)

    def get_contact(self) -> context.ContactInformation:
        """
        Return the contact information between the ball and the racket.
        The returned instance has the attributes:
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
        """

        return self._handle.get_contact(_ball_segment_id)

    def activate_contact(self, index: types.ListOrIndex = None) -> None:
        """
        Contact will no longer be ignored (if 'deactivate_contact'
        has been previously called)
        """
        self._handle.activate_contact(_ball_segment_id)

    def deactivate_contact(self) -> None:
        """
        Contacts between the ball and the racket
        will be ignored.
        """
        self._handle.deactivate_contact(_ball_segment_id)

    def reset_contact(self) -> None:
        """
        Reset the contact between the ball and the racket,
        i.e. past contacts will be 'forgotten' and the 
        o80 backend regains control of the ball.
        """
        self._handle.reset_contact(_ball_segment_id)

    def get_state(self) -> types.MainSimState:
        """
        Returns the current state of the simulation. """
        # ball observation
        ball_obs = self._frontend_ball.latest()
        ball = ball_obs.get_observed_states()
        ball_position = [None] * 3
        ball_velocity = [None] * 3
        for dim in range(3):
            ball_position[dim] = ball.get(2 * dim).get()
            ball_velocity[dim] = ball.get(2 * dim + 1).get()
        # robot observation
        robot_obs = self._frontend_robot.latest()
        cartesian = (
            robot_obs.get_cartesian_position(),
            robot_obs.get_cartesian_orientation(),
        )
        # returning
        return types.MainSimState(
            ball_position,
            ball_velocity,
            robot_obs.get_positions(),
            robot_obs.get_velocities(),
            cartesian,
            robot_obs.get_iteration(),
            robot_obs.get_time_stamp(),
        )

    def set_robot(
        self, positions: types.JointStates, velocities: types.JointStates
    ) -> None:
        """
        Set a command for the o80 backend of the robot. 
        """
        self._frontend_robot.add_command(positions, velocities, o80.Mode.OVERWRITE)
        self._frontend_robot.pulse()

    @staticmethod
    def get_mujoco_id() -> str:
        return _mujoco_id_g
