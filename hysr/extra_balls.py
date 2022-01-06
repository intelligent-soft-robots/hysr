import typing

# imports from pam_mujoco colcon space
import o80, context, pam_mujoco

# imports from hysr
from .types import (
    ListOrIndex,
    AcceptedNbOfBalls,
    Point3D,
    ExtraBall,
    ExtraBallsState,
    JointStates,
)
from .scene import Scene
from .ball_trajectories import TrajectoryGetter, RandomRecordedTrajectory

# the underlying c++ controller do not accept any number of
# extra balls per pam_mujoco processes (because each
# has to be templated)
# see pam_mujoco/srcpy/wrappers.cpp
_nb_balls_accepted_values_g = (3, 10, 20, 50, 100)

# each extra balls set is related to a pam_mujoco instance
# by sharing the same mujoco_id, e.g. for set 1, "hysr_extra_balls_1"
_mujoco_id_prefix_g: str = "hysr_extra_balls_"


class ExtraBallsSet:

    """
    Class for managing extra balls as well as their mujoco simulation.
    A set of extra balls is a set of balls which are all simulated by the same mujoco simulation,
    which is expected to mirror the real (or pseudo real) robot.
    The mujoco simulation as configured by this class will run in accelerated time and in
    bursting mode.

    Arguments
    ---------
    setid: 
      id of the extra ball set (arbitrary, but must be different of all sets).
    nb_balls: 
      has to be 3, 10, 20, 50 or 100.
    graphics: 
      if the mujoco simulation should run graphics.
    scene: 
      position and orientation of the table and robot.
    contact:
      which contact between the enviromnent and the balls should be.
    monitored:
      optional, default: the racket of the robot.
    trajectory_getter: 
      will be used to set the trajectories the balls
      will be required to follow
    """

    def __init__(
        self,
        setid: int,
        nb_balls: AcceptedNbOfBalls,
        graphics: bool,
        scene: Scene,
        contact: pam_mujoco.ContactTypes = pam_mujoco.ContactTypes.racket1,
        trajectory_getter: TrajectoryGetter = RandomRecordedTrajectory(),
    ):

        # not any number of balls is accepted. Checking the
        # user entered an accepted number.
        if nb_balls not in _nb_balls_accepted_values_g:
            accepted_values_str = str("{}, " * len(_nb_balls_accepted_values_g)).format(
                *[str(av) for av in _nb_balls_accepted_values_g]
            )
            raise ValueError(
                str(
                    "ExtraBalls can instantiate only with "
                    "nb_balls having one of the values: {} "
                    "(tried to instanciate with {}"
                ).format(accepted_values_str, nb_balls)
            )

        self._size = nb_balls

        # the mujoco simulation this constructor will configure, i.e it is assumed
        # that in a terminal ```pam_mujoco {mujoco_id}``` was called
        self._mujoco_id = self.get_mujoco_id(setid)

        # will be used to setup the trajectories performed by the balls
        self._trajectory_getter = trajectory_getter

        # for creating o80 frontends pointings to the correct shared memory
        # (the corresponding o80 backends are hosted by the mujoco simulation)
        self._segment_id_table = str(setid) + "_table"
        self._segment_id_robot = str(setid) + "_robot"
        # used to send command to all balls
        self._segment_id_balls_set = str(setid) + "_balls"
        # used to monitor the contact status of each individual ball
        self._segment_id_balls = [
            "{}_{}_ball".format(setid, index) for index in range(nb_balls)
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
        balls = pam_mujoco.MujocoItems(self._segment_id_balls_set)
        for index, ball_segment_id in enumerate(self._segment_id_balls):
            ball = pam_mujoco.MujocoItem(
                ball_segment_id,
                control=pam_mujoco.MujocoItem.CONSTANT_CONTROL,
                contact_type=contact,
            )
            balls.add_ball(ball)

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

        # balls frontends
        self._balls_frontend = self._handle.frontends[self._segment_id_balls_set]

        # robot frontend
        self._robot_frontend = self._handle.frontends[self._segment_id_robot]

    def _get_segment_ids(self, index: ListOrIndex) -> typing.Sequence[str]:
        # convenience method returning all segment ids if index is None,
        # and a list of segment ids otherwise (of len 1 if index is an int)
        def _get_int_list(i: ListOrIndex, max_index) -> typing.Sequence[int]:
            if isinstance(i, int):
                r = [i]
            else:
                r = i
            if any([i_ >= max_index for i_ in i]):
                raise IndexError()
            return r

        if index is None:
            return self._segment_id_balls
        else:
            return [self._segment_id_balls[i] for i in _get_int_list(index, self._size)]

    def burst(self, nb_iterations: int):
        """
        Requests the corresponding pam_mujoco instance to burst
        """
        self._handle.burst(nb_iterations)

    def get_contacts(
        self, index: ListOrIndex = None
    ) -> typing.Sequence[context.ContactInformation]:
        """
        Returns the list contact informations between the balls and the contact
        object (see argument "contact" of the constructor), i.e. an object with
        attributes:
        - contact occured : if true, at least one contact has occured
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
        return list(map(self._handle.get_contact, self._get_segment_ids(index)))

    def reset_contacts(self, index: ListOrIndex = None) -> None:
        """
        Reset the contacts, i.e. get_contacts will return instances that
        indicates no contact occured. Also, restore o80 control of the ball.
        """
        list(map(self._handle.reset_contact, self._get_segment_ids(index)))

    def activate_contacts(self, index: ListOrIndex = None) -> None:
        """
        Contacts will no longer be ignored (if 'deactivate_contacts'
        has been previously called)
        """
        list(map(self._handle.activate_contact, self._get_segment_ids(index)))

    def deactivate_contacts(self, index: ListOrIndex = None) -> None:
        """
        Contacts will be ignored
        """
        list(map(self._handle.deactivate_contact, self._get_segment_ids(index)))

    def set_trajectory_getter(self, trajectory_getter: TrajectoryGetter) -> None:
        """
        Overwrite the current instance of trajectory_getter
        (that will be used by the method load_trajectories).
        """
        self._trajectory_getter = trajectory_getter

    def get_state(self) -> ExtraBallsState:
        """
        Returns the current state of this extra balls set
        """
        # balls observation
        balls_observation = self._balls_frontend.latest()
        racket_cartesian = balls_observation.get_extended_state().robot_position
        contacts = balls_observation.get_extended_state().contacts
        iteration = balls_observation.get_iteration()
        time_stamp = balls_observation.get_time_stamp()
        state = balls_observation.get_observed_states()
        balls = [state.get(index) for index in range(self._size)]
        ball_positions = []
        ball_velocities = []
        for b in balls:
            ball_positions.append(b.get_position())
            ball_velocities.append(b.get_velocity())
        # robot observation
        robot_observation = self._robot_frontend.latest()
        return ExtraBallsState(
            ball_positions,
            ball_velocities,
            contacts,
            robot_observation.get_positions(),
            robot_observation.get_velocities(),
            racket_cartesian,
            iteration,
            time_stamp,
        )

    def reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the 
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        """
        self._handle.reset()

    def load_trajectories(self) -> None:
        """
        Generate trajectories using the trajectory_getter (cf constructor)
        and load these trajectories to the mujoco backend. Note that
        as the mujoco backend is running in bursting mode, the trajectory
        will not start playing until the burst method of the handle is
        called.
        """
        trajectories = self._trajectory_getter.get(self._size)
        rate = self._trajectory_getter.get_sample_rate()
        rate_ns = int(rate * 1e9)
        duration = o80.Duration_us.nanoseconds(rate_ns)
        item3d = o80.Item3dState()
        # loading one trajectory per ball
        for index_ball, trajectory in enumerate(trajectories):
            # going to first trajectory point
            item3d.set_position(trajectory[0].position)
            item3d.set_velocity(trajectory[0].velocity)
            self._balls_frontend.add_command(index_ball, item3d, o80.Mode.OVERWRITE)
            # loading full trajectory
            for item in trajectory[1:]:
                item3d.set_position(item.position)
                item3d.set_velocity(item.velocity)
                self._balls_frontend.add_command(
                    index_ball, item3d, duration, o80.Mode.QUEUE
                )
        self._balls_frontend.pulse()

    def set_robot(self, positions: JointStates, velocities: JointStates) -> None:
        """
        Set a command for the o80 backend of the robot. Will not be shared with
        the backend until the burst method is called.
        """
        self._robot_frontend.add_command(positions, velocities, o80.Mode.OVERWRITE)
        self._robot_frontend.pulse()

    @staticmethod
    def get_mujoco_id(setid: int) -> str:
        """
        returns the mujoco id corresponding to a
        ball set id
        """
        return _mujoco_id_prefix_g + str(setid)
