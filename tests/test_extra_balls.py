import typing
import pytest
import o80
import o80_pam
import pam_mujoco
import hysr
from hysr import ExtraBallsSet, Scene, Defaults, ball_trajectories, ParallelBursts
from . import pam_mujoco_utils

HDF5_DEFAULT_GROUP_G: str = "originals"


def _load_line_trajectory(
    extra_balls_sets: typing.Sequence[ExtraBallsSet],
    start_position: typing.Tuple[float, float, float],
    end_position: typing.Tuple[float, float, float],
    duration: float,
    sampling_rate: float,
    mujoco_time_step: float = Defaults.mujoco_time_step,
) -> int:

    trajectory_getter = ball_trajectories.LineTrajectory(
        start_position, end_position, duration, sampling_rate
    )
    for eb in extra_balls_sets:
        eb.set_trajectory_getter(trajectory_getter)
        eb.load_trajectories()

    nb_bursts: int = int(duration / mujoco_time_step)

    return nb_bursts


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> typing.Sequence[ExtraBallsSet]:
    """
    request.param is a list of n integers corresponding to the 
    number of extra balls to be holded by each of n corresponding pam_mujoco processes.
    startup: starts n pam_mujoco processes and instanciate n instances
             of ExtraBallsSet, each with the number of balls. The extra balls will
             be set with no graphics, default scene and contact with table.
    cleanup: stops the pam mujoco processes
    """
    nb_balls = request.param
    setids = [index for index in range(len(nb_balls))]
    mujoco_ids = [ExtraBallsSet.get_mujoco_id(setid) for setid in setids]
    process = pam_mujoco_utils.start_pam_mujocos(mujoco_ids)
    graphics = False
    scene = Scene.get_defaults()
    contact = pam_mujoco.ContactTypes.table
    trajectory_getter = ball_trajectories.RandomRecordedTrajectory(HDF5_DEFAULT_GROUP_G)
    extra_balls = [
        ExtraBallsSet(
            setid, nb_balls[setid], graphics, scene, trajectory_getter, contact=contact
        )
        for setid in setids
    ]
    yield extra_balls
    del process
    pam_mujoco_utils.stop_pam_mujocos()


@pytest.mark.parametrize("run_pam_mujocos", [[3, 10]], indirect=True)
def test_line_trajectory(run_pam_mujocos):
    """
    check all extra balls can follow a line trajectory
    """

    start_position = (0.0, 0.0, 3.0)
    end_position = (1.0, 0.0, 3.0)
    duration = 5.0
    sampling_rate = 0.01

    extra_balls: typing.Sequence[ExtraBallsSet] = run_pam_mujocos
    nb_bursts: int = _load_line_trajectory(
        extra_balls, start_position, end_position, duration, sampling_rate
    )

    first_bursts = int(nb_bursts / 2.0)
    second_bursts = nb_bursts - first_bursts

    def _check_mid_positions(extra_balls: ExtraBallsSet):
        def _check_mid_position(position: typing.Tuple[float, float, float]):
            # the ball is moving along x axis, so no change expected
            # on the y and z axis
            precision = 0.001
            assert position[1] == pytest.approx(0.0, abs=precision)
            assert position[2] == pytest.approx(3.0, abs=precision)
            # between start and stop
            assert position[0] > 0.0
            assert position[0] < 1.0

        state = extra_balls.get_state()
        for position in state.ball_positions:
            _check_mid_position(position)

    def _check_end_positions(extra_balls: ExtraBallsSet):
        def _check_end_position(position: typing.Tuple[float, float, float]):
            precision = 0.002
            assert position[0] == pytest.approx(1.0, abs=precision)
            assert position[1] == pytest.approx(0.0, abs=precision)
            assert position[2] == pytest.approx(3.0, abs=precision)

        state = extra_balls.get_state()
        for position in state.ball_positions:
            _check_end_position(position)

    with ParallelBursts(extra_balls) as pb:
        pb.burst(first_bursts)
        for eb in extra_balls:
            _check_mid_positions(eb)
        pb.burst(second_bursts)
        for eb in extra_balls:
            _check_end_positions(eb)


@pytest.mark.parametrize("run_pam_mujocos", [[3, 10]], indirect=True)
def test_contacts(run_pam_mujocos):
    """
    check contacts are reported.
    Extra balls, except for the first one of each set, 
    are requested to take a line trajectory that goes through the table. 
    They should all report contact, except for the first one
    """

    # trajectory going through the table
    table_position = Scene.get_defaults().table.position
    start_position = [tp + 0.5 for tp in table_position]
    end_position = [tp - 0.5 for tp in table_position]
    duration = 1.0
    sampling_rate = 0.01

    # for controlling the balls
    extra_balls: typing.Sequence[ExtraBallsSet] = run_pam_mujocos

    # balls going to start position. Contacts are deactivated
    # in case this setup has the balls going through the tables.
    # contacts reactivated at the end, for the rest of testing
    for eb in extra_balls:
        eb.deactivate_contacts()
    init_position = extra_balls[0].get_state().ball_positions[0]
    nb_bursts: int = _load_line_trajectory(
        extra_balls, init_position, start_position, duration, sampling_rate
    )
    with ParallelBursts(extra_balls) as pb:
        pb.burst(nb_bursts)
    for eb in extra_balls:
        eb.activate_contacts()
        contacts: typing.Sequence[context.ContactInformation] = eb.get_contacts()
        # checking the test starts with no contact detected
        assert all([not c.contact_occured for c in contacts[1:]])
    
    # loading the test trajectory
    nb_bursts: int = _load_line_trajectory(
        extra_balls, start_position, end_position, duration, sampling_rate
    )

    # a bit of hacking: requesting only the first ball
    # not to take the trajectory
    for eb in extra_balls:
        index_ball = 0
        item3d = o80.Item3dState()
        item3d.set_position(start_position)
        item3d.set_velocity((0.0, 0.0, 0.0))
        eb._balls_frontend.add_command(index_ball, item3d,
                                       o80.Duration_us.seconds(int(duration)),
                                       o80.Mode.OVERWRITE)
        eb._balls_frontend.pulse()

    # running the trajectory
    with ParallelBursts(extra_balls) as pb:
        pb.burst(nb_bursts)

    # checking all reported contacts
    for eb in extra_balls:
        contacts: typing.Sequence[context.ContactInformation] = eb.get_contacts()
        assert not contacts[0].contact_occured
        assert all([c.contact_occured for c in contacts[1:]])

    # trajectory going to another point above the table
    start_position = (0.5, 0.5, 1.0)
    end_position = (0.0, 0.0, 1.0)
    duration = 1.0
    sampling_rate = 0.01

    # loading the trajectory
    nb_bursts: int = _load_line_trajectory(
        extra_balls, start_position, end_position, duration, sampling_rate
    )

    # running the trajectory
    with ParallelBursts(extra_balls) as pb:
        pb.burst(nb_bursts)

    # control should have been lost on all balls except the
    # first one. So, only the first one should be at the end position
    for eb in extra_balls:
        state: ExtraBallsState = eb.get_state()
        positions = state.ball_positions
        precision = 5e-3
        for dim in range(3):
            assert positions[0][dim] == pytest.approx(end_position[dim], abs=precision)
        for position in positions[1:]:
            for dim in range(3):
                assert not position[dim] == pytest.approx(
                    end_position[dim], abs=precision
                )

    # redoing, after deactivating the contacts
    for eb in extra_balls:
        eb.deactivate_contacts()

    # loading the trajectory
    nb_bursts: int = _load_line_trajectory(
        extra_balls, start_position, end_position, duration, sampling_rate
    )

    # running the trajectory
    with ParallelBursts(extra_balls) as pb:
        pb.burst(nb_bursts)

    # now, all balls should be at the end position
    for eb in extra_balls:
        state: ExtraBallsState = eb.get_state()
        positions = state.ball_positions
        precision = 5e-3
        for position in positions:
            for dim in range(3):
                assert position[dim] == pytest.approx(end_position[dim], abs=precision)


@pytest.mark.parametrize("run_pam_mujocos", [[10]], indirect=True)
def test_random_trajectories(run_pam_mujocos):
    """
    Test that loading of random trajectories results in a different
    trajectory for each ball
    """

    # RandomRecordedTrajectory is the default arg in the constructor
    # of ExtraBallsSet
    extra_balls: ExtraBallsSet = run_pam_mujocos[0]
    traj_sizes = extra_balls.load_trajectories()
    extra_balls.burst(min(traj_sizes))
    state: hysr.types.ExtraBallsSet = extra_balls.get_state()
    for index, p1 in enumerate(state.ball_positions):
        for p2 in state.ball_positions[index + 1 :]:
            assert not p1 == p2

    # sanity check: when the trajectories are expected to be the same
    trajectory_getter = hysr.IndexedRecordedTrajectory(0,hysr.Defaults.ball_trajectory_group)
    extra_balls.set_trajectory_getter(trajectory_getter)
    traj_sizes = extra_balls.load_trajectories()
    assert len(set(traj_sizes))==1
    extra_balls.burst(traj_sizes[0])
    state: hysr.types.ExtraBallsSet = extra_balls.get_state()
    for index, p1 in enumerate(state.ball_positions):
        for p2 in state.ball_positions[index + 1 :]:
            assert p1 == p2


@pytest.mark.parametrize("run_pam_mujocos", [[3, 10]], indirect=True)
def test_reset(run_pam_mujocos):
    """
    Test simulation reset. The balls and the robot are requested
    to move, and then reset if called. Checking the initial state
    is retrieved.
    """

    ###########################################
    # methods for comparing simulation states #
    ###########################################
    list_of_list_attrs = ("ball_positions", "ball_velocities")
    list_attrs = ("joint_positions", "racket_cartesian")

    def _assert_compare(
        a: hysr.types.ExtraBallsState,
        b: hysr.types.ExtraBallsState,
        precision: float,
        same: bool,
    ) -> None:
        for attr in list_of_list_attrs:
            for list_a, list_b in zip(getattr(a, attr), getattr(b, attr)):
                if same:
                    assert list_a == pytest.approx(list_b, abs=precision)
                else:
                    assert not list_a == pytest.approx(list_b, abs=precision)
        for attr in list_attrs:
            if same:
                assert getattr(a, attr) == pytest.approx(
                    getattr(b, attr), abs=precision
                )
            else:
                assert not getattr(a, attr) == pytest.approx(
                    getattr(b, attr), abs=precision
                )

    def _assert_same(
        a: hysr.types.ExtraBallsState, b: hysr.types.ExtraBallsState, precision: float
    ) -> None:
        _assert_compare(a, b, precision, True)

    def _assert_diff(
        a: hysr.types.ExtraBallsState, b: hysr.types.ExtraBallsState, precision: float
    ) -> None:
        _assert_compare(a, b, precision, False)

    ###########################################

    extra_balls: typing.Sequence[ExtraBallsSet] = run_pam_mujocos

    # initial state
    init_states = [eb.get_state() for eb in extra_balls]

    # some motions
    robot_position = [1.0] * 4
    robot_velocities = [0.0] * 4
    for eb in extra_balls:
        eb.set_robot(robot_position, robot_velocities)
        eb.load_trajectories()
    with ParallelBursts(extra_balls) as pb:
        pb.burst(3000)

    # new states
    post_states = [eb.get_state() for eb in extra_balls]

    # ini and post should be different
    for ini, post in zip(init_states, post_states):
        _assert_diff(ini, post, 0.05)

    # reset
    for eb in extra_balls:
        eb.reset()
    with ParallelBursts(extra_balls) as pb:
        pb.burst(1)

    # updated states
    reset_states = [eb.get_state() for eb in extra_balls]

    # reset and init states should be the same
    for init, reset in zip(init_states, reset_states):
        _assert_same(init, reset, 0.05)
