import typing
import pytest
import pam_mujoco
import hysr
from . import pam_mujoco_utils


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> typing.Generator[None, None, None]:
    """
    Spawns a pam_mujoco with mujoco_id suitable for
    an instance of MainSim.
    startup: starts a pam_mujoco process
    cleanup: stops the pam mujoco processes
    """
    mujoco_id = hysr.MainSim.get_mujoco_id()
    pam_mujoco_utils.start_pam_mujocos([mujoco_id])
    yield None
    pam_mujoco_utils.stop_pam_mujocos()


@pytest.fixture
def run_2_pam_mujocos(request, scope="function") -> typing.Generator[None, None, None]:
    """
    Spawns a pam_mujoco with mujoco_id suitable for
    an instance of MainSim.
    startup: starts a pam_mujoco process
    cleanup: stops the pam mujoco processes
    """
    main_sim_mujoco_id = hysr.MainSim.get_mujoco_id()
    extra_balls_mujoco_id = hysr.ExtraBallsSet.get_mujoco_id(1)
    pam_mujoco_utils.start_pam_mujocos([main_sim_mujoco_id, extra_balls_mujoco_id])
    yield None
    pam_mujoco_utils.stop_pam_mujocos()


def test_robot(run_pam_mujocos):
    """
    Test the robot is properly controlled
    """

    robot_type = pam_mujoco.RobotType.PAMY2
    graphics = False
    scene = hysr.Scene.get_defaults()
    trajectory_getter = hysr.Defaults.trajectory_getter
    main_sim = hysr.MainSim(robot_type, graphics, scene, trajectory_getter)

    # playing a trajectory going from position 0 to position 0.1
    # in 20 iterations
    nb_iter = 20
    delta = 0.005
    target_positions = (0.0, 0.0, 0.0, 0.0)
    mujoco_period = hysr.Defaults.mujoco_time_step
    target_velocities = tuple([delta / mujoco_period] * 4)
    final_positions = tuple([delta * nb_iter] * 4)
    for iter in range(nb_iter):
        main_sim.set_robot(target_positions, target_velocities)
        main_sim.burst(1)
        target_positions = tuple([tp + delta for tp in target_positions])

    # checking the robot is at the expected position
    main_sim.burst(1)
    state: hysr.hysr_types.MainSimState = main_sim.get_state()
    precision = 1e-3
    for p1, p2 in zip(final_positions, state.joint_positions):
        assert p1 == pytest.approx(p2, abs=precision)


def test_ball(run_pam_mujocos):
    """
    Test the ball is properly controlled
    """

    start_position = (0.0, 0.0, 3.0)
    end_position = (1.0, 0.0, 3.0)
    duration = 5.0
    sampling_rate = 0.01

    trajectory_getter = hysr.ball_trajectories.LineTrajectory(
        start_position, end_position, duration, sampling_rate
    )

    robot_type = pam_mujoco.RobotType.PAMY2
    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(
        robot_type, graphics, scene, trajectory_getter=trajectory_getter
    )

    main_sim.load_trajectory()

    nb_iterations = int(duration / hysr.Defaults.mujoco_time_step)

    main_sim.burst(nb_iterations)

    state: hysr.hysr_types.MainSimState = main_sim.get_state()
    precision = 5e-3
    for p1, p2 in zip(end_position, state.ball_position):
        assert p1 == pytest.approx(p2, abs=precision)


def test_contacts(run_pam_mujocos):
    """
    Tests all the contacts related method
    of MainSim
    """

    # instantiating MainSim
    graphics = False
    robot_type = pam_mujoco.RobotType.PAMY2
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(
        robot_type, graphics, scene, hysr.Defaults.trajectory_getter
    )

    # 3d position of the racket
    racket_position = main_sim.get_state().racket_cartesian[0]

    def _play_contact_trajectory(delta_z: float = 0):
        """
        Has the ball playing a trajectory
        going through the racket
        """
        delta_x = 0.5
        position_start = [p for p in racket_position]
        position_end = [p for p in racket_position]
        # a horizontal line going through the racket
        position_start[0] += delta_x
        position_end[0] -= delta_x
        # a horizontal line going above (or below) the racket
        position_start[2] += delta_z
        position_end[2] += delta_z
        duration = 5.0
        sampling_rate = 0.01

        trajectory_getter = hysr.ball_trajectories.LineTrajectory(
            position_start, position_end, duration, sampling_rate
        )
        main_sim.set_trajectory_getter(trajectory_getter)
        main_sim.load_trajectory()

        nb_iterations = int(duration / hysr.Defaults.mujoco_time_step)
        main_sim.burst(nb_iterations)

    # starting state: no contact.
    contact = main_sim.get_state().contact
    assert not contact.contact_occured

    # contact detected when playing a
    # trajectory going through the racket.
    _play_contact_trajectory()
    contact = main_sim.get_state().contact
    assert contact.contact_occured

    # because there has been a contact and
    # there has been no reset, control on the
    # ball should be lost, and contact should have
    # remained unchanged.
    time_stamp = contact.time_stamp
    _play_contact_trajectory()
    contact = main_sim.get_state().contact
    assert contact.time_stamp == time_stamp

    # replaying the trajectory after reset,
    # new contact should be detected.
    main_sim.reset_contact()
    _play_contact_trajectory()
    contact = main_sim.get_state().contact
    assert contact.contact_occured
    assert not contact.time_stamp == time_stamp

    # playing a trajectory going above the racket,
    # there should be no contact, and minimal distance
    # ball / racket should be properly computed.
    main_sim.reset_contact()
    delta_z = 0.5
    _play_contact_trajectory(delta_z=delta_z)
    contact = main_sim.get_state().contact
    assert not contact.contact_occured
    assert contact.minimal_distance == pytest.approx(delta_z, abs=1e-3)
