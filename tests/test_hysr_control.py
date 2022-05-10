import time
import pytest
import math
import pam_interface
import pam_mujoco
import hysr
from . import pam_mujoco_utils


_pressure_robot_mujoco_id_g = "tests_pressure_robot_mid"
_pressure_robot_segment_id_g = "tests_pressure_robot_sid"


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> None:
    """
    Spawns a three pam_mujocos with mujoco_id suitable for
    an instance of MainSim, an instance of ExtraBallsSet
    and an instance of PamMujocoPressureRobot.
    startup: starts the pam_mujoco processes
    cleanup: stops the pam mujoco processes
    """
    main_sim_mujoco_id = hysr.MainSim.get_mujoco_id()
    extra_balls_mujoco_id = hysr.ExtraBallsSet.get_mujoco_id(1)
    process = pam_mujoco_utils.start_pam_mujocos(
        [_pressure_robot_mujoco_id_g, main_sim_mujoco_id, extra_balls_mujoco_id]
    )
    yield None
    pam_mujoco_utils.stop_pam_mujocos()


@pytest.fixture(scope="function", params=[True, False])
def hysr_control_instance(request) -> hysr.HysrControl:
    """
    Instantiate an instance of HysrControl and yields it.
    It is parametrized to yield both real time (hysr.SimPressureRobot)
    and accelerated time (hysr.SimAcceleratedPressureRobot) instances.
    """
    accelerated = request.param

    # yielding instances of different classes depending if
    # parametrized for real or accelerated time
    if accelerated:
        PressureRobotClass = hysr.SimAcceleratedPressureRobot
    else:
        PressureRobotClass = hysr.SimPressureRobot

    # instantiating the robot and simulations
    graphics = False
    scene = hysr.Scene.get_defaults()
    robot_type = pam_mujoco.RobotType.PAMY2
    trajectory_getter = hysr.Defaults.trajectory_getter
    main_sim = hysr.MainSim(robot_type, graphics, scene, trajectory_getter)
    setid = 1
    nb_balls = 3
    extra_balls_set = hysr.ExtraBallsSet(
        setid, nb_balls, graphics, scene, trajectory_getter
    )
    robot_config_path = hysr.Defaults.pam_config[robot_type]["sim"]
    pressure_robot = PressureRobotClass(
        robot_type,
        _pressure_robot_mujoco_id_g,
        _pressure_robot_segment_id_g,
        robot_config_path,
        hysr.Defaults.muscle_model,
        graphics,
        hysr.Defaults.mujoco_time_step,
    )
    mujoco_time_step = hysr.Defaults.mujoco_time_step
    algo_time_step = hysr.Defaults.algorithm_time_step

    # instantiating and yielding the instance
    hysr_control = hysr.HysrControl(
        pressure_robot, main_sim, (extra_balls_set,), mujoco_time_step, algo_time_step
    )
    yield hysr_control

    # stopping the parallel bursting threads
    hysr_control.stop()


def test_align_robots(run_pam_mujocos, hysr_control_instance):
    """
    Test the align_robots method from HysrControl
    """

    hysr_control = hysr_control_instance

    # getting the underlying instances
    pressure_robot: hysr.PressureRobot = hysr_control._pressure_robot
    main_sim: hysr.MainSim = hysr_control._main_sim
    extra_balls: hysr.typing.Sequence[ExtraBallsSet] = hysr_control._extra_balls

    # moving all robots in different positions
    pos1 = [1.0, 1.5, -0.5, -1.0]
    vel1 = [0] * 4
    main_sim.set_robot(pos1, vel1)
    main_sim.burst(1000)
    pos2 = [0.0, 1.1, 0.5, -1.0]
    vel2 = [0] * 4
    for eb in extra_balls:
        eb.set_robot(pos2, vel2)
        eb.burst(1000)

    # ensuring all robots are in different positions
    states: hysr.types.States = hysr_control.get_states()
    pressure_pos = states.pressure_robot.joint_positions
    main_pos = states.main_sim.joint_positions
    extra_pos = states.extra_balls[0].joint_positions
    assert pressure_pos != pytest.approx(main_pos, abs=0.01)
    assert pressure_pos != pytest.approx(extra_pos, abs=0.01)
    assert main_pos != pytest.approx(extra_pos, abs=0.01)

    # aligning the robots
    hysr_control.align_robots()

    # ensuring all robot are now in the same position
    states: hysr.types.States = hysr_control.get_states()
    pressure_pos = states.pressure_robot.joint_positions
    main_pos = states.main_sim.joint_positions
    extra_pos = states.extra_balls[0].joint_positions
    assert pressure_pos == pytest.approx(main_pos, abs=0.005)
    assert pressure_pos == pytest.approx(extra_pos, abs=0.005)
    assert main_pos == pytest.approx(extra_pos, abs=0.005)


def test_stepping(run_pam_mujocos, hysr_control_instance):
    """
    Test the step method of HysrControl
    """
    hysr_control = hysr_control_instance

    # ensuring the robots are properly aligned before starting
    hysr_control.align_robots()

    # setting desired pressures and letting iterate 20 times
    target_pressures: hysr.types.RobotPressures = (
        (21000, 19000),
        (20000, 18000),
        (18000, 18000),
        (22000, 20000),
    )
    nb_steps = 20
    previous_states = None
    for step in range(nb_steps):
        states = hysr_control.step(target_pressures)
        hysr_control.enforce_algo_frequency()
        # checking the robots remain aligned
        if previous_states is not None:
            assert previous_states.pressure_robot.joint_positions == pytest.approx(
                states.main_sim.joint_positions, abs=0.01
            )
            for eb in states.extra_balls:
                assert previous_states.pressure_robot.joint_positions == pytest.approx(
                    eb.joint_positions, abs=0.01
                )
        previous_states = states
        hysr_control.enforce_algo_frequency()

    # some sanity checks
    states = hysr_control.get_states()
    assert states.pressure_robot.desired_pressures == target_pressures
    if hysr_control.is_accelerated_time():
        assert states.pressure_robot.iteration == states.main_sim.iteration
    for eb in states.extra_balls:
        assert states.main_sim.iteration == eb.iteration


def test_ball_trajectories(run_pam_mujocos, hysr_control_instance):
    """
    Checking loading trajectories do not raise exceptions
    """
    hysr_control = hysr_control_instance
    hysr_control.load_trajectories()


def test_instant_reset(run_pam_mujocos, hysr_control_instance):
    """
    Test the instanct_reset method of HysrControl
    """

    hysr_control = hysr_control_instance

    # initial states
    initial_states = hysr_control.get_states()

    # ensuring the robots are properly aligned before starting
    hysr_control.align_robots()

    # going to some random positions
    target_pressures: hysr.types.RobotPressures = (
        (21000, 19000),
        (20000, 18000),
        (18000, 18000),
        (22000, 20000),
    )
    nb_steps = 20
    for step in range(nb_steps):
        hysr_control.step(target_pressures)
        hysr_control.enforce_algo_frequency()

    # new states, expected different
    after_states = hysr_control.get_states()
    assert after_states.pressure_robot.joint_positions != pytest.approx(
        initial_states.pressure_robot.joint_positions, 0.01
    )
    assert after_states.pressure_robot.joint_positions != pytest.approx(
        initial_states.pressure_robot.joint_positions, 0.01
    )
    for eb1, eb2 in zip(after_states.extra_balls, initial_states.extra_balls):
        assert eb1.joint_positions != pytest.approx(eb2.joint_positions, 0.01)

    # resetting all
    hysr_control.instant_reset()

    # reset states, expected same as initial_states
    reset_states = hysr_control.get_states()
    assert reset_states.pressure_robot.joint_positions != pytest.approx(
        initial_states.pressure_robot.joint_positions, 0.01
    )
    assert reset_states.pressure_robot.joint_positions != pytest.approx(
        initial_states.pressure_robot.joint_positions, 0.01
    )
    for eb1, eb2 in zip(reset_states.extra_balls, initial_states.extra_balls):
        assert eb1.joint_positions != pytest.approx(eb2.joint_positions, 0.01)
