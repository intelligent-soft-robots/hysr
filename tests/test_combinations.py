import typing
import pytest
import pam_mujoco
import hysr
from . import pam_mujoco_utils

_pressure_robot_mujoco_id_g = "tests_pressure_robot_mid"
_pressure_robot_segment_id_g = "tests_pressure_robot_sid"


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> typing.Generator[None, None, None]:
    """
    Spawns a three pam_mujocos with mujoco_id suitable for
    an instance of MainSim, an instance of ExtraBallsSet
    and an instance of PamMujocoPressureRobot.
    startup: starts the pam_mujoco processes
    cleanup: stops the pam mujoco processes
    """
    main_sim_mujoco_id = hysr.MainSim.get_mujoco_id()
    extra_balls_mujoco_id = hysr.ExtraBallsSet.get_mujoco_id(1)
    pam_mujoco_utils.start_pam_mujocos(
        [_pressure_robot_mujoco_id_g, main_sim_mujoco_id, extra_balls_mujoco_id]
    )
    yield None
    pam_mujoco_utils.stop_pam_mujocos()


def test_mirroring(run_pam_mujocos):
    """
    Testing that instances of MainSim and ExtraBallsSet can
    successfully mirror the pseudo real robot. Also check
    that the balls play the same trajectories in the main sim and in
    the extra balls.
    """

    # instantiating the robot and simulations
    robot_type = pam_mujoco.RobotType.PAMY2
    graphics = False
    scene = hysr.Scene.get_defaults()
    trajectory_index = 1
    trajectory_getter = hysr.ball_trajectories.IndexedRecordedTrajectory(
        trajectory_index, hysr.Defaults.ball_trajectory_group
    )
    main_sim = hysr.MainSim(robot_type, graphics, scene, trajectory_getter)
    setid = 1
    nb_balls = 3
    extra_balls_set = hysr.ExtraBallsSet(
        setid, nb_balls, graphics, scene, trajectory_getter
    )
    pressure_robot = hysr.SimAcceleratedPressureRobot(
        robot_type,
        _pressure_robot_mujoco_id_g,
        _pressure_robot_segment_id_g,
        hysr.Defaults.pam_config[robot_type]["sim"],
        hysr.Defaults.muscle_model,
        graphics,
        hysr.Defaults.mujoco_time_step,
    )

    # for bursting main_sim and extra_balls_set together
    with hysr.ParallelBursts((main_sim, extra_balls_set)) as parallel_bursts:

        main_traj_size = main_sim.load_trajectory()
        extra_traj_sizes = extra_balls_set.load_trajectories()

        # all trajectories are expected to be the same
        extra_traj_sizes = set(extra_traj_sizes)
        assert len(extra_traj_sizes) == 1
        extra_traj_size = extra_traj_sizes.pop()
        assert extra_traj_size == main_traj_size

        # setting desired pressures to the pseudo real robot
        target_pressure = 20000
        ago_antago_target_pressures = tuple([target_pressure] * 2)
        target_pressures = tuple([ago_antago_target_pressures] * 4)
        pressure_robot.set_desired_pressures(target_pressures)

        # running a few iterations, with the robots of the main sim and
        # of the extra balls sim mirroring the pseudo real robot
        nb_iterations = main_traj_size

        for iter in range(nb_iterations):
            pressure_robot.burst(1)
            robot_state = pressure_robot.get_state()
            main_sim.set_robot(
                robot_state.joint_positions, robot_state.joint_velocities
            )
            extra_balls_set.set_robot(
                robot_state.joint_positions, robot_state.joint_velocities
            )
            parallel_bursts.burst(1)

        # getting final states of all
        robot_state = pressure_robot.get_state()
        main_sim_state = main_sim.get_state()
        extra_balls_state = extra_balls_set.get_state()

        # checking time stamp and iterations aligned
        assert robot_state.time_stamp == main_sim_state.time_stamp
        assert robot_state.iteration == main_sim_state.iteration
        assert robot_state.time_stamp == extra_balls_state.time_stamp
        assert robot_state.iteration == extra_balls_state.iteration

        position_precision = 5e-3

        # checking all robots are aligned
        assert robot_state.joint_positions == pytest.approx(
            main_sim_state.joint_positions, abs=position_precision
        )
        assert robot_state.joint_positions == pytest.approx(
            extra_balls_state.joint_positions, abs=position_precision
        )

        # checking same positions of the racket
        assert main_sim_state.racket_cartesian[0] == pytest.approx(
            extra_balls_state.racket_cartesian, abs=position_precision
        )

        # checking all positions of all the balls
        for extra_position in extra_balls_state.ball_positions:
            assert main_sim_state.ball_position == pytest.approx(
                extra_position, abs=position_precision
            )
