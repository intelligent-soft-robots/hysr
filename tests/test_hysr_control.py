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


@pytest.mark.parametrize("accelerated", [True,False])    
def test_to_robot_pressures(run_pam_mujocos,accelerated):

    # TODO: update so that it runs both for accelerated and non accelerated robots
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

    # setting up the time steps duration
    mujoco_time_step = hysr.Defaults.mujoco_time_step

    # "grouping" all simulations into an instance
    # of hysr control
    with hysr.HysrControl(
        pressure_robot, main_sim, (extra_balls_set,), mujoco_time_step
    ) as hysr_control:

        # moving all robots to a set of pressures
        config = pam_interface.JsonConfiguration(str(robot_config_path))
        min_pressures = [
            config.min_pressure(dof, pam_interface.sign.agonist) for dof in range(4)
        ]
        max_pressures = [
            config.max_pressure(dof, pam_interface.sign.antagonist) for dof in range(4)
        ]
        robot_pressures = [
            (minp, maxp) for minp, maxp in zip(min_pressures, max_pressures)
        ]
        nb_mujoco_iterations = 200
        hysr_control.to_robot_pressures(robot_pressures, nb_mujoco_iterations)
        
        # states of all robots and simulations
        states: hysr.types.States = hysr_control.get_states()

        # is the pressure controlled robot to the correct pressures ?
        for dof, joint_pressures in enumerate(states.pressure_robot.observed_pressures):
            assert joint_pressures[0] == pytest.approx(min_pressures[dof], abs=100)
            assert joint_pressures[1] == pytest.approx(max_pressures[dof], abs=100)
        for dof, joint_pressures in enumerate(states.pressure_robot.desired_pressures):
            assert joint_pressures[0] == min_pressures[dof]
            assert joint_pressures[1] == max_pressures[dof]

        # are all the robot in the same positions ?
        for pref, p1, p2 in zip(
            states.pressure_robot.joint_positions,
            states.main_sim.joint_positions,
            states.extra_balls[0].joint_positions,
        ):
            print("****",accelerated,pref,p1,p2)
            assert pref == pytest.approx(p1, abs=0.01)
            assert pref == pytest.approx(p2, abs=0.01)
