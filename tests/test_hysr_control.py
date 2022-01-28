import pytest
import math
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


def test_to_robot_pressures(run_pam_mujocos):

    # TODO: update so that it runs both for accelerated and non accelerated robots
    
    # instantiating the robot and simulations
    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(graphics, scene)
    setid = 1
    nb_balls = 3
    extra_balls_set = hysr.ExtraBallsSet(setid, nb_balls, graphics, scene)
    pressure_robot = hysr.SimAcceleratedPressureRobot(
        _pressure_robot_mujoco_id_g,
        _pressure_robot_segment_id_g,
        hysr.Defaults.pam_config["sim"],
        hysr.Defaults.muscle_model,
        graphics,
    )

    # setting up the time steps duration
    mujoco_time_step = hysr.defaults.mujoco_period

    # "grouping" all simulations into an instance
    # of hysr control
    hysr_control = hysr.HysrControl(
        pressure_robot,
        main_sim,
        (extra_balls,),
        mujoco_time_step
    )

    # moving all robots to a set of pressures
    pressure1 = math.pi/4.0
    pressure2 = -pressure1
    joint_pressures = (pressure1,pressure2)
    robot_pressures : types.RobotPressures = tuple([joint_pressures]*4)
    nb_mujoco_iterations = 10
    hysr_control.to_robot_pressures(robot_pressures,nb_mujoco_iterations)

    # states of all robots and simulations
    states : hysr.types.States = hysr_control.get_states()
    
    # is the pressure controlled robot to the correct pressures ?
    for joint_pressures in states.pressure_robot.observed_pressures:
        assert joint_pressures[0]==pytest.approx(pressure1,abs=5)
        assert joint_pressures[1]==pytest.approx(pressure2,abs=5)
    for joint_pressures in states.pressure_robot.desired_pressures:
        assert joint_pressures[0]==pressure1
        assert joint_pressures[1]==pressure2

    # are all the robot in the same positions ?
    for pref,p1,p2 in zip(
            states.pressure_robot.joint_positions,
            states.main_sim.joint_positions,
            states.extra_balls[0].joint_positions
    ):
        assert pref==pytest.approx(p1,abs=0.01)
        assert pref==pytest.approx(p2,abs=0.01)
    
