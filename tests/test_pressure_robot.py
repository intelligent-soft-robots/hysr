import time, typing, pytest
import o80, o80_pam, pam_mujoco
from hysr import (
    PressureRobotState,
    RealRobot,
    SimPressureRobot,
    SimAcceleratedPressureRobot,
)
from hysr import Defaults
from hysr import types
from hysr import scene
from . import pam_mujoco_utils

_mujoco_id_g = "test_pressure_robot_mujoco_id"
_segment_id_g = "test_pressure_robot_segment_id"


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> bool:
    """
    startup: starts a pam_mujoco process
    cleanup: stops the pam mujoco processes
    """
    process = pam_mujoco_utils.start_pam_mujocos([_mujoco_id_g])
    yield None
    pam_mujoco_utils.stop_pam_mujocos()


def _test_sim_pressure_robot(accelerated):

    graphics = False

    # instantiating the robot
    if accelerated:
        class_ = SimAcceleratedPressureRobot
    else:
        class_ = SimPressureRobot
    robot = class_(
        pam_mujoco.RobotType.PAMY2,
        _mujoco_id_g,
        _segment_id_g,
        Defaults.pam_config[pam_mujoco.RobotType.PAMY2]["sim"],
        Defaults.muscle_model,
        graphics,
        Defaults.mujoco_time_step,
    )

    # checking the backend is running
    time_stamp1 = robot.get_state().time_stamp
    iteration1 = robot.get_state().iteration
    time_wait = 0.2
    if accelerated:
        nb_iterations = int(time_wait / Defaults.mujoco_time_step)
        robot.burst(nb_iterations)
    else:
        time.sleep(time_wait)
    time_stamp2 = robot.get_state().time_stamp
    iteration2 = robot.get_state().iteration
    assert time_stamp2 > time_stamp1
    assert iteration2 > iteration1

    # checking one can set desired pressures
    set_pressures: types.RobotPressures = (
        (15001, 15004),
        (15002, 15003),
        (15003, 15002),
        (15004, 15001),
    )
    robot.set_desired_pressures(set_pressures)
    robot.pulse()
    if accelerated:
        robot.burst(1)
    time.sleep(0.1)
    get_pressures: types.RobotPressures = robot.get_state().desired_pressures
    assert set_pressures == get_pressures


def test_normal_time_sim_pressure_robot(run_pam_mujocos):
    """Test mujoco simulated control (not accelerated time)"""

    accelerated = False
    _test_sim_pressure_robot(accelerated)


def test_accelerated_time_sim_pressure_robot(run_pam_mujocos):
    """Test mujoco simulated control (accelerated time)"""

    accelerated = True
    _test_sim_pressure_robot(accelerated)
