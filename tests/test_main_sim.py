import typing
import pytest
import o80
import o80_pam
import pam_mujoco
import hysr
from . import pam_mujoco_utils


@pytest.fixture
def run_pam_mujocos(request, scope="function") -> bool:
    """
    startup: starts a pam_mujoco process 
    cleanup: stops the pam mujoco processes
    """
    mujoco_id = hysr.MainSim.get_mujoco_id()
    process = pam_mujoco_utils.start_pam_mujocos([mujoco_id])
    yield None
    pam_mujoco_utils.stop_pam_mujocos()



def test_robot(run_pam_mujocos):
    """
    Test the robot is properly controlled
    """
    
    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(graphics,scene)

    state_ini : hysr.types.MainSimState = main_sim.get()

    # playing a trajectory going from position 0 to position 0.1
    # in 20 iterations
    nb_iter = 20
    delta = 0.005
    target_positions = (0.,0.,0.,0.)
    mujoco_period = hysr.Defaults.mujoco_period
    target_velocities = tuple([delta/mujoco_period]*4)
    final_positions = tuple([delta*nb_iter]*4)
    for iter in range(nb_iter):
        state_ini : hysr.types.MainSimState = main_sim.get()
        main_sim.set_robot(target_positions,target_velocities)
        main_sim.burst(1)
        target_positions = tuple([tp+delta for tp in target_positions])

    # checking the robot is at the expected position
    main_sim.burst(1)
    state : hysr.types.MainSimState = main_sim.get()
    precision = 1e-3
    for p1,p2 in zip(final_positions,state.joint_positions):
        assert p1 == pytest.approx(p2,abs=precision)


def test_ball(run_pam_mujocos):
    """
    Test the ball is properly controlled
    """

    start_position = (0.0, 0.0, 3.0)
    end_position = (1.0, 0.0, 3.0)
    duration = 5.0
    sampling_rate = 0.01

    trajectory_getter = hysr.ball_trajectories.LineTrajectory(
        start_position,
        end_position,
        duration,
        sampling_rate
    )
         
    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(
        graphics,
        scene,
        trajectory_getter=trajectory_getter
    )

    main_sim.load_trajectory()
    
    nb_iterations = int(duration / hysr.Defaults.mujoco_period)

    main_sim.burst(nb_iterations)

    state : hysr.types.MainSimState = main_sim.get()
    precision = 1e-3
    for p1,p2 in zip(end_position,state.ball_position):
        assert p1 == pytest.approx(p2,abs=precision)
    
    
def test_contacts(run_pam_mujocos):

    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(
        graphics,
        scene,
        trajectory_getter=trajectory_getter
    )

    state = main_sim.get()
    racket_position = state.racket_cartesian[0]

    delta = 0.5
    position_start = [p-delta for p in racket_position]
    position_end = [p+delta for p in racket_position]
    duration = 5.0
    sampling_rate = 0.01
    
    trajectory_getter = hysr.ball_trajectories.LineTrajectory(
        start_position,
        end_position,
        duration,
        sampling_rate
    )
         
    graphics = False
    scene = hysr.Scene.get_defaults()
    main_sim = hysr.MainSim(
        graphics,
        scene,
        trajectory_getter=trajectory_getter
    )

    main_sim.load_trajectory()
    
    nb_iterations = int(duration / hysr.Defaults.mujoco_period)
    main_sim.burst(nb_iterations)

    

    
