import time
import typing
import pytest
import o80_pam
import pam_mujoco
from hysr import ExtraBallsSet, Scene, Defaults, ball_trajectories, ParallelBursts
from . import pam_mujoco_utils


@pytest.fixture
def run_pam_mujocos(request) -> typing.Sequence[ExtraBallsSet]:
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
    extra_balls = [
        ExtraBallsSet(setid, nb_balls[setid], graphics, scene, contact) for setid in setids
    ]
    yield extra_balls
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
    trajectory_getter = ball_trajectories.LineTrajectory(
        start_position, end_position, duration, sampling_rate
    )

    extra_balls = run_pam_mujocos
    for eb in extra_balls:
        eb.set_trajectory_getter(trajectory_getter)
        eb.load_trajectories()

    mujoco_period = Defaults.mujoco_period
    nb_bursts = int(duration / mujoco_period)
    first_bursts = int(nb_bursts / 2.0)
    second_bursts = nb_bursts - first_bursts

    def _check_mid_positions(extra_balls: ExtraBallsSet):
        data = extra_balls.get()
        balls = data[0]

        def _check_mid_position(position: typing.Tuple[float, float, float]):
            # the ball is moving along x axis, so no change expected
            # on the y and z axis
            assert position[1] == 0.0
            assert position[2] == 3.0
            # between start and stop
            assert position[0] > 0.0
            assert position[0] < 1.0

        for ball in balls:
            _check_mid_position(ball[0])

    def _check_end_positions(extra_balls: ExtraBallsSet):
        data = extra_balls.get()
        balls = data[0]

        def _check_end_position(position: typing.Tuple[float, float, float]):
            assert position[0] == 1.0
            assert position[1] == 0.0
            assert position[2] == 3.0

        for ball in balls:
            _check_end_position(ball[0])

    with ParallelBursts(extra_balls) as pb:
        pb.burst(first_bursts)
        for eb in extra_balls:
            _check_mid_positions(eb)
        pb.burst(second_bursts)
        for eb in extra_balls:
            _check_end_positions(eb)
