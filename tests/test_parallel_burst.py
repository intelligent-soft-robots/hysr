import time
import typing
import pytest
import o80_pam
import pam_mujoco
from hysr import ParallelBursts
from . import pam_mujoco_utils
        

def _configure_pam_mujocos(
    mujoco_ids: typing.Sequence[str],
) -> typing.Sequence[o80_pam.FrontEnd]:
    # create and return mujoco handles that configure pam_mujoco to
    # run without graphics, in accelerated time, in burst mode
    # and with a controllable ball
    def _configure(mujoco_id):
        segment_id = "ball"
        graphics = False
        accelerated_time = True
        burst_mode = True
        ball = pam_mujoco.MujocoItem(
            segment_id, control=pam_mujoco.MujocoItem.CONSTANT_CONTROL
        )
        handle = pam_mujoco.MujocoHandle(
            mujoco_id,
            balls=(ball,),
            graphics=graphics,
            accelerated_time=accelerated_time,
            burst_mode=burst_mode,
        )
        return handle

    return list(map(_configure, mujoco_ids))


@pytest.fixture
def run_pam_mujocos(request) -> typing.Generator[pam_mujoco.MujocoHandle,None,None]:
    """
    setup : starts pam_mujoco processes corresponding to
    the mujoco_ids passed as arguments, configure them
    to run accelerated time and bursting mode, and return
    the corresponding frontends.
    cleanup : stops the pam_mujoco_processes
    """
    mujoco_ids = request.param
    process = pam_mujoco_utils.start_pam_mujocos(mujoco_ids)
    handles = _configure_pam_mujocos(mujoco_ids)
    yield handles
    pam_mujoco_utils.stop_pam_mujocos(mujoco_ids)


@pytest.mark.parametrize("run_pam_mujocos", [['m1','m2','m3']], indirect=True)
def test_parallel_bursts(run_pam_mujocos):
    """
    check the all backends bursts in parallel when 
    using ParallelBursts
    """
    handles = run_pam_mujocos

    assert len(handles) == 3

    def _get_steps(handles):
        return [handle.get_mujoco_step() for handle in handles]

    def _assert_steps(handles, iteration):
        assert all([it == iteration for it in _get_steps(handles)])

    _assert_steps(handles, 1)
    with ParallelBursts(handles) as pb:
        pb.burst(1)
        _assert_steps(handles, 2)
        pb.burst(10)
        _assert_steps(handles, 12)
        pb.burst(20)
        _assert_steps(handles, 32)


@pytest.mark.parametrize("run_pam_mujocos", [['m4']], indirect=True)
def test_single_parallel_bursts(run_pam_mujocos):
    """
    check ParallelBursts also handles a single
    instance of pam_mujoco well
    """
    handles = run_pam_mujocos
    assert len(handles) == 1
    handle = handles[0]
    assert handle.get_mujoco_step() == 1
    with ParallelBursts(handles) as pb:
        pb.burst(1)
        assert handle.get_mujoco_step() == 2
        pb.burst(10)
        assert handle.get_mujoco_step() == 12
        pb.burst(20)
        assert handle.get_mujoco_step() == 32
