import time
import typing
import pytest
import o80_pam
import pam_mujoco
from hysr import ExtraBalls, Pos, Scene, Default
from . import pam_mujoco_utils




@pytest.fixture
def run_pam_mujocos(request) -> typing.Sequence[ExtraBalls]:
    """
    request.param is a list of n integers corresponding to the 
    number of extra balls to be holded by each of n corresponding pam_mujoco processes.
    startup: starts n pam_mujoco processes and instanciate n instances
             of ExtraBalls, each with the number of balls. The extra balls will
             be set with no graphics, default scene and contact with table.
    cleanup: stops the pam mujoco processes
    """
    nb_balls = request.param
    setids = [index for index in range(nb_balls)]
    mujoco_ids = [ExtraBalls.get_mujoco_id(setid)
                  for setid in setids]
    process = pam_mujoco_utils.start_pam_mujocos(mujoco_ids)
    graphics = False
    scene = Scene.get_defaults()
    contacts = pam_mujoco.ContactTypes.table
    extra_balls = [ExtraBalls(setid,nb_balls[setid],graphics,scene,contact)
                   for setid in setids]
    yield extra_balls
    pam_mujoco_utils.stop_pam_mujocos()


    
@pytest.mark.parametrize("run_pam_mujocos",[[3,10]],indirect=True)
def test_line_trajectory(run_pam_mujocos):
    """
    check all extra balls can follow a line trajectory
    """
    pass
