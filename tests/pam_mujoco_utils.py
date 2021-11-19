import time
import typing
import subprocess


def start_pam_mujocos(mujoco_ids: typing.Sequence[str]):
    """
    calls 'pam_mujoco_no_xterms mujocod_id1 mujoco_id2 ...'
    """
    command = ["pam_mujoco_no_xterms"] + mujoco_ids
    process = subprocess.Popen(command, stdout=None, stderr=None)
    time.sleep(0.5)
    return process


def stop_pam_mujocos():
    """
    calls 'pam_mujoco_stop_all'
    """
    command = "pam_mujoco_stop_all"
    subprocess.Popen(command, stdout=None, stderr=None)
    time.sleep(0.5)
