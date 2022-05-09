import time
import typing
import subprocess
import shared_memory


def start_pam_mujocos(mujoco_ids: typing.Sequence[str]):
    """
    disconnect the shared memory to existing segments related
    to previous runs (if any) and then
    calls 'pam_mujoco_no_xterms mujocod_id1 mujoco_id2 ...'
    """
    shared_memory.delete_all_segments()
    command = ["pam_mujoco_no_xterms"] + mujoco_ids
    process = subprocess.Popen(command, stdout=None, stderr=None)
    time.sleep(0.5)
    return process


def stop_pam_mujocos():
    """
    to previous runs, and calls 'pam_mujoco_stop_all'
    """
    command = ["pam_mujoco_stop_all"]
    subprocess.Popen(command, stdout=None, stderr=None)
    time.sleep(0.5)
