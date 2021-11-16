from pathlib import Path


class PseudoRealRobot:

    def __init__(self,
                 pam_config_file: Path,
                 segment_id: str,
                 mujoco_id: str,
                 graphics: bool,
                 accelerated_time: bool):

        if accelerated_time:
            burst_mode = True
        else:
            burst_mode = False

        robot = pam_mujoco.MujocoRobot(
            segment_id,
            control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL,
            json_control_path=pam_config_file,
        )
        
        self._handle = pam_mujoco.MujocoHandle(
            mujoco_id,
            graphics=graphics,
            accelerated_time=accelerated_time,
            burst_mode=burst_mode,
            robot1=robot,
        )

        self._frontend = handle.frontends[segment_id]

        


