import typing
from collections.abc import Callable
from pathlib import Path
from .types import Pressures, Joints
from .robot_state import RobotState,JointsState



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

    def read(self) -> RobotState:
        """
        Returns the current state of the robot.
        """
        def _get_pressure(f:Callable)->typing.Tuple[Pressures,Pressures]:
            pressures = f()
            pressures_ago = tuple([pressures[dof][0] for dof in range(4)])
            pressures_antago = tuple([pressures[dof][1] for dof in range(4)])
            return (pressures_ago,pressures_antago)

        obs = self._frontend.latest()
        desired_pressures = _get_pressure(obs.get_desired_pressures)
        observed_pressures = _get_pressure(obs.get_observed_pressures)
        robot_joints = tuple(obs.get_positions())
        robot_joint_velocities = tuple(obs.get_velocities())

        return RobotState(
            desired_pressures, observed_pressures,
            JointsState(robot_joints, robot_joint_velocities)
        )



