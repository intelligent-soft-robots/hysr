import pathlib, typing
from dataclasses import dataclass
import o80, o80_pam, pam_mujoco
from .types import RobotPressures, JointPressures, JointStates, PressureRobotState


class PressureRobot:

    """ Superclass for all pressure controlled robot, subclasses being
    the real pam robot and (normal time and accelerated time) pam_mujoco
    simulated pressure controlled robots.
    """

    def __init__(self, frontend: o80_pam.FrontEnd):
        self._frontend = frontend

    def get(self) -> PressureRobotState:
        """ Returns the current state of the robot
        """
        obs = self._frontend.latest()
        state = PressureRobotState()
        state.desired_pressures = tuple(obs.get_desired_pressures())
        state.observed_pressures = tuple(obs.get_observed_pressures())
        state.joint_positions = obs.get_positions()
        state.joint_velocities = obs.get_velocities()
        state.iteration = obs.get_iteration()
        state.time_stamp = obs.get_time_stamp()
        return state

    def set_desired_pressures(self, robot_pressures: RobotPressures) -> None:
        """ (o80) adds the desired pressures, but does not send them to 
        the robot.
        """
        for dof, (ago, antago) in enumerate(robot_pressures):
            self._frontend.add_command(dof, ago, antago, o80.Mode.OVERWRITE)


class PamMujocoPressureRobot:
    """ Superclass for controlling a pam_mujoco simulated pressure controlled
    robot. Subclasses will be either "normal" time simulated robot, or 
    accelerated time simulated robot."""

    def __init__(
        self,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
        accelerated_time: bool,
    ):

        if accelerated_time:
            burst_mode = True
        else:
            burst_mode = False

        robot = pam_mujoco.MujocoRobot(
            segment_id,
            control=pam_mujoco.MujocoRobot.PRESSURE_CONTROL,
            json_control_path=str(pam_config_file),
            json_ago_hill_path=str(pam_model_file),
            json_antago_hill_path=str(pam_model_file),
        )

        self._handle = pam_mujoco.MujocoHandle(
            mujoco_id,
            graphics=graphics,
            accelerated_time=accelerated_time,
            burst_mode=burst_mode,
            robot1=robot,
        )
        self._frontend: o80_pam.FrontEnd = self._handle.frontends[segment_id]


class RealTimePressureRobot(PressureRobot):

    """ Superclass for real time pressure robot, which 
    subclass can be mujoco simulated robot or the real
    pam robot.
    """

    def __init__(self, frontend: o80_pam.FrontEnd):
        super().__init__(frontend)

    def pulse(self) -> None:
        """ Has the frontend pulsing, i.e. sharing the 
        commands that have been set via the set_desired_pressures
        method (of the PressureRobot superclass), if any.
        """
        self._frontend.pulse()

    def burst(self, nb_burst: int) -> None:
        """ Raise a NotImplementedError, as o80 backend of the real time
        robots (real or simulated) are not running in bursting mode.
        """
        raise NotImplementedError(
            "Normal time simulated robots do not have a burst method, use "
            "pulse instead"
        )


class RealRobot(RealTimePressureRobot):

    """ For controlling the real robot 
    """

    def __init__(self, segment_id: str):
        try:
            frontend = o80_pam.FrontEnd(segment_id)
        except RuntimeError as e:
            raise Exception(
                str(
                    "RealRobot: failed to instantiate a frontend to  "
                    "segment_id: {}. Has the related backend been started ?"
                ).format(segment_id)
            )
        super().__init__(frontend)


class SimPressureRobot(PamMujocoPressureRobot, RealTimePressureRobot):

    """ For controlling a pam_mujoco simulated robot
    running in "normal" time (as opposed to accelerated time) """

    def __init__(
        self,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
    ):
        accelerated_time: bool = False
        PamMujocoPressureRobot.__init__(
            self,
            mujoco_id,
            segment_id,
            pam_config_file,
            pam_model_file,
            graphics,
            accelerated_time,
        )
        RealTimePressureRobot.__init__(self, self._frontend)


class SimAcceleratedPressureRobot(PamMujocoPressureRobot, PressureRobot):

    """ For controlling a pam_mujoco pressure controlled
    pam robot running in accelerated time.
    """

    def __init__(
        self,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
    ):
        accelerated_time = True
        PamMujocoPressureRobot.__init__(
            self,
            mujoco_id,
            segment_id,
            pam_config_file,
            pam_model_file,
            graphics,
            accelerated_time,
        )
        PressureRobot.__init__(self, self._frontend)

    def burst(self, nb_bursts: int) -> None:
        """ Has the frontend pulsing, i.e. sharing the 
        commands that have been set via the set_desired_pressures
        method (of the PressureRobot superclass), if any. Then
        the handle bursts, i.e. triggers the related pam_mujoco
        instance run (nb_bursts) iterations.
        """
        self._frontend.pulse()
        self._handle.burst(nb_bursts)

    def pulse() -> None:
        """ Raises a NotImplementedError, 
        as accelerated time simulated robot's o80 backend run in
        bursting mode.
        """
        raise NotImplementedError(
            "Accelerated time simulated robots do not have a pulse method, use "
            "burst instead"
        )
