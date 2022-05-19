import pathlib
import o80
import o80_pam
import pam_mujoco
from .hysr_types import RobotPressures, PressureRobotState


class PressureRobot:

    """Superclass for all pressure controlled robot, subclasses being
    the real pam robot and (normal time and accelerated time) pam_mujoco
    simulated pressure controlled robots.
    """

    def __init__(self, frontend: o80_pam.FrontEnd):
        self._frontend = frontend

    def get_time_step(self) -> float:
        """Returns the control iteration period of the robot,
        for simulated robots, should be the same as the
        mujoco period. For real, it should be the
        o80 control frequency
        """
        raise NotImplementedError()

    def is_accelerated_time(self) -> bool:
        """Returns True if the instance controls
        a robot running in (simulated) accelerated time.
        Value will be different based on the
        subclass of PressureRobot that is implementing
        the method.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        For the real robot: raise a NotImplementedError
        """
        raise NotImplementedError()

    def get_state(self) -> PressureRobotState:
        """Returns the current state of the robot"""
        obs = self._frontend.latest()
        state = PressureRobotState()
        state.desired_pressures = tuple(obs.get_desired_pressures())  # type: ignore
        state.observed_pressures = tuple(obs.get_observed_pressures())  # type: ignore
        state.joint_positions = obs.get_positions()
        state.joint_velocities = obs.get_velocities()
        state.iteration = obs.get_iteration()
        state.time_stamp = obs.get_time_stamp()
        return state

    def set_desired_pressures(self, robot_pressures: RobotPressures) -> None:
        """(o80) adds the desired pressures, but does not send them to
        the robot.
        """
        agos = [rp[0] for rp in robot_pressures]
        antagos = [rp[1] for rp in robot_pressures]
        self._frontend.add_command(agos, antagos, o80.Mode.OVERWRITE)

    def pulse(self) -> None:
        """Share the queue of commmands with the robot."""
        raise NotImplementedError()


class PamMujocoPressureRobot:
    """Superclass for controlling a pam_mujoco simulated pressure controlled
    robot. Subclasses will be either "normal" time simulated robot, or
    accelerated time simulated robot."""

    def __init__(
        self,
        robot_type: pam_mujoco.RobotType,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
        accelerated_time: bool,
        mujoco_time_step: float,
    ):

        self._time_step = mujoco_time_step

        if accelerated_time:
            burst_mode = True
        else:
            burst_mode = False

        robot = pam_mujoco.MujocoRobot(
            robot_type,
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

    def get_time_step(self) -> float:
        """
        Returns the time step (control period) of the simulated robot,
        in seconds
        """
        return self._time_step

    def reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        """
        self._handle.reset()


class RealTimePressureRobot(PressureRobot):

    """Superclass for real time pressure robot, which
    subclass can be mujoco simulated robot or the real
    pam robot.
    """

    def __init__(self, frontend: o80_pam.FrontEnd):
        super().__init__(frontend)

    def is_accelerated_time(self) -> bool:
        return False

    def pulse(self) -> None:
        """Has the frontend pulsing, i.e. sharing the
        commands that have been set via the set_desired_pressures
        method (of the PressureRobot superclass), if any.
        """
        self._frontend.pulse()

    def burst(self, nb_burst: int) -> None:
        """Raise a NotImplementedError, as o80 backend of the real time
        robots (real or simulated) are not running in bursting mode.
        """
        raise NotImplementedError(
            "Normal time simulated robots do not have a burst method, use "
            "pulse instead"
        )


class RealRobot(RealTimePressureRobot):

    """For controlling the real robot"""

    def __init__(self, segment_id: str):
        try:
            frontend = o80_pam.FrontEnd(segment_id)
        except RuntimeError:
            raise Exception(
                str(
                    "RealRobot: failed to instantiate a frontend to  "
                    "segment_id: {}. Has the related backend been started ?"
                ).format(segment_id)
            )
        super().__init__(frontend)

    def get_time_step(self) -> float:
        frequency: float = self._frontend.get_frequency()
        return 1.0 / frequency

    def is_accelerated_time(self) -> bool:
        return False


class SimPressureRobot(PamMujocoPressureRobot, RealTimePressureRobot):

    """For controlling a pam_mujoco simulated robot
    running in "normal" time (as opposed to accelerated time)"""

    def __init__(
        self,
        robot_type: pam_mujoco.RobotType,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
        mujoco_time_step: float,
    ):
        accelerated_time: bool = False
        PamMujocoPressureRobot.__init__(
            self,
            robot_type,
            mujoco_id,
            segment_id,
            pam_config_file,
            pam_model_file,
            graphics,
            accelerated_time,
            mujoco_time_step,
        )
        RealTimePressureRobot.__init__(self, self._frontend)

    def is_accelerated_time(self) -> bool:
        return False


class SimAcceleratedPressureRobot(PamMujocoPressureRobot, PressureRobot):

    """For controlling a pam_mujoco pressure controlled
    pam robot running in accelerated time.
    """

    def __init__(
        self,
        robot_type: pam_mujoco.RobotType,
        mujoco_id: str,
        segment_id: str,
        pam_config_file: pathlib.Path,
        pam_model_file: pathlib.Path,
        graphics: bool,
        mujoco_time_step: float,
    ):
        accelerated_time = True
        PamMujocoPressureRobot.__init__(
            self,
            robot_type,
            mujoco_id,
            segment_id,
            pam_config_file,
            pam_model_file,
            graphics,
            accelerated_time,
            mujoco_time_step,
        )
        PressureRobot.__init__(self, self._frontend)

    def is_accelerated_time(self) -> bool:
        return True

    def burst(self, nb_bursts: int) -> None:
        """Triggers the related pam_mujoco
        instance run (nb_bursts) iterations.
        """
        self._handle.burst(nb_bursts)

    def pulse(self) -> None:
        """Share the queue of commmands with the robot.
        (but does not trigger the execution of any simulation
        step, see the burst method)
        """
        self._frontend.pulse()
