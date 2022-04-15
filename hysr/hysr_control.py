import typing
import o80_pam
from .scene import Scene
from .ball_trajectories import TrajectoryGetter
from .pressure_robot import PressureRobot
from .main_sim import MainSim
from .extra_balls import ExtraBallsSet
from .parallel_bursts import ParallelBursts
from . import types


class HysrControl:
    def __init__(
            self,
            pressure_robot: PressureRobot,
            main_sim: MainSim,
            extra_balls: typing.Sequence[ExtraBallsSet],
            mujoco_time_step : float
    ):
        self._pressure_robot = pressure_robot
        self._main_sim = main_sim
        self._extra_balls = extra_balls
        self._parallel_bursts = ParallelBursts([main_sim] + list(extra_balls))
        self._mujoco_time_step = mujoco_time_step
        self._pressure_robot_time_step = pressure_robot.get_time_step()

    def load_trajectories(self) -> None:
        self._main_sim.load_trajectory()
        for extra_ball in self._extra_balls:
            extra_ball.load_trajectories()

    def reset_contacts(self) -> None:
        self._main_sim.reset_contact()
        for extra_balls in self._extra_balls:
            extra_ball.reset_contacts()

    def get_states(self) -> types.States:
        pressure_robot = self._pressure_robot.get_state()
        main_sim = self._main_sim.get_state()
        extra_balls = [extra_ball.get_state() for extra_ball in self._extra_balls]
        return types.States(pressure_robot, main_sim, extra_balls)

    def _robot_mirror(self, nb_bursts: int = 1) -> types.PressureRobotState:
        robot_state: types.PressureRobotState = self._pressure_robot.get_state()
        self._main_sim.set_robot(
            robot_state.joint_positions, robot_state.joint_velocities
        )
        for extra_balls in self._extra_balls:
            extra_balls.set_robot(
                robot_state.joint_positions, robot_state.joint_velocities
            )
        self._parallel_bursts.burst(nb_bursts)
        return robot_state

    def align_robots(self, bursts_per_step: int = 10):
        def _one_step(arg: types.Tuple[float, float]) -> types.Tuple[bool, float]:
            target, current = arg
            diff = target - current
            if abs(diff) < step:
                current = target
                return True, current
            else:
                if diff > 0:
                    current += step
                else:
                    current -= step
                return False, current

        robot_state: types.PressureRobotState = self._pressure_robot.get_state()
        main_sim_state: types.MainSimState = self._main_sim.get_state()
        target_positions = robot_state.joint_positions
        target_velocities = robot_state.joint_velocities
        positions = main_sim_state.joint_positions
        velocities = main_sim_state.joint_velocities
        nb_dofs = len(target_positions)
        over = [False] * nb_dofs

        while not all(over):
            p = list(map(_one_step, zip(target_positions, positions)))
            v = list(map(_one_step, zip(target_velocities, velocities)))
            positions = [p_[1] for p_ in p]
            velocities = [v_[1] for v_ in v]
            over = [p_[0] for p_ in p]
            self._main_sim.set_robot(positions, velocities)
            for extra_ball in self._extra_balls:
                extra_ball.set_robot(positions, velocities)
            self._parallel_bursts.burst(bursts_per_step)

    def to_robot_pressures(
        self, pressures: types.RobotPressures, nb_mujoco_steps: int
    ) -> None:

        self._pressure_robot.set_desired_pressures(pressures)
        try:
            self._pressure_robot.pulse()
        except NotImplementedError:
            pass
        for step in range(nb_mujoco_steps):
            self._robot_mirror()
            try:
                self._pressure_robot.burst(1)
            except NotImplementedError:
                pass

    def to_robot_position(
        self,
        position: types.JointStates,
        position_controller_factory: o80_pam.PositionControllerFactory,
    ):
        def _divisable(label1: str, v1: float, label2: str, v2: float) -> int:
            if v1 % v2 == 0:
                return int(v1 / v2)
            error = str(
                "the reminder of the division of {} by {} " "should be 0, but {}%{}={}"
            ).format(v1, v2, v1 % v2)
            raise ValueError(error)

        controller_time_step = position_controller_factory.time_step

        if self._pressure_robot.is_accelerated_time():
            frequency_manager = None
            nb_robot_bursts = _divisable(
                "position controller time step",
                controller_time_step,
                "robot pressure controller step",
                self._pressure_robot_time_step,
            )
        else:
            frequency_manager = o80.FrequencyManager(1.0 / controller_time_step)
            nb_robot_bursts = None

        nb_sim_bursts = _divisable(
            "position controller time step",
            controller_time_step,
            "mujoco time step",
            self._mujoco_time_step,
        )

        for _ in range(2):

            robot_position: types.JointStates = self._pressure_robot.get_state().joint_positions
            controller: o80_pam.PositionController = position_controller_factory(
                robot_position, position
            )

            while controller.has_next():
                robot_state: types.PressureRobotState = self._robot_mirror(
                    nb_bursts=nb_sim_bursts
                )
                pressures = controller.next(
                    robot_state.joint_positions, robot_state.joint_velocities
                )
                self._pressure_robot.set(pressures)
                try:
                    self._pressure_robot.burst(nb_robot_bursts)
                except NotImplementedError:
                    frequency_manager.wait()

    def instant_reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the 
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        """
        self._pressure_robot.reset()
        self._main_sim.reset()
        self._main_sim.load_trajectory()
        for extra_ball in self._extra_balls:
            extra_ball.reset()
            extra_ball.load_trajectories()

    def natural_reset(
        self,
        starting_posture: types.JointStates,
        position_controller_factory: o80_pam.PositionControllerFactory,
    ) -> None:
        """
        Move the robots to the starting posture (desired position for each
        joint, in radian) using a position controller
        """
        self.to_robot_position(starting_posture, position_controller_factory)


