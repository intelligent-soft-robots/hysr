import typing
import functools
import o80
import o80_pam
from .scene import Scene
from .ball_trajectories import TrajectoryGetter
from .pressure_robot import PressureRobot
from .main_sim import MainSim
from .extra_balls import ExtraBallsSet
from .parallel_bursts import ParallelBursts
from . import types


class _FrequencyController:
    """
    Helper for running HysrControl at the correct frequency, i.e.
    respecting the algorithm time step, if the pressure robot does
    not run in accelerated mode (if the pressure robot runs in accelerated mode, 
    the wait method has not effect). Also computes the number of bursts the 
    simulations have to perform per algorithm step in order to be kept aligned
    with the pressure robot.

    Arguments
    ---------
    accelerated_time:
      Mode of the pressure robot.
    mujoco_time_step:
      In seconds
    algorithm_time_step:
      In seconds
    """

    def __init__(
        self,
        accelerated_time: bool,
        mujoco_time_step: float,
        algorithm_time_step: float,
    ):

        if algorithm_time_step % mujoco_time_step != 0:
            raise ValueError(
                "The algorithm time step must be a multiple "
                "of the mujoco time step. "
                "{} is not a multiple of {}.".format(
                    algorithm_time_step, mujoco_time_step
                )
            )

        if algorithm_time_step < mujoco_time_step:
            raise ValueError(
                "algorithm_time_step must be lower or equals "
                "to the mujoco_time_step ({}<{}) ".format(
                    algorithm_time_step, mujoco_time_step
                )
            )

        self._active: bool = not accelerated_time
        self._mujoco_steps_per_algo_step = round(algorithm_time_step / mujoco_time_step)

        if self._active:
            self._frequency = 1.0 / algorithm_time_step
            self._frequency_manager = o80.FrequencyManager(self._frequency)

    def get_nb_bursts(self) -> int:
        """
        Returns the number of bursts the simulations should execute
        per algorithm step.
        """
        return self._mujoco_steps_per_algo_step

    def wait(self) -> None:
        """
        Wait the time required to run at the algorithm frequency (no effect
        if the pressure robot run in accelerated mode)
        """
        if self._active:
            self._frequency_manager.wait()

    def reset(self) -> None:
        """
        Reset the wait method, i.e. the time stamp of the previous call to the 
        wait method is replaced by the current time.
        """
        if self._active:
            self._frequency_manager = o80.FrequencyManager(self._frequency)


class HysrControl:
    """
    Convenience wrapper over the pressure robot, the main simulation and the extra
    balls simulation, allowing to send higher level commands to the pressure robot
    will keeping the simulations aligned.

    In order to run the simulations in parallel, an instance of HysrControl spawns some
    threads. One should be careful to either use the instance as a context manager, or
    to call the stop method at the end of the usage.

    Arguments
    ---------
    pressure_robot:
      the interface to the pressure controlled robot, either real ("real robot") 
      or simulated ("pseudo-real robot"), and if simulated, either real or accelerated time
    main_sim:
      the interface to the simulation managing the virtual ball playing the pre-recorded
      ball trajectories, and of the position controlled robot that should mirror the 
      the real or pseudo-real robot.
    extra_balls:
      the interfaces to the simulation managing extra balls, also playing pre-recorded
      ball trajectories and which robots should also mirror the real or pseudo real robot
    mujoco_time_step:
      the time step of the mujoco simulation hosting the pressure robot (if simulated),
      the main simulation and the extra balls simulations. In seconds.
    algorithm_time_step:
      the time step of the learning algorithm used to set input pressures to the real or 
      pseudo real robot. The step method will advance the simulations and sleep the amount of time 
      required to keep the corresponding frequency. In seconds.
    """

    def __init__(
        self,
        pressure_robot: PressureRobot,
        main_sim: MainSim,
        extra_balls: typing.Sequence[ExtraBallsSet],
        mujoco_time_step: float,
        algorithm_time_step: float,
    ):
        self._pressure_robot = pressure_robot
        self._main_sim = main_sim
        self._extra_balls = extra_balls

        self._accelerated_time = self._pressure_robot.is_accelerated_time()

        if self._accelerated_time:
            self._parallel_bursts = ParallelBursts(
                [pressure_robot, main_sim] + list(extra_balls)
            )
        else:
            self._parallel_bursts = ParallelBursts([main_sim] + list(extra_balls))

        self._mujoco_time_step = mujoco_time_step
        self._algorith_time_step = algorithm_time_step

        self._frequency_controller = _FrequencyController(
            pressure_robot.is_accelerated_time(), mujoco_time_step, algorithm_time_step
        )

        self._pressure_robot_time_step = pressure_robot.get_time_step()

    def is_accelerated_time(self):
        """
        Returns True if the pressure robot is running in accelerated time,
        False otherwise.
        """
        return self._accelerated_time

    def stop(self):
        """
        Stops all the threads.
        """
        self._parallel_bursts.stop()

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

    def load_trajectories(self) -> None:
        """
        Has the main simulation and the extra balls simulations
        load ball trajectories. The nature of these trajectories
        will depend on the instance of TrajectoryGetter encapsulated
        by the instances of MainSim and ExtraBallsSet used.
        """
        self._main_sim.load_trajectory()
        for extra_ball in self._extra_balls:
            extra_ball.load_trajectories()

    def reset_contacts(self) -> None:
        """
        After contacts between the balls and the robot racket,
        the control of the balls is disabled (i.e. the balls stop
        playing their trajectory and mujoco physic engines apply to them).
        A call to this method restore the control of the ball (i.e. call to
        the load_trajectories method has an effect). Also, the contacts information
        between the balls and the racket is reset to False. 
        """
        self._main_sim.reset_contact()
        for extra_balls in self._extra_balls:
            extra_ball.reset_contacts()

    def get_states(self) -> types.States:
        """
        Returns the current states of all balls and robots.
        """
        pressure_robot = self._pressure_robot.get_state()
        main_sim = self._main_sim.get_state()
        extra_balls = [extra_ball.get_state() for extra_ball in self._extra_balls]
        return types.States(pressure_robot, main_sim, extra_balls)

    def set_mirroring_state(self) -> None:
        """
        Set the desired position of the robot of the main simulation and of the
        extra balls simulation to the observed position of the real (or pseudo-real)
        robot. This will have no effect until the "step" method is called.
        """
        robot_state: types.PressureRobotState = self._pressure_robot.get_state()
        self._main_sim.set_robot(
            robot_state.joint_positions, robot_state.joint_velocities
        )
        for extra_balls in self._extra_balls:
            extra_balls.set_robot(
                robot_state.joint_positions, robot_state.joint_velocities
            )

    def set_desired_pressures(self, desired_pressures: types.RobotPressures) -> None:
        """
        Set the desired pressure of the real or pseudo real robot. 
        """
        self._pressure_robot.set_desired_pressures(desired_pressures)
        self._pressure_robot.pulse()

    def step(self, desired_pressures: types.RobotPressures) -> types.States:
        """
        Perform, in this order:
        1- read the joint positions and velocities from the real/pseudo-real robot
        2- read the ball informations from the simulations (main and extra balls)
        3- apply the desired pressures to the real/pseudo-real robot
        4- set the joint positions and velocities of the real/pseudo-real robot
          (step 1) to the main and extra ball simulation
        5- Burst all simulations (only main sim and extra balls if the real/pseudo-real
           robot is running in real time, otherwise also the pseudo-real robot)
        6- returns the state (hysr.types.States) as read in step 1 and 2
        """
        # step 1 and 2
        states: types.States = self.get_states()
        # step 3
        self.set_desired_pressures(desired_pressures)
        # step 4
        self._main_sim.set_robot(
            states.pressure_robot.joint_positions,
            states.pressure_robot.joint_velocities,
        )
        for extra_balls in self._extra_balls:
            extra_balls.set_robot(
                states.pressure_robot.joint_positions,
                states.pressure_robot.joint_velocities,
            )
        # step 5
        self._parallel_bursts.burst(self._frequency_controller.get_nb_bursts())
        # step 6
        return states

    def enforce_algo_frequency(self):
        """
        Wait the time required such that two successive call to this 
        function enforce the desired algorithm time step
        """
        self._frequency_controller.wait()

    def reset_frequency(self) -> None:
        """
        Reset the time stamp used in the "step" method.
        """
        self._frequency_controller.reset()

    def align_robots(self, bursts_per_step: int = 10, precision: float = 0.005) -> None:
        """
        Aligns the position/velocity of the simulated robots
        with the real / pseudo-real robot. Using the "set_mirroring_state" method
        could destabilize the mujoco simulations if the difference of position between 
        the real and the simulated robots is too high. This method aligns the robots
        over several mujoco steps in order to avoid this issue.
        """

        def _one_joint_step(
            arg: typing.Tuple[float, float], step=precision
        ) -> typing.Tuple[bool, float]:
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

        def _one_step(
            pressure_robot: PressureRobot,
            precision: float,
            client: typing.Union[MainSim, ExtraBallsSet],
        ) -> bool:
            robot_state = pressure_robot.get_state()
            client_state = client.get_state()
            target_positions = robot_state.joint_positions
            target_velocities = robot_state.joint_velocities
            positions = client_state.joint_positions
            velocities = client_state.joint_velocities
            p = list(map(_one_joint_step, zip(target_positions, positions)))
            v = list(map(_one_joint_step, zip(target_velocities, velocities)))
            positions = [p_[1] for p_ in p]
            velocities = [v_[1] for v_ in v]
            over = [p_[0] for p_ in p]
            client.set_robot(positions, velocities)
            return all(over)

        _one_step_p = functools.partial(_one_step, self._pressure_robot, precision)
        over = [False]
        while not all(over):
            over = list(map(_one_step_p, [self._main_sim] + list(self._extra_balls)))
            self._parallel_bursts.burst(bursts_per_step)

    def instant_reset(self) -> None:
        """
        Do a full simulation reset, i.e. restore the state of the 
        first simulation step, where all items are set according
        to the mujoco xml configuration file.
        """
        self._pressure_robot.reset()
        self._main_sim.reset()
        for extra_ball in self._extra_balls:
            extra_ball.reset()
        self._frequency_controller.reset()

    def natural_reset(
        self,
        starting_posture: types.JointStates,
        position_controller_factory: o80_pam.position_control.PositionControllerFactory,
    ) -> None:
        """
        Move the robots to the starting posture (desired position for each
        joint, in radian) using a position controller
        """
        self.to_robot_position(starting_posture, position_controller_factory)
        self._frequency_controller.reset()
