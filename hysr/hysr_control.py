import pathlib
import toml
import collections
import inspect
import importlib
import typing
import functools
import o80
import o80_pam
from .pressure_robot import PressureRobot
from .main_sim import MainSim
from .extra_balls import ExtraBallsSet
from .parallel_bursts import Burster, ParallelBursts
from . import hysr_types


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

        bursters: typing.List[Burster]
        l1: typing.List[Burster]
        l2: typing.List[Burster]
        if self._accelerated_time:
            l1 = [pressure_robot, main_sim]
            l2 = list(extra_balls)
            bursters = l1 + l2
        else:
            l1 = [main_sim]
            l2 = list(extra_balls)
            bursters = l1 + l2
        self._parallel_bursts = ParallelBursts(bursters)

        self._mujoco_time_step = mujoco_time_step
        self._algorithm_time_step = algorithm_time_step

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
            extra_balls.reset_contacts()

    def get_states(self) -> hysr_types.States:
        """
        Returns the current states of all balls and robots.
        """
        pressure_robot = self._pressure_robot.get_state()
        main_sim = self._main_sim.get_state()
        extra_balls = [extra_ball.get_state() for extra_ball in self._extra_balls]
        return hysr_types.States(pressure_robot, main_sim, extra_balls)

    def set_mirroring_state(self) -> None:
        """
        Set the desired position of the robot of the main simulation and of the
        extra balls simulation to the observed position of the real (or pseudo-real)
        robot. This will have no effect until the "step" method is called.
        """
        robot_state: hysr_types.PressureRobotState = self._pressure_robot.get_state()
        self._main_sim.set_robot(
            robot_state.joint_positions, robot_state.joint_velocities
        )
        for extra_balls in self._extra_balls:
            extra_balls.set_robot(
                robot_state.joint_positions, robot_state.joint_velocities
            )

    def set_desired_pressures(
        self, desired_pressures: hysr_types.RobotPressures
    ) -> None:
        """
        Set the desired pressure of the real or pseudo real robot.
        """
        self._pressure_robot.set_desired_pressures(desired_pressures)
        self._pressure_robot.pulse()

    def step(self, desired_pressures: hysr_types.RobotPressures) -> hysr_types.States:
        """
        Perform, in this order:
        1- read the joint positions and velocities from the real/pseudo-real robot
        2- read the ball informations from the simulations (main and extra balls)
        3- apply the desired pressures to the real/pseudo-real robot
        4- set the joint positions and velocities of the real/pseudo-real robot
          (step 1) to the main and extra ball simulation
        5- Burst all simulations (only main sim and extra balls if the real/pseudo-real
           robot is running in real time, otherwise also the pseudo-real robot)
        6- returns the state (hysr.hysr_types.States) as read in step 1 and 2
        """
        # step 1 and 2
        states: hysr_types.States = self.get_states()
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
            positions = typing.cast(hysr_types.JointStates, tuple([p_[1] for p_ in p]))
            velocities = typing.cast(hysr_types.JointStates, [v_[1] for v_ in v])
            over = [p_[0] for p_ in p]
            client.set_robot(positions, velocities)
            return all(over)

        _one_step_p = functools.partial(_one_step, self._pressure_robot, precision)
        over: typing.List[bool] = [False]
        while not all(over):
            l1: typing.List[typing.Union[MainSim, ExtraBallsSet]] = [self._main_sim]
            l2: typing.List[typing.Union[MainSim, ExtraBallsSet]] = list(
                self._extra_balls
            )
            over = list(map(_one_step_p, l1 + l2))
            self._parallel_bursts.burst(bursts_per_step)

    def to_robot_position(
        self,
        target_posture: hysr_types.JointStates,
        controller_factory: o80_pam.position_control.PositionControllerFactory,
    ) -> None:
        """Uses a position controller to bring the pseudo-real robot (and the
        simulated robots which mirror it) to the specified target posture.
        Position control for pneumatic muscles actuated robots are tricky, so
        not guaranteed to give proper results.
        The frequency of the controller must be the same than the frequency
        at which the algorithm runs (to keep the robots aligned), a ValueError
        error is raised otherwise.
        """
        self.align_robots()
        current_posture = self._pressure_robot.get_state().joint_positions
        controller = controller_factory.get(current_posture, target_posture)
        controller_time_step = controller.get_time_step()
        if controller_time_step != self._algorithm_time_step:
            raise ValueError(
                "hysr_control.to_robot_position: "
                "the position controller must run at "
                "the same perio than the algorithm "
                "({} != {})".format(controller_time_step, self._algorithm_time_step)
            )
        if not self._accelerated_time:
            frequency_manager = o80.FrequencyManager(1.0 / self._algorithm_time_step)
        while controller.has_next():
            state = self._pressure_robot.get_state()
            posture = state.joint_positions
            velocity = state.joint_velocities
            pressures = controller.next(posture, velocity)
            self.step(pressures)
            if not self._accelerated_time:
                frequency_manager.wait()

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
        starting_posture: hysr_types.JointStates,
        position_controller_factory: o80_pam.position_control.PositionControllerFactory,
    ) -> None:
        """
        Move the robots to the starting posture (desired position for each
        joint, in radian) using a position controller
        """
        self.to_robot_position(starting_posture, position_controller_factory)
        self._frequency_controller.reset()


ExpectedSuperclass = typing.TypeVar("ExpectedSuperclass", bound=object)


def _get_class(class_path: str) -> typing.Type:
    """
    class_path: something like "package.subpackage.module.class_name".
    Imports package.subpackage.module and returns the class.
    """

    # class_path is only the name of the class, which is thus expected
    # to be in global scope
    if "." not in class_path:
        try:
            class_ = globals()[class_path]
        except KeyError:
            raise ValueError(
                f"class {class_path} could not be found in the global scope"
            )

    # importing the package the class belongs to
    parts = class_path.split(".")
    to_import = ".".join(parts[:-1])
    try:
        imported = importlib.import_module(to_import)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"failed to import {to_import} (needed to instantiate {class_path}): {e}"
        )

    # getting the class
    try:
        class_ = getattr(imported, parts[-1])
    except AttributeError:
        raise ValueError(
            f"class {parts[-1]} (provided path: {class_path}) could not be found"
        )

    return class_


def _instantiate(
    factory_class: hysr_types.FactoryClass,
    expected_superclass: typing.Type[ExpectedSuperclass],
) -> ExpectedSuperclass:
    """
    Uses the information of the factory class (i.e. class name and
    arguments) to import the required package and instantiate
    the class. Also checks the class is a subclass of expected
    super class (ValueError raised if not).
    """
    class_name, args, kwargs = factory_class

    # getting the class from its path
    # e.g. MainSim from 'hysr.MainSim'
    class_ = _get_class(class_name)

    # checking the class is indeed a class
    if not inspect.isclass(class_):
        raise ValueError(f"{class_name} is not a class (it is a {type(class_name)})")

    # checking the class is a subclass
    if not issubclass(class_, expected_superclass):
        raise ValueError(
            f"{class_name} provided as the subclass to use of "
            f"{expected_superclass}, but {class_name} is not a subclass of "
            f"{expected_superclass}"
        )

    # reading the arguments requested by class_
    parameters = inspect.signature(class_.__init__).parameters
    nb_all_args = len(parameters) - 1  # -1: self argument
    valid_kwargs = {
        k: v.default for k, v in parameters.items() if v.default != inspect._empty
    }
    nb_kwargs = len(valid_kwargs)
    nb_args = nb_all_args - nb_kwargs

    # checking we have the expected number of args
    if nb_args != len(args):
        raise ValueError(
            f"the class {class_name} request {nb_args} "
            f"arguments, but {len(args)} provided"
        )

    # checking we have the expected of kwargs
    if nb_kwargs != len(kwargs):
        raise ValueError(
            f"the class {class_name} request {nb_kwargs} "
            f"arguments, but {len(kwargs)} provided"
        )

    # checking all kwargs are accepted by class_
    for kwarg_name in kwargs.keys():
        valid = list(valid_kwargs.keys())
        if kwarg_name not in valid:
            raise ValueError(f"{kwarg_name} is not a known kwarg name for {class_name}")

    # all seems ok, instantiating
    try:
        instance = class_(*args, **kwargs)
    except Exception as e:
        raise Exception(f"failed to instantiate {class_name}: {e}")

    return instance


def hysr_control_factory(
    pressure_robot: hysr_types.FactoryClass,
    main_sim: hysr_types.FactoryClass,
    extra_balls: typing.Iterable[hysr_types.FactoryClass],
    mujoco_time_step: float,
    algorithm_time_step: float,
) -> HysrControl:
    """
    Returns an instance of HysrControl based on the provided
    arguments.
    """
    return HysrControl(
        _instantiate(pressure_robot, PressureRobot),
        _instantiate(main_sim, MainSim),
        [_instantiate(extra_ball, ExtraBallsSet) for extra_ball in extra_balls],
        mujoco_time_step,
        algorithm_time_step,
    )


def hysr_control_from_toml_content(toml_string: str) -> HysrControl:
    """
    Instantiate HysrControl based on toml configuration.
    toml_string should be toml formated content, with at least the keys:
    - 'pressure_robot', 'main_sim': should have at least the keys 'module' and
      the key 'class' (string content) and optionally the keys 'args' and 'kwargs'.
      The value for 'args' should be either a list or a string. If the latest, the
      string should evaluate to a list. The value for 'kwargs' should be a dict or a
      string. If the latest, it should evaluate to a dict. Instances will be created
      by importing the class from the module, and instantiating the class passing the
      provided args and kargs. The classes should be subclasses of hysr.PressureRobot
      and hysr.MainSim.
    - 'algorithm_time_step' and "mujoco_time_step": should be float (in seconds)
    Optional are the keys 'extra_balls' and 'imports':
    - 'extra_balls': If provided, the value for 'extra_balls'
      should be a dict, with each entry providing the configuration required to instantiate
      hysr.ExtraBalls (or a subclass of it). The key of each entry is arbitrary and will
      not be used.
    - 'imports': list of packages (as strings) to import before constructing the instances. May be useful
      if the args and kwargs are string to evaluate.
    """

    try:
        t = toml.loads(toml_string)
    except toml.TomlDecodeError as tde:
        raise ValueError(f"hysr control factory: invalid toml content: {tde}")

    def _read_factory_class(
        toml_content: typing.Dict[str, typing.Any], key: str
    ) -> hysr_types.FactoryClass:
        if key not in toml_content.keys():
            raise ValueError(
                f"hysr control factory: failed to find required key {key} in toml configuration"
            )
        d = toml_content[key]
        required_keys = ("class",)
        optional_keys = ("args", "kwargs")
        for rk in required_keys:
            if rk not in d.keys():
                raise ValueError(
                    f"hysr control factory: failed to find required key {key}/{rk} in toml configuration"
                )
        for k in d.keys():
            if k not in required_keys + optional_keys:
                raise ValueError(
                    f"hysr control factory: the toml configuration has an unexpected key: {key}/{k}"
                )
        class_ = str(d["class"])
        if "args" in d.keys():
            if isinstance(d["args"], str):
                try:
                    args = eval(d["args"])
                except NameError as ne:
                    raise ValueError(
                        f"hysr_control_factory: failed to evaluate args {d['args']}: {ne} "
                    )
            else:
                args = d["args"]
            if not isinstance(args, collections.abc.Sequence):
                raise ValueError(
                    f"hysr control factory: the toml configuration has an unexpected value for: {key}/args (should be a sequence)"
                )
        else:
            args = []
        if "kwargs" in d.keys():
            if isinstance(d["kwargs"], str):
                kwargs = eval(d["kwargs"])
            else:
                kwargs = d["kwargs"]
        else:
            kwargs = {}

        return (class_, args, kwargs)

    # importing requested imports, if any
    if "imports" in t.keys():
        imports = t["imports"]
        if not isinstance(imports, collections.abc.Iterable):
            raise ValueError(
                "hysr control factory: the toml configuration has an unexpected value for imports (should be iterable of string)"
            )
        for import_ in imports:
            try:
                imported = importlib.import_module(str(import_))
            except ModuleNotFoundError as e:
                raise ValueError(
                    f"hysr control factory: failed to import module {import_}: " f"{e}"
                )
            globals()[import_] = imported

    # reading from toml the values for algo time step
    # and mujoco time step
    for k in ("algorithm_time_step", "mujoco_time_step"):
        if k not in t.keys():
            raise ValueError(
                f"hysr control factory: the toml configuration does not provide the required key: '{k}'"
            )
    try:
        algorithm_time_step = float(t["algorithm_time_step"])
    except TypeError:
        raise ValueError(
            "hysr control factory: the toml configuration provide an invalid value for 'algorithm_time_step' (should be a float)"
        )
    try:
        mujoco_time_step = float(t["mujoco_time_step"])
    except TypeError:
        raise ValueError(
            "hysr control factory: the toml configuration provide an invalid value for 'mujoco_time_step' (should be a float)"
        )

    # reading from toml the configuration for the pressure
    # robot and the main simulation
    pressure_robot = _read_factory_class(t, "pressure_robot")
    main_sim = _read_factory_class(t, "main_sim")

    # reading from toml the configurations for the extra balls
    if "extra_balls" in t.keys():
        if not isinstance(t["extra_balls"], dict):
            raise ValueError(
                "hysr control factory: the toml configuration for 'extra_balls' should be a dictionary"
            )
        extra_balls = [
            _read_factory_class(t["extra_balls"], key)
            for key in t["extra_balls"].keys()
        ]
    else:
        extra_balls = []

    return hysr_control_factory(
        pressure_robot, main_sim, extra_balls, mujoco_time_step, algorithm_time_step
    )


def hysr_control_from_toml_file(file_path: pathlib.Path) -> HysrControl:
    """
    Calls hysr_control_from_toml_content. Content of 'file_path' should
    be toml formatted.
    """

    if not file_path.is_file():
        raise FileNotFoundError(
            f"hysr control factory: failed to find toml configuration file: {file_path}"
        )

    content = file_path.read_text()

    try:
        hysr_control = hysr_control_from_toml_content(content)
    except ValueError as ve:
        raise ValueError(
            f"failed to use configuration file {file_path} to instantiate HysrControl: {ve}"
        )

    return hysr_control
