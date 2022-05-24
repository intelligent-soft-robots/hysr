import typing
import numpy as np

from .hysr_types import (
    Box,
    RobotPressures,
    PressureRobotState,
    ExtraBallsState,
    MainSimState,
    States,
    Observation,
)

from .normalize import (
    normalize_pressure_robot_state,
    normalize_extra_balls_state,
    normalize_main_sim_state,
)


def _pack(
    state: typing.Union[MainSimState, ExtraBallsState, PressureRobotState],
    attributes: typing.Sequence[str],
) -> Observation:
    values: typing.List[float] = []
    for attr in attributes:
        if attr == "contact":
            if getattr(getattr(state, attr), "contact_occured"):
                values.append(1.0)
            else:
                values.append(0.0)
        elif attr == "contacts":
            values.extend([1.0 if c else 0.0 for c in getattr(state, attr)])
        else:
            attr_value = getattr(state, attr)
            values.extend(list(attr_value))
    return np.array(values, dtype=np.float32)


def pack_main_sim_state(
    state: MainSimState,
    attributes: typing.Sequence[str],
    position_box: Box,
    max_velocity: float,
    max_angular_velocity: float,
) -> Observation:
    """
    Returns a normalized flat numpy float array containing all the values of the specified
    attributes (in order). For contact (if in the list of attributes): the instance of ContactInformation is cast to a float:
    1. if contact occured, else 0.
    The box is used for the normalization of the 3d points (i.e. ball_positions, ball_velocities and racket_cartesian).
    Attributes must be attributes of an instance of MainSimState.
    An AttributeError is raised if a non valid attribute is passed as argument.
    """
    normalized = normalize_main_sim_state(
        state, position_box, max_velocity, max_angular_velocity
    )
    return _pack(normalized, attributes)


def pack_extra_balls_state(
    state: ExtraBallsState,
    attributes: typing.Sequence[str],
    position_box: Box,
    max_velocity: float,
    max_angular_velocity: float,
) -> Observation:
    """
    Returns a normalized flat numpy float array containing all the values of the specified
    attributes (in order). For contact (if in the list of attributes): cast this list of booleans to a list
    of floats (0. if no contact else 1.)
    The box is used for the normalization of the 3d points (i.e. ball_positions, ball_velocities and racket_cartesian).
    Attributes must be attributes of an instance of ExtraBallsState.
    An AttributeError is raised if a non valid attribute is passed as argument.
    """
    normalized = normalize_extra_balls_state(
        state, position_box, max_velocity, max_angular_velocity
    )
    return _pack(normalized, attributes)


def pack_pressure_robot_state(
    state: PressureRobotState,
    attributes: typing.Sequence[str],
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
    max_angular_velocity: float,
) -> Observation:
    """
    Returns a normalized flat numpy float array containing all the values of the specified
    attributes (in order).
    The box, the min_pressures and the max_pressures are used for the normalization.
    Attributes must be attributes of an instance of PressureRobotState.
    An AttributeError is raised if a non valid attribute is passed as argument.
    """
    normalized = normalize_pressure_robot_state(
        state, min_pressures, max_pressures, max_angular_velocity
    )
    return _pack(normalized, attributes)


def pack(
    states: States,
    config: typing.Iterable[typing.Tuple[str, typing.Sequence[str]]],
    position_box: typing.Optional[Box] = None,
    max_velocity: typing.Optional[float] = None,
    max_angular_velocity: typing.Optional[float] = None,
    min_pressures: typing.Optional[RobotPressures] = None,
    max_pressures: typing.Optional[RobotPressures] = None,
) -> Observation:

    r: typing.List[Observation] = []

    for main_attr, sub_attrs in config:
        if main_attr == "pressure_robot":
            if min_pressures is None or max_pressures is None:
                raise ValueError(
                    "pack: min and max pressures should not"
                    "be none when packing an instance of PressureRobotState"
                )
            if max_angular_velocity is None:
                raise ValueError(
                    "pack: a maximal angular velocity (max_angular_velocity) "
                    "should not be None when packing an instance of "
                    "PressureRobotState"
                )
            r.append(
                pack_pressure_robot_state(
                    getattr(states, main_attr),
                    sub_attrs,
                    min_pressures,
                    max_pressures,
                    max_angular_velocity,
                )
            )
        elif main_attr == "main_sim":
            if position_box is None:
                raise ValueError(
                    "pack: box should not"
                    "be none when packing an instance of MainSimState"
                )
            if max_velocity is None:
                raise ValueError(
                    "pack: max_velocity should not"
                    "be none when packing an instance of ExtraBallsState"
                )
            if max_angular_velocity is None:
                raise ValueError(
                    "pack: max_angular_velocity should not"
                    "be none when packing an instance of ExtraBallsState"
                )
            r.append(
                pack_main_sim_state(
                    getattr(states, main_attr),
                    sub_attrs,
                    position_box,
                    max_velocity,
                    max_angular_velocity,
                )
            )
        elif main_attr == "extra_balls":
            if position_box is None:
                raise ValueError(
                    "pack: position_box should not"
                    "be none when packing an instance of ExtraBallsState"
                )
            if max_velocity is None:
                raise ValueError(
                    "pack: max_velocity should not"
                    "be none when packing an instance of ExtraBallsState"
                )
            if max_angular_velocity is None:
                raise ValueError(
                    "pack: max_angular_velocity should not"
                    "be none when packing an instance of ExtraBallsState"
                )
            extra_balls_sets: typing.Sequence[ExtraBallsState] = getattr(
                states, main_attr
            )
            for extra_balls_set in extra_balls_sets:
                r.append(
                    pack_extra_balls_state(
                        extra_balls_set,
                        sub_attrs,
                        position_box,
                        max_velocity,
                        max_angular_velocity,
                    )
                )
        else:
            raise AttributeError(
                "hysr_types.States does not have attribute: {}".format(main_attr)
            )

    return np.concatenate(tuple(*r), axis=0, dtype=np.float32)


class ObservationFactory:
    def __init__(
        self,
        config: typing.Iterable[typing.Tuple[str, typing.Sequence[str]]],
        position_box: Box,
        max_velocity: float,
        max_angular_velocity: float,
        min_pressures: typing.Optional[RobotPressures] = None,
        max_pressures: typing.Optional[RobotPressures] = None,
    ) -> None:

        self._config = config
        self._box = position_box
        self._max_velocity = max_velocity
        self._max_angular_velocity = max_angular_velocity
        self._min_pressures = min_pressures
        self._max_pressures = max_pressures

    def get(self, states: States) -> Observation:
        return pack(
            states,
            self._config,
            self._box,
            self._max_velocity,
            self._max_angular_velocity,
            self._min_pressures,
            self._max_pressures,
        )
