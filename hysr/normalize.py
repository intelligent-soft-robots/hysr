import copy
import math
import typing
from .hysr_types import (
    Point3D,
    Box,
    Orientation3D,
    CartesianPose,
    JointStates,
    JointPressures,
    RobotPressures,
    PressureRobotState,
    ExtraBallsState,
    MainSimState,
    States,
    StatesHistory,
)

NumberTuple = typing.TypeVar(
    "NumberTuple",
    Point3D,
    Orientation3D,
    JointStates,
    JointPressures,
    typing.Tuple[float, ...],
)


def _value_error_if_out_of_bounds(
    input_: NumberTuple,
    min_values: NumberTuple,
    max_values: NumberTuple,
) -> None:
    """
    Checks that the input_ is in the provided bounds,
    raises a ValueError if not.
    """
    under_min = [i < min_ for i, min_ in zip(input_, min_values)]
    above_max = [i > max_ for i, max_ in zip(input_, max_values)]
    if any(under_min):
        fault_index = under_min.index(True)
        raise ValueError(
            "can not normalize {} from [{},{}] to [0,1] "
            "because {}<{}".format(
                input_[fault_index],
                min_values[fault_index],
                max_values[fault_index],
                input_[fault_index],
                min_values[fault_index],
            )
        )
    if any(above_max):
        fault_index = above_max.index(True)
        raise ValueError(
            "can not normalize {} from [{},{}] to [0,1] "
            "because {}>{}".format(
                input_[fault_index],
                min_values[fault_index],
                max_values[fault_index],
                input_[fault_index],
                max_values[fault_index],
            )
        )


def normalize(
    input_: NumberTuple,
    min_values: NumberTuple,
    max_values: NumberTuple,
) -> typing.Tuple[float, ...]:
    """
    Cast all values in the input_ to [0,1],
    assuming the input_ values are all between
    the provided min and max values.
    Raise a ValueError if any input value in not
    in the expected min/max interval.
    """
    _value_error_if_out_of_bounds(input_, min_values, max_values)
    normalized = [
        float(p - min_) / float(max_ - min_)
        for p, min_, max_ in zip(input_, min_values, max_values)
    ]
    return typing.cast(NumberTuple, tuple(normalized))


def denormalize(
    input_: NumberTuple,
    min_values: NumberTuple,
    max_values: NumberTuple,
) -> NumberTuple:
    """
    Cast all values in the input_ to [min,max],
    assuming the input_ values are in [0,1].
    Raise a ValueError if any input value in not
    in the expected min/max interval.
    """
    _value_error_if_out_of_bounds(
        input_,
        typing.cast(NumberTuple, [0.0 for _ in range(len(input_))]),
        typing.cast(NumberTuple, [1.0 for _ in range(len(input_))]),
    )
    denormalized = [
        min_ + i * (max_ - min_)
        for i, min_, max_ in zip(input_, min_values, max_values)
    ]
    return typing.cast(NumberTuple, tuple(denormalized))


def normalize_point3D(
    point: Point3D,
    position_box: typing.Optional[Box] = None,
    max_velocity: typing.Optional[float] = None,
) -> Point3D:
    """
    Normalize the coordinates. If a 3d point, the Box argument
    should not be null, and the point should be in the box.
    If a 3d velocity vector, then the max velocity should not be None
    and should be positive.
    """
    if position_box is None and max_velocity is None:
        raise ValueError(
            "failed to normalize a 3d point: "
            "either a position box or a max velocity "
            "should be provided as argument."
        )
    if position_box is not None:
        return typing.cast(Point3D, normalize(point, position_box[0], position_box[1]))
    if max_velocity is not None:
        min_ = (0.0, 0.0, 0.0)
        max_ = (max_velocity, max_velocity, max_velocity)
        return typing.cast(Point3D, normalize(point, min_, max_))
    # will never happen. The last "if" is there
    # only to help mypy (which complains about max_velocity
    # being possibly None)
    return (0.0, 0.0, 0.0)


def normalize_orientation3D(orientation: Orientation3D) -> Orientation3D:
    """
    Normalize the orientation, i.e. cast
    its values from [-1,1] to [0,1]
    """
    min_ = typing.cast(Orientation3D, tuple([-1.0] * 9))
    max_ = typing.cast(Orientation3D, tuple([+1.0] * 9))
    return typing.cast(Orientation3D, normalize(orientation, min_, max_))


def normalize_cartesian_pose(
    cartesian_pose: CartesianPose, position_box: Box
) -> CartesianPose:
    """
    Normalize the cartesian pose, i.e. normalize its
    position (see normalize_point3D) and its orientation
    (see normalize_orientation3D)
    """
    return (
        typing.cast(Point3D, normalize_point3D(cartesian_pose[0], position_box)),
        typing.cast(Orientation3D, normalize_orientation3D(cartesian_pose[1])),
    )


def normalize_joint_states(
    joint_states: JointStates, max_angular_velocity: typing.Optional[float] = None
) -> JointStates:
    """
    If max_angular_velocity is None (i.e. joints position in radians),
    casts all values from [-2pi,2pi] to [0.,1.]. Else (i.e. joint velocity
    in radians per second), cast to [0,max_angular_velocity]
    """
    if max_angular_velocity is None:
        return typing.cast(
            JointStates,
            normalize(
                joint_states,
                typing.cast(JointStates, tuple([-2.0 * math.pi] * 4)),
                typing.cast(JointStates, tuple([+2.0 * math.pi] * 4)),
            ),
        )
    else:
        return typing.cast(
            JointStates,
            normalize(
                joint_states,
                typing.cast(JointStates, tuple([0.0] * 4)),
                typing.cast(JointStates, tuple([max_angular_velocity] * 4)),
            ),
        )


def normalize_joint_pressures(
    pressures: JointPressures,
    min_pressures: JointPressures,
    max_pressures: JointPressures,
) -> JointPressures:
    """
    Cast all pressures value to [0.,1.]
    """
    return typing.cast(
        JointPressures, normalize(pressures, min_pressures, max_pressures)
    )


def normalize_robot_pressures(
    robot_pressures: RobotPressures,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> RobotPressures:
    """
    Cast all values to [0.,1.]
    """
    return typing.cast(
        RobotPressures,
        tuple(
            [
                normalize_joint_pressures(j, min_, max_)
                for j, min_, max_ in zip(robot_pressures, min_pressures, max_pressures)
            ]
        ),
    )


def normalize_pressure_robot_state(
    state: PressureRobotState,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
    max_angular_velocity: float,
) -> PressureRobotState:
    """
    Cast all values to [0.,1.].
    Max velocity in radians per second.
    """
    return PressureRobotState(
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(
            state.joint_velocities, max_angular_velocity=max_angular_velocity
        ),
        normalize_robot_pressures(
            state.desired_pressures, min_pressures, max_pressures
        ),
        normalize_robot_pressures(
            state.observed_pressures, min_pressures, max_pressures
        ),
        state.iteration,
        state.time_stamp,
    )


def normalize_extra_balls_state(
    state: ExtraBallsState,
    position_box: Box,
    max_velocity: float,
    max_angular_velocity: float,
) -> ExtraBallsState:
    """
    Cast all values to [0.,1.]
    """
    return ExtraBallsState(
        [
            normalize_point3D(ball_position, position_box=position_box)
            for ball_position in state.ball_positions
        ],
        [
            normalize_point3D(ball_velocity, max_velocity=max_velocity)
            for ball_velocity in state.ball_velocities
        ],
        copy.deepcopy(state.contacts),
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(
            state.joint_velocities, max_angular_velocity=max_angular_velocity
        ),
        normalize_point3D(state.racket_cartesian, position_box=position_box),
        state.iteration,
        state.time_stamp,
    )


def normalize_main_sim_state(
    state: MainSimState,
    position_box: Box,
    max_velocity: float,
    max_angular_velocity: float,
) -> MainSimState:
    """
    Cast all values to [0.,1.]
    """
    return MainSimState(
        normalize_point3D(state.ball_position, position_box=position_box),
        normalize_point3D(state.ball_velocity, max_velocity=max_velocity),
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(
            state.joint_velocities, max_angular_velocity=max_angular_velocity
        ),
        normalize_cartesian_pose(state.racket_cartesian, position_box=position_box),
        state.contact,
        state.iteration,
        state.time_stamp,
    )


def normalize_states(
    state: States,
    position_box: Box,
    max_velocity: float,
    max_angular_velocity: float,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> States:
    """
    Cast all values to [0.,1.]
    """
    return States(
        normalize_pressure_robot_state(
            state.pressure_robot,
            min_pressures,
            max_pressures,
            max_angular_velocity,
        ),
        normalize_main_sim_state(
            state.main_sim, position_box, max_velocity, max_angular_velocity
        ),
        [
            normalize_extra_balls_state(
                eb, position_box, max_velocity, max_angular_velocity
            )
            for eb in state.extra_balls
        ],
    )


def normalize_states_history(
    states_history: StatesHistory,
    box: Box,
    max_velocity: float,
    max_angular_velocity: float,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> StatesHistory:
    """
    Cast all values to [0.,1.]
    """
    return [
        normalize_states(
            state, box, max_velocity, max_angular_velocity, min_pressures, max_pressures
        )
        for state in states_history
    ]
