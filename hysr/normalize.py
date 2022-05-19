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

FloatTuple = typing.Union[
    Point3D,
    Orientation3D,
    JointStates,
    typing.Tuple[float, float],
]

NumberTuple = typing.TypeVar(
    "NumberTuple",
    Point3D,
    Orientation3D,
    JointStates,
    JointPressures,
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
    should_round: bool = False,
) -> NumberTuple:
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
    if should_round:
        normalized = [round(n) for n in normalized]
    return typing.cast(NumberTuple, tuple(normalized))


def normalize_point3D(point: Point3D, box: Box) -> Point3D:
    """
    Normalize the coordinates of the points, assuming
    all points are inside the box.
    """
    return normalize(point, box[0], box[1])


def normalize_orientation3D(orientation: Orientation3D) -> Orientation3D:
    """
    Normalize the orientation, i.e. cast
    its values from [-1,1] to [0,1]
    """
    min_ = typing.cast(Orientation3D, tuple([-1.0] * 9))
    max_ = typing.cast(Orientation3D, tuple([+1.0] * 9))
    return normalize(orientation, min_, max_)


def normalize_cartesian_pose(
    cartesian_pose: CartesianPose, box3d: Box
) -> CartesianPose:
    """
    Normalize the cartesian pose, i.e. normalize its
    position (see normalize_point3D) and its orientation
    (see normalize_orientation3D)
    """
    return (
        normalize_point3D(cartesian_pose[0], box3d),
        normalize_orientation3D(cartesian_pose[1]),
    )


def normalize_joint_states(joint_states: JointStates) -> JointStates:
    """
    Casts all values from [-2pi,2pi] to [0,1]
    """
    return normalize(
        joint_states,
        typing.cast(JointStates, tuple([-2.0 * math.pi] * 4)),
        typing.cast(JointStates, tuple([+2.0 * math.pi] * 4)),
    )


def normalize_joint_pressures(
    pressures: JointPressures,
    min_pressures: JointPressures,
    max_pressures: JointPressures,
) -> JointPressures:
    """
    Cast all pressures value to [0,1]
    """
    return normalize(pressures, min_pressures, max_pressures, should_round=True)


def normalize_robot_pressures(
    robot_pressures: RobotPressures,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> RobotPressures:
    """
    Cast all values to [0,1]
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
) -> PressureRobotState:
    """
    Cast all values to [0,1]
    """
    return PressureRobotState(
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(state.joint_velocities),
        normalize_robot_pressures(
            state.desired_pressures, min_pressures, max_pressures
        ),
        normalize_robot_pressures(
            state.observed_pressures, min_pressures, max_pressures
        ),
        state.iteration,
        state.time_stamp,
    )


def normalize_extra_balls_state(state: ExtraBallsState, box: Box) -> ExtraBallsState:
    """
    Cast all values to [0,1]
    """
    return ExtraBallsState(
        [
            normalize_point3D(ball_position, box)
            for ball_position in state.ball_positions
        ],
        [
            normalize_point3D(ball_velocity, box)
            for ball_velocity in state.ball_velocities
        ],
        copy.deepcopy(state.contacts),
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(state.joint_velocities),
        normalize_point3D(state.racket_cartesian, box),
        state.iteration,
        state.time_stamp,
    )


def normalize_main_sim_state(state: MainSimState, box: Box) -> MainSimState:
    """
    Cast all values to [0,1]
    """
    return MainSimState(
        normalize_point3D(state.ball_position, box),
        normalize_point3D(state.ball_velocity, box),
        normalize_joint_states(state.joint_positions),
        normalize_joint_states(state.joint_velocities),
        normalize_cartesian_pose(state.racket_cartesian, box),
        state.contact,
        state.iteration,
        state.time_stamp,
    )


def normalize_states(
    state: States,
    box: Box,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> States:
    """
    Cast all values to [0,1]
    """
    return States(
        normalize_pressure_robot_state(
            state.pressure_robot, min_pressures, max_pressures
        ),
        normalize_main_sim_state(state.main_sim, box),
        [normalize_extra_balls_state(eb, box) for eb in state.extra_balls],
    )


def normalize_states_history(
    states_history: StatesHistory,
    box: Box,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> StatesHistory:
    """
    Cast all values to [0,1]
    """
    return [
        normalize_states(state, box, min_pressures, max_pressures)
        for state in states_history
    ]
