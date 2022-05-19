import typing
from numbers import Number
from .hysr_types import (
    Point3D,
    Box,
    Orientation3D,
    CartesianPose,
    JointStates,
    JointPressures,
    RobotPressures,
)

FloatTuple = typing.Union[
    Point3D,
    Orientation3D,
    JointStates,
    JointPressures,
    RobotPressures,
    typing.Tuple[float, float],
]

NumberTuple = typing.TypeVar(
    "NumberTuple",
    Point3D,
    Orientation3D,
    JointStates,
    JointPressures,
    RobotPressures,
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


def _check_bounds(
    f: typing.Callable[[NumberTuple, NumberTuple, NumberTuple], FloatTuple]
) -> typing.Callable[[NumberTuple, NumberTuple, NumberTuple], FloatTuple]:
    """
    Decorator applying _value_error_if_out_of_bounds
    """
    def wrapper(
        input_: NumberTuple, min_values: NumberTuple, max_values: NumberTuple
    ) -> FloatTuple:
        _value_error_if_out_of_bounds(input_, min_values, max_values)
        return f(input_, min_values, max_values)
    return wrapper


@_check_bounds
def normalize(
    input_: NumberTuple, min_values: NumberTuple, max_values: NumberTuple
) -> FloatTuple:
    """
    Cast all values in the input_ to [0,1],
    assuming the input_ values are all between
    the provided min and max values.
    Raise a ValueError if any inpout value in not
    in the expected min/max interval.
    """
    return typing.cast(
        FloatTuple,
        tuple(
            [
                float(p - min_) / float(max_ - min_)
                for p, min_, max_ in zip(input_, min_values, max_values)
            ]
        ),
    )


@_check_bounds
def normalize_point3D(point: Point3D, box: Box) -> Point3D:
    """
    Normalize the coordinates of the points, assuming
    all points are inside the box.
    """
    return normalize(point, box[0], box[1])


@_check_bounds
def normalize_orientation3D(orientation: Orientation3D) -> Orientation3D:
    """
    Normalize the orientation, i.e. cast
    its values from [-1,1] to [0,1]
    """
    return normalize(orientation, (-1.0, -1.0), (1.0, 1.0))


@_check_bounds
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


@_check_bounds
def normalize_joint_states(joint_states: JointStates) -> JointStates:
    """
    Casts all values from [-2pi,2pi] to [0,1]
    """
    return normalize(
        joint_states,
        typing.cast(JointStates, tuple([-2.0 * math.pi] * 4)),
        typing.cast(JointStates, tuple([+2.0 * math.pi] * 4)),
    )


@_check_bounds
def normalize_joint_pressures(
    pressures: JointPressures,
    min_pressures: JointPressures,
    max_pressures: JointPressures,
) -> JointPressures:
    """
    Cast all pressures value to [0,1]
    """
    normalized: FloatTuple = normalize(pressures, min_pressures, max_pressures)
    return typing.cast(JointPressures, tuple([round(n) for n in normalize]))


def normalize_robot_pressures(
    robot_pressures: RobotPressures,
    min_pressures: RobotPressures,
    max_pressures: RobotPressures,
) -> RobotPressures:
    return typing.cast(
        RobotPressures,
        tuple(
            [
                normalize_joint_pressures(j, min_, max_)
                for j, min_, max_ in zip(robot_pressures, min_pressures, max_pressures)
            ]
        ),
    )
