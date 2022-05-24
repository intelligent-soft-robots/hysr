import math
import typing
import random
import pytest
from hysr import normalize
from hysr import hysr_types


def test_normalize():
    """Testing normalize method"""

    assert normalize.normalize((1.0, 2.0), (0.0, 0.0), (2.0, 4.0)) == (0.5, 0.5)

    assert normalize.normalize((1, 2), (0, 0), (2, 4)) == (0.5, 0.5)

    assert normalize.normalize((0.0, 0.0), (0.0, 0.0), (2.0, 4.0)) == (0.0, 0.0)

    assert normalize.normalize((2.0, 4.0), (0.0, 0.0), (2.0, 4.0)) == (1.0, 1.0)

    assert normalize.normalize((1.5, 3.0), (1.0, 2.0), (2.0, 4.0)) == (0.5, 0.5)

    with pytest.raises(ValueError):
        assert normalize.normalize((0.5, 3.0), (1.0, 2.0), (2.0, 4.0))

    with pytest.raises(ValueError):
        assert normalize.normalize((1.5, 4.1), (1.0, 2.0), (2.0, 4.0))


def test_denormalize():
    """Testing denormalize method"""

    v = (20.0, 100.0)
    min_ = (14.0, 99.0)
    max_ = (31.0, 133)

    normalized = normalize.normalize(v, min_, max_)
    denormalized = normalize.denormalize(normalized, min_, max_)

    assert denormalized == v


def test_normalize_items():
    """
    Testing: normalize_point3D, normalize_orientation3D,
    normalize_cartesian_pose, normalize_robot_pressures
    """

    box = ((1.0, 2.0, 2.0), (2.0, 4.0, 6.0))
    assert normalize.normalize_point3D((1.5, 3.0, 4.0), position_box=box) == (
        0.5,
        0.5,
        0.5,
    )

    max_velocity = 2.0
    assert normalize.normalize_point3D((2.0, 1.0, 0.0), max_velocity=max_velocity) == (
        1.0,
        0.5,
        0.0,
    )

    for _ in range(20):
        orientation = typing.cast(
            hysr_types.Orientation3D,
            tuple([random.uniform(-1.0, 1.0) for _ in range(9)]),
        )
        assert all(
            [
                o >= 0.0 and o <= 1.0
                for o in normalize.normalize_orientation3D(orientation)
            ]
        )

    box = ((1.0, 2.0), (2.0, 4.0))
    point = (1.0, 4.0)
    orientation = typing.cast(hysr_types.Orientation3D, tuple([0.0] * 9))
    normalized = normalize.normalize_cartesian_pose((point, orientation), box)
    assert normalized[0] == (0.0, 1.0)
    assert all([o == 0.5 for o in normalized[1]])

    joints = (0, math.pi, -math.pi, 2 * -math.pi)
    normalized = normalize.normalize_joint_states(joints)
    assert normalized[0] == 0.5
    assert normalized[1] == 0.75
    assert normalized[2] == 0.25
    assert normalized[3] == 0

    min_pressures = typing.cast(hysr_types.JointPressures, tuple([(1000, 2000)] * 4))
    max_pressures = typing.cast(hysr_types.JointPressures, tuple([(2000, 6000)] * 4))
    pressures = ((1000, 2000), (2000, 6000), (1500, 4000), (1000, 6000))
    normalized = normalize.normalize_robot_pressures(
        pressures, min_pressures, max_pressures
    )
    assert normalized[0] == (0.0, 0.0)
    assert normalized[1] == (1.0, 1.0)
    assert normalized[2] == (0.5, 0.5)
    assert normalized[3] == (0.0, 1.0)
