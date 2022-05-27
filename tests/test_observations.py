import pytest
import random
import math
import typing
import functools
import itertools
import context
from hysr import hysr_types
from hysr import observations

# instance's attribute name : tuple of normalized corresponding values
# (used for tests "ground truth")
AttrDict = typing.Dict[str, typing.Tuple[float, ...]]

# for normalization configuration, i.e. position box, max velocity, etc
ConfigDict = typing.Dict[
    str, typing.Union[hysr_types.Box, float, hysr_types.RobotPressures]
]


@pytest.fixture
def get_state_instances(
    request, scope="function"
) -> typing.Generator[
    typing.Tuple[
        ConfigDict,
        typing.Tuple[hysr_types.MainSimState, AttrDict],
        typing.Tuple[hysr_types.ExtraBallsState, AttrDict],
        typing.Tuple[hysr_types.PressureRobotState, AttrDict],
    ],
    None,
    None,
]:

    """
    Generating "ground truth" normalizations, i.e. instances
    of states and dictionaries of related normalized values
    (tuple of float computed "by hand").
    Returns also the related configuration dictionary.
    """

    box = (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)
    max_velocity = 2.0
    max_angular_velocity = 1.0

    min_pressures: hysr_types.RobotPressures = (
        (1000, 1000),
        (2000, 2000),
        (1000, 2000),
        (1000, 2000),
    )

    max_pressures: hysr_types.RobotPressures = (
        (6000, 6000),
        (6000, 6000),
        (9000, 6000),
        (6000, 9000),
    )

    norm_config: ConfigDict = {
        "position_box": box,
        "max_velocity": max_velocity,
        "max_angular_velocity": max_angular_velocity,
        "min_pressures": min_pressures,
        "max_pressures": max_pressures,
    }

    main_sim = hysr_types.MainSimState(
        (5.0, 7.5, 5.0),  # ball_position (0.5,0.75,0.5)
        (0.5, 0.5, -1.0),  # ball_velocity (0.625,0.625,0.25)
        (0.0, 0.0, 0.0, 0.0),  # joint_positions (0.5,0.5,0.5,0.5)
        (0.5, 0.5, 0.5, 0.5),  # joint_velocities (0.75,0.75,0.75,0.75)
        (
            (2.5, 2.5, 2.5),  # racket_cartesian (position 3d) (0.25,0.25,0.25)
            (
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),  # racket_cartesian (orientation) (1.,.5,.5,.5,1,.5,.5,.5,1)
        ),
        context.ContactInformation(),  # contact (0,)
        -1,  # iteration
        -1,  # time_stamp
    )

    main_sim_normalized: AttrDict = {
        "ball_position": (0.5, 0.75, 0.5),
        "ball_velocity": (0.625, 0.625, 0.25),
        "joint_positions": (0.5, 0.5, 0.5, 0.5),
        "joint_velocities": (0.75, 0.75, 0.75, 0.75),
        "racket_cartesian": (0.25, 0.25, 0.25, 1.0, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1),
        "contact": (0.0,),
    }

    pressure_robot = hysr_types.PressureRobotState(
        (0.0, math.pi, 0.0, -math.pi),  # joint_positions (0.5,0.75,0.5,0.25)
        (0.0, 1.0, -1.0, -0.5),  # joint_velocities (0.5,1.,0.,0.25)
        (  # desired_pressures
            (1000, 1000),  # (0.,0.,1.,1.,0.25,0.5,0.75,0.75)
            (6000, 6000),
            (3000, 4000),
            (4750, 7250),
        ),
        (  # observed_pressures
            (1000, 1000),  # (0.,0.,1.,1.,0.25,0.5,0.5,0.25)
            (6000, 6000),
            (3000, 4000),
            (3500, 3750),
        ),
        -1,  # iteration
        -1,  # time_stamp
    )

    pressure_robot_normalized: AttrDict = {
        "joint_positions": (0.5, 0.75, 0.5, 0.25),
        "joint_velocities": (0.5, 1.0, 0.0, 0.25),
        "desired_pressures": (0.0, 0.0, 1.0, 1.0, 0.25, 0.5, 0.75, 0.75),
        "observed_pressures": (0.0, 0.0, 1.0, 1.0, 0.25, 0.5, 0.5, 0.25),
    }

    extra_balls = hysr_types.ExtraBallsState(
        (
            (5.0, 7.5, 5.0),  # ball positions (ball 1) (0.5,0.75,0.5)
            (0.0, 7.5, 10.0),  # ball positions (ball 2) (0.,0.75,1.)
        ),
        (
            (0, 0, 1),  # ball velocities (ball 1) (0.5,0.5,0.75)
            (1, 0.0, -1),  # ball velocities (Ball 2) (0.75.,0.5,0.25)
        ),
        (True, False),  # contacts (1.,0.)
        (0.0, 0.0, 0.0, 0.0),  # joint_positions (0.5,0.5,0.5,0.5)
        (0.5, 0.5, 0.5, -0.5),  # joint_velocities (0.75,0.75,0.75,0.25)
        (2.5, 2.5, 7.5),  # racket_cartesian (position 3d) (0.25,0.25,0.75)
        -1,
        -1,
    )

    extra_balls_normalized: AttrDict = {
        "ball_positions": (0.5, 0.75, 0.5, 0.0, 0.75, 1.0),
        "ball_velocities": (0.5, 0.5, 0.75, 0.75, 0.5, 0.25),
        "contacts": (1.0, 0.0),
        "joint_positions": (0.5, 0.5, 0.5, 0.5),
        "joint_velocities": (0.75, 0.75, 0.75, 0.25),
        "racket_cartesian": (0.25, 0.25, 0.75),
    }

    yield (
        norm_config,
        (main_sim, main_sim_normalized),
        (extra_balls, extra_balls_normalized),
        (pressure_robot, pressure_robot_normalized),
    )


def _test_packing(
    instance: typing.Union[
        hysr_types.MainSimState,
        hysr_types.ExtraBallsState,
        hysr_types.PressureRobotState,
    ],
    normalized_attr: AttrDict,
    packing_function: typing.Callable[[typing.Any], observations.Observation],
    norm_config: typing.Dict[str, typing.Any],
):
    """
    applies the packing function to the instance, and compare the returned
    list of floats to the ground truth (normalized attrs).
    """

    # partial packing function with correct configuration
    # (for box, max_velocity, max_pressures, etc
    function = functools.partial(
        packing_function,
        **{
            arg_name: value
            for arg_name, value in norm_config.items()
            if arg_name in packing_function.__code__.co_varnames
        }
    )

    nb_attrs = len(normalized_attr)
    attrs = list(normalized_attr.keys())

    for _ in range(100):
        # attributes of the states that will be packed
        random.shuffle(attrs)
        test_attrs = attrs[: random.randint(1, nb_attrs - 1)]
        # applying the packing function, i.e. casting the
        # values of the attributes to a numpy array of floats
        normalized: typing.Tuple[float, ...] = tuple(function(instance, test_attrs))
        # computed manually
        ground_truth = tuple(
            itertools.chain.from_iterable(
                [normalized_attr[attr] for attr in test_attrs]
            )
        )
        # packed and manually computed are matching ?
        assert normalized == ground_truth


def test_packing(get_state_instances):
    """
    Testing pack_main_sim_state,
    pack_pressure_robot_state,
    pack_extra_balls_state
    """

    (
        norm_config,
        (main_sim_state, main_sim_normalized),
        (extra_balls_state, extra_balls_normalized),
        (pressure_robot_state, pressure_robot_normalized),
    ) = get_state_instances

    _test_packing(
        main_sim_state,  # instance
        main_sim_normalized,  # ground truth
        observations.pack_main_sim_state,  # function to test
        norm_config,  # configuration for normalization
    )

    _test_packing(
        pressure_robot_state,
        pressure_robot_normalized,
        observations.pack_pressure_robot_state,
        norm_config,
    )

    _test_packing(
        extra_balls_state,
        extra_balls_normalized,
        observations.pack_extra_balls_state,
        norm_config,
    )


def test_states_packing(get_state_instances):
    """
    Testing pack function
    """

    (
        norm_config,
        (main_sim_state, main_sim_normalized),
        (extra_balls_state, extra_balls_normalized),
        (pressure_robot_state, pressure_robot_normalized),
    ) = get_state_instances

    states = hysr_types.States(
        pressure_robot_state, main_sim_state, (extra_balls_state,)
    )

    # returns for the instance passed as argument:
    # - a list of attributes of the instance, randomly selected (e.g. "ball_positions","max_pressures",...)
    # - the list of related normalized values (floats)
    def _random_config(
        state_instance: typing.Union[
            hysr_types.PressureRobotState,
            hysr_types.MainSimState,
            hysr_types.ExtraBallsState,
        ],
        ground_truth: typing.Dict[str, typing.Tuple[float, ...]],
    ) -> typing.Tuple[typing.Tuple[str, ...], typing.Tuple[float, ...]]:

        all_attrs = list(ground_truth.keys())
        random.shuffle(all_attrs)
        test_attrs = all_attrs[: random.randint(0, len(ground_truth.keys()) - 1)]
        if len(test_attrs) == 0:
            return (), ()
        attrs_ground_truth = tuple(
            itertools.chain.from_iterable([ground_truth[attr] for attr in test_attrs])
        )
        return tuple(test_attrs), attrs_ground_truth

    # testing 100 configurations
    for _ in range(100):

        # list of randomly selected attributes and
        # related array of normalized float values
        # (computed by hand)
        main_sim_attrs, main_sim_ground = _random_config(
            main_sim_state, main_sim_normalized
        )
        extra_balls_attrs, extra_balls_ground = _random_config(
            extra_balls_state, extra_balls_normalized
        )
        pressure_robot_attrs, pressure_robot_ground = _random_config(
            pressure_robot_state, pressure_robot_normalized
        )

        # state attribute name (e.g. "main_sim"), related attribute (e.g. main_sim.ball_position)
        # and ground truth (normalized list of floats)
        config: typing.List[
            typing.Tuple[
                hysr_types.PackableState, typing.Sequence[str], typing.Sequence[float]
            ]
        ] = []

        # building the configuration for packing
        if main_sim_attrs:
            config.append(("main_sim", main_sim_attrs, main_sim_ground))
        if extra_balls_attrs:
            config.append(("extra_balls", extra_balls_attrs, extra_balls_ground))
        if pressure_robot_attrs:
            config.append(
                ("pressure_robot", pressure_robot_attrs, pressure_robot_ground)
            )

        if config:
            random.shuffle(config)
            ground_truth = tuple(itertools.chain.from_iterable([c[2] for c in config]))
            # removing the ground truth from the configuration
            config_: typing.Iterable[
                typing.Tuple[hysr_types.PackableState, typing.Sequence[str]]
            ] = [(c[0], c[1]) for c in config]
            observation_factory = observations.ObservationFactory(
                config_, **norm_config
            )
            packed = observation_factory.get(states)
            assert tuple(packed) == ground_truth
