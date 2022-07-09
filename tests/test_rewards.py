from enum import Enum
import pytest
import typing
import hysr


@pytest.fixture
def get_factory_class(
    request, scope="function"
) -> typing.Generator[hysr.hysr_types.FactoryClass, None, None]:

    reward_class: typing.Union[
        typing.Type[hysr.native_rewards.BasicRewards],
        typing.Type[hysr.native_rewards.SmashRewards],
    ]

    c: float = (3.0,)
    rtt_cap: float = (-0.2,)

    reward_class = request.test_class

    yield hysr.hysr_types.FactoryClass(
        f"hysr.native_rewards.{reward_class}", [c, rtt_cap], {}
    )


class Distance(Enum):
    FAR = 0
    CLOSE = 1
    CLOSER = 2


class Velocity(Enum):
    SLOW = 0
    FAST = 1


def _get_trajectories(
    ball_racket: Distance, ball_target: Distance, ball_velocity: Velocity
) -> typing.Tuple[
    hysr.hysr_types.Point3D,
    typing.Sequence[hysr.hysr_types.Point3D],
    typing.Sequence[hysr.hysr_types.Point3D],
    typing.Sequence[hysr.hysr_types.Point3D],
]:
    """
    Returns the position of the target, as well as 6 points trajectories
    for the ball position, the ball velocity and the robot racket position.
    """

    target = [-1.0] * 3

    racket = [1.0] * 3

    racket_traj = [racket] * 6

    ball_far_racket = [[0.0] * 3] * 3

    ball_close_racket = [[0.0] * 3, [1.2] * 3, [2.0] * 3]

    ball_closer_racket = [[0.0] * 3, [1.1] * 3, [2.0] * 3]

    ball_far_target = [[-2.0] * 3] * 3

    ball_close_target = [[-0.9] * 3, [-1.2] * 3, [-2.0] * 3]

    ball_closer_target = [[-0.9] * 3, [-1.01] * 3, [-2.0] * 3]

    if ball_racket == Distance.FAR:
        position_traj = ball_far_racket
    elif ball_racket == Distance.CLOSE:
        position_traj = ball_close_racket
    elif ball_racket == Distance.CLOSER:
        position_traj = ball_closer_racket

    if ball_target == Distance.FAR:
        position_traj.extend(ball_far_target)
    elif ball_target == Distance.CLOSE:
        position_traj.extend(ball_close_target)
    elif ball_target == Distance.CLOSER:
        position_traj.extend(ball_closer_target)

    if ball_velocity == Velocity.SLOW:
        velocity_traj = [[1.0] * 3] * 6
    elif ball_velocity == Velocity.FAST:
        velocity_traj = [[2.0] * 3] * 6

    return target, racket_traj, position_traj, velocity_traj  # type: ignore


def _compute_reward(
    reward_function: hysr.hysr_types.RewardFunction,
    contacts: typing.Sequence[bool],
    ball_racket: Distance,
    ball_target: Distance,
    ball_velocity: Velocity,
    c: float = 3.0,
    rtt_cap: float = -0.2,
):
    """
    Generate trajectories for the ball position, ball velocity
    and racket position, and compute the related reward.
    """

    target, racket_traj, position_traj, velocity_traj = _get_trajectories(
        ball_racket, ball_target, ball_velocity
    )

    return hysr.rewards.compute_reward(
        reward_function,
        target,
        position_traj,
        velocity_traj,
        contacts,
        racket_traj,
        c,
        rtt_cap,
    )


def test_basic_and_smash_reward():

    """
    Test basic_reward and smash_reward, by comparing
    rewards one with another (e.g. ball finding themselves
    closer to the target should have higher reward, etc)
    """

    reward_types = {
        "basic_reward": hysr.native_rewards.basic_reward,
        "smash_reward": hysr.native.rewards.smash_reward,
    }

    for reward_type, reward_function in reward_types.items():

        no_contacts = [False] * 6
        contacts = [False] * 3 + [True] * 3

        ##############
        # no contact #
        ##############

        # far from the racket
        r1 = (
            _compute_reward(
                reward_function, no_contacts, Distance.FAR, Distance.FAR, Velocity.SLOW
            ),
        )

        # closer to the racket
        r2 = (
            _compute_reward(
                reward_function,
                no_contacts,
                Distance.CLOSE,
                Distance.FAR,
                Velocity.SLOW,
            ),
        )

        # even closer to the racket
        r3 = (
            _compute_reward(
                reward_function,
                no_contacts,
                Distance.CLOSER,
                Distance.FAR,
                Velocity.SLOW,
            ),
        )

        # the closer to the racket, the higher the
        # reward
        assert r1 < r2
        assert r2 < r3

        # if no contact, distance ball/target
        # does not matter
        r2_bis = (
            _compute_reward(
                reward_function,
                no_contacts,
                Distance.CLOSE,
                Distance.CLOSE,  # change here !
                Velocity.SLOW,
            ),
        )
        assert r2 == r2_bis

        # if no contact, velocity of the ball does not
        # matter
        r2_ter = (
            _compute_reward(
                reward_function,
                no_contacts,
                Distance.CLOSE,
                Distance.CLOSE,
                Velocity.FAST,  # change here !
            ),
        )
        assert r2 == r2_ter

        ###########
        # contact #
        ###########

        # far from the target
        r4 = (
            _compute_reward(
                reward_function, contacts, Distance.CLOSER, Distance.FAR, Velocity.SLOW
            ),
        )

        # closer to the target
        r5 = (
            _compute_reward(
                reward_function,
                contacts,
                Distance.CLOSER,
                Distance.CLOSE,
                Velocity.SLOW,
            ),
        )

        # even closer to the target
        r6 = (
            _compute_reward(
                reward_function,
                contacts,
                Distance.CLOSER,
                Distance.CLOSER,
                Velocity.SLOW,
            ),
        )

        # the closer to the target, the higher the reward
        assert r5 > r4
        assert r6 > r5

        # contact is always better than no contact
        assert r4 > r3
        assert r5 > r3
        assert r6 > r3

        r7 = (
            _compute_reward(
                reward_function,
                contacts,
                Distance.CLOSER,
                Distance.CLOSER,
                Velocity.FAST,
            ),
        )

        # Faster is better than slow
        if reward_type == "basic_reward":
            assert r7 == r6
        else:
            assert r7 > r6


@pytest.mark.parametrize(
    "get_factory_class",
    [[hysr.native_rewards.BasicRewards, hysr.native_rewards.SmashRewards]],
    indirect=True,
)
def test_native_rewards_factory(get_factory_class):
    """
    Test reward_factory succeed in returning an instance of Rewards.
    """

    factory_class: hysr.FactoryClass = get_factory_class

    reward: hysr.Rewards = hysr.reward_factory(
        factory_class
    )  # noqa: F841 (variable not used)
