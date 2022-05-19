import math
import typing
from typing import Optional
from . import hysr_types


def _norm(vector: hysr_types.Point3D) -> float:
    """
    Returns the norm of the vector
    """
    return math.sqrt(sum([v ** 2 for v in vector]))


def _distance(p1: hysr_types.Point3D, p2: hysr_types.Point3D) -> float:
    """
    Returns the distance between p1 and p2
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def _min_distance(
    t1: typing.Iterable[hysr_types.Point3D], t2: typing.Iterable[hysr_types.Point3D]
) -> float:
    """
    Returns the minimal distance between points in
    t1 and t2 that are at the same index
    """
    return min([_distance(p1, p2) for p1, p2 in zip(t1, t2)])


def _no_hit_reward(min_distance_ball_racket: float) -> float:
    # used for rewards for which the ball did
    # not touch the racket
    return -min_distance_ball_racket


def _return_task_reward(
    min_distance_ball_target: float, c: float, rtt_cap: float
) -> float:
    # used for rewards for which the ball did
    # touch the racket
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = max(reward, rtt_cap)
    return reward


def _smash_task_reward(
    min_distance_ball_target: float, max_ball_velocity: float, c: float, rtt_cap: float
) -> float:
    # used for rewards for which the ball did
    # touch the racket, with smashing objective
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = reward * max_ball_velocity
    reward = max(reward, rtt_cap)
    return reward


def _compute_reward(
    smash: bool,
    min_distance_ball_racket: Optional[float],
    min_distance_ball_target: Optional[float],
    max_ball_velocity: Optional[float],
    c: float,
    rtt_cap: float,
) -> float:
    # common code for computing both the
    # basic and smash reward

    # i.e. the ball did not hit the racket,
    # so computing a reward based on the minimum
    # distance between the racket and the ball
    if min_distance_ball_racket is not None:
        return _no_hit_reward(min_distance_ball_racket)

    # the ball did hit the racket, so the reward
    # will depend on the distance between the
    # ball and the target. Checking it is not None.
    if min_distance_ball_target is None:
        raise ValueError(
            "Computing reward: the ball hit the racket "
            "but min_distance_ball_target, required to "
            "compute the reward, is None"
        )

    # the ball did hit the racket, so computing
    # a reward based on the ball / target

    if smash:
        # computing the smash reward requires max_ball_velocity, which
        # should not be None
        if max_ball_velocity is None:
            raise ValueError(
                "Computing reward: max_ball_velocity, required to "
                "compute the smash reward, is None"
            )
        return _smash_task_reward(
            min_distance_ball_target, max_ball_velocity, c, rtt_cap
        )

    else:
        return _return_task_reward(min_distance_ball_target, c, rtt_cap)


def basic_reward(
    min_distance_ball_racket: Optional[float],
    min_distance_ball_target: Optional[float],
    max_ball_velocity: Optional[float],
    c: float,
    rtt_cap: float,
):
    """
    Computes a reward suitable for table tennis:
    the closest the ball goes to the racket, the
    higher the reward. If there is a contact: the closest
    the ball to the target, the highest the reward.
    For details, see the publication "Learning to Play Table
    Tennis From Scratch using Muscular Robots"

    Arguments
    ---------
    min_distance_ball_racket:
      in meters, None if there has been a contact
      between the ball and the racket
    min_distance_ball_target:
      in meters, None if there has been no contact
    max_ball_velocity:
      in meters per second, as observed after contact
      (None if no contact). Unused.
    c :
      normalization constant
    rtt_cap
      reward cap
    """

    return _compute_reward(
        False,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


def smash_reward(
    min_distance_ball_racket: Optional[float],
    min_distance_ball_target: Optional[float],
    max_ball_velocity: Optional[float],
    c: float,
    rtt_cap: float,
):
    """
    Similar to basic_reward, except that the max_ball_velocity
    is used: the faster the ball goes toward the target,
    the higher the reward
    """
    return _compute_reward(
        True,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


def compute_reward(
    reward_function: hysr_types.RewardFunction,
    target: hysr_types.Point3D,
    ball_positions: typing.Sequence[hysr_types.Point3D],
    ball_velocities: typing.Iterable[hysr_types.Point3D],
    contacts: typing.Sequence[bool],
    racket_cartesians: typing.Iterable[hysr_types.Point3D],
    c: float,
    rtt_cap: float,
) -> float:

    """
    Computes the minimal distance
    between the ball and the racket (None if contact occured),
    the maximal observed ball velocity (None if no contact occured)
    and the minimal distance between the ball and the target
    (None if no contact between the ball and the racket occured);
    and apply the reward function.
    """

    contact = any(contacts)

    min_distance_ball_racket: Optional[float]

    if not contact:
        min_distance_ball_racket = _min_distance(ball_positions, racket_cartesians)
        min_distance_ball_target = None
        max_ball_velocity = None

    else:
        contact_index = contacts.index(True)
        min_distance_ball_racket = None
        # trimming ball positions to steps after contact
        positions = ball_positions[contact_index:]
        min_distance_ball_target = min([_distance(p, target) for p in positions])
        max_ball_velocity = max([_norm(v) for v in ball_velocities])

    return reward_function(
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


def compute_rewards(
    reward_function: hysr_types.RewardFunction,
    target: hysr_types.Point3D,
    states_history: hysr_types.StatesHistory,
    c: float,
    rtt_cap: float,
) -> typing.Union[
    float,  # if no extra balls
    typing.Tuple[
        float, typing.Sequence[typing.Sequence[float]]
    ],  # if extra balls, one list per extra balls set
]:

    """
    For each ball (in the main simulation and in the extra balls
    simulations): Computes the minimal distance
    between the ball and the racket (None if contact occured),
    the maximal observed ball velocity (None if no contact occured)
    and the minimal distance between the ball and the target
    (None if no contact between the ball and the racket occured);
    and apply the reward function. Returns either a float (if no
    extra balls) or both a float (reward for the ball in the main
    simulation) and a list of list of rewards (for the extra balls,
    one list of reward per extra balls set).
    """

    # reward for the ball of the main sim
    main_sims = [s.main_sim for s in states_history]
    ball_positions = [ms.ball_position for ms in main_sims]
    ball_velocities = [ms.ball_velocity for ms in main_sims]
    contacts = [ms.contact.contact_occured for ms in main_sims]
    racket_cartesians = [ms.racket_cartesian[0] for ms in main_sims]
    main_ball_reward = compute_reward(
        reward_function,
        target,
        ball_positions,
        ball_velocities,
        contacts,
        racket_cartesians,
        c,
        rtt_cap,
    )

    # if not extra balls, returning only
    # the reward of the main ball
    if not states_history[0].extra_balls:
        return main_ball_reward

    # reward for all extra balls

    # [ [ExtraBallsSet 1 at time stamp 1,ExtraBallsSet 2 at time stamp1, ...],
    #   [ExtraBallsSet 1 at time stamp 1,ExtraBallsSet 2 at time stamp2, ...],
    #   ... ]

    # for each set, a list of rewards (one per ball in the set)
    extra_balls_rewards: typing.List[typing.List[float]] = []
    for set_index in range(len(states_history[0].extra_balls)):
        set_history: typing.Sequence[hysr_types.ExtraBallsState] = [
            sh.extra_balls[set_index] for sh in states_history
        ]
        set_racket_cartesians: typing.Sequence[hysr_types.Point3D] = [
            s.racket_cartesian for s in set_history
        ]
        # rewards for each ball in the set
        rewards = []
        for ball_index in range(len(set_history[0].ball_positions)):
            ball_positions = [s.ball_positions[ball_index] for s in set_history]
            ball_velocities = [s.ball_velocities[ball_index] for s in set_history]
            contacts = [s.contacts[ball_index] for s in set_history]
            rewards.append(
                compute_reward(
                    reward_function,
                    target,
                    ball_positions,
                    ball_velocities,
                    contacts,
                    set_racket_cartesians,
                    c,
                    rtt_cap,
                )
            )
        extra_balls_rewards.append(rewards)

    # returning all the rewards
    return main_ball_reward, extra_balls_rewards
