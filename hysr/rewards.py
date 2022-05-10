import math
from .types import *


def _norm(vector: Point3D):
    """
    Returns the norm of the vector
    """
    return math.sqrt(sum([v**2 for v in vector]))


def _distance(p1: Point3D, p2: Point3D):
    """
    Returns the distance between p1 and p2
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def _min_distance(t1: typing.Sequence[Point3D], t2: typing.Sequence[Point3D]):
    """
    Returns the minimal distance between points in
    t1 and t2 that are at the same index
    """
    return min([_distance(p1, p2) for p1, p2 in zip(t1, t2)])


def _no_hit_reward(min_distance_ball_racket):
    return -min_distance_ball_racket


def _return_task_reward(min_distance_ball_target, c, rtt_cap):
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = max(reward, rtt_cap)
    return reward


def _smash_task_reward(min_distance_ball_target, max_ball_velocity, c, rtt_cap):
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = reward * max_ball_velocity
    reward = max(reward, rtt_cap)
    return reward


def _compute_reward(
    smash,
    min_distance_ball_racket,
    min_distance_ball_target,
    max_ball_velocity,
    c,
    rtt_cap,
):

    # i.e. the ball did not hit the racket,
    # so computing a reward based on the minimum
    # distance between the racket and the ball
    if min_distance_ball_racket is not None:
        return _no_hit_reward(min_distance_ball_racket)

    # the ball did hit the racket, so computing
    # a reward based on the ball / target

    if smash:
        return _smash_task_reward(
            min_distance_ball_target, max_ball_velocity, c, rtt_cap
        )

    else:
        return _return_task_reward(min_distance_ball_target, c, rtt_cap)


def basic_reward(
    min_distance_ball_racket: float,
    min_distance_ball_target: float,
    max_ball_velocity: float,
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
    min_distance_ball_racket, min_distance_ball_target, max_ball_velocity, c, rtt_cap
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
    reward_function: RewardFunction,
    target: Point3D,
    ball_positions: typing.Sequence[Point3D],
    ball_velocities: typing.Sequence[Point3D],
    contacts: typing.Sequence[bool],
    racket_cartesians: typing.Sequence[Point3D],
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
    reward_function: RewardFunction, target: Point3D, states_history: StatesHistory
) -> typing.Union[
    float,  # if no extra balls
    typing.Tuple[float, typing.Sequence[float]],  # if extra balls
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
    simulation) and a list of rewards (for the extra balls).
    """

    # reward for the ball of the main sim
    main_sims = [s.main_sim for s in states_history]
    ball_positions = [ms.ball_position for ms in main_sims]
    ball_velocities = [ms.ball_velocity for ms in main_sims]
    contacts = [ms.contact.contact_occured for ms in main_sims]
    racket_cartesians = [ms.racket_cartesian for ms in main_sims]
    main_ball_reward = compute_reward(
        reward_function,
        target,
        ball_positions,
        ball_velocities,
        contacts,
        racket_cartesians,
    )

    # if not extra balls, returning only
    # the reward of the main ball
    if not states_history[0].extra_balls:
        return main_ball_reward

    # reward for all extra balls
    def _extra_ball_reward(
        reward_function: RewardFunction,
        target: Point3D,
        index: int,
        extra_balls_history: typing.Sequence[ExtraBallsState],
        racket_cartesians: typing.Sequence[Point3D],
    ):
        ball_positions = [eb.ball_positions[index] for eb in extra_balls_history]
        ball_velocities = [eb.ball_velocities[index] for eb in extra_balls_history]
        contacts = [eb.contacts[index] for eb in extra_balls_history]
        return compute_reward(
            reward_function,
            target,
            ball_positions,
            ball_velocities,
            contacts,
            racket_cartesian,
        )

    extra_balls_history = [s.extra_balls for s in states_history]
    racket_cartesians = [eb.racket_cartesian for eb in extra_balls_history]
    extra_balls_rewards = [
        _extra_ball_reward(
            reward_function, target, index, extra_balls_history, racket_cartesians
        )
        for index in range(len(states_histories[0].extra_balls))
    ]

    # returning all the rewards
    return main_ball_reward, extra_balls_rewards
