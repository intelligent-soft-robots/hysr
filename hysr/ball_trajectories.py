import pathlib
import typing
import context
from .types import StampedTrajectory, StampedTrajectories, DurationPoint


class TrajectoryGetter:
    """
    Abstract super class for objets returning trajectories.
    """

    def get_one(self) -> StampedTrajectory:
        """
        Returns a trajectory.
        """
        raise NotImplementedError(
            str("Subclasses of TrajectoryGetter" "must implement the get_one method")
        )

    def get(self) -> StampedTrajectories:
        """
        Returns a sequence of trajectories.
        """
        raise NotImplementedError(
            str("Subclasses of TrajectoryGetter" "must implement the get method")
        )

    @staticmethod
    def iterate(
        input: StampedTrajectory
    ) -> typing.Generator[DurationPoint, None, None]:
        """
        Generator over the trajectory. 
        Yields tuples (duration in microseconds, state), state having
        a position and a velocity attribute.
        """
        return context.ball_trajectories.BallTrajectories.iterate(input)


class LineTrajectory(TrajectoryGetter):

    """
    Class for getting a line trajectory, i.e. trajectories of position
    going from the specified starting point to the specified end point
    over the specified duration, considering the ball is moving from one
    position to the next at the specified rate.
    """

    def __init__(
        self,
        start_position: typing.Sequence[float],
        end_position: typing.Sequence[float],
        duration_seconds: float,
        sampling_rate_seconds: float,
    ):
        self._sampling_rate = sampling_rate_seconds
        duration_trajectory = context.ball_trajectories.duration_line_trajectory(
            start_position,
            end_position,
            duration_seconds * 1000.0,
            sampling_rate_seconds,
        )
        self._stamped_trajectory = context.ball_trajectories.to_stamped_trajectory(
            duration_trajectory
        )

    def get_one(self) -> StampedTrajectory:
        """
        Returns a trajectory
        """
        return self._stamped_trajectory

    def get(self, nb_trajectories: int) -> StampedTrajectories:
        """
        Returns a sequence of trajectories, all idendical.
        """
        return [self._stamped_trajectory] * nb_trajectories

    def get_sample_rate(self) -> float:
        """
        Returns the sampling rate of the trajectories
        returned by the get and get_one methods (in seconds)
        """
        return self._sampling_rate


class IndexedRecordedTrajectory(TrajectoryGetter):

    """
    Class for getting the pre-recorded ball trajectory corresponding
    to the provided index.
    The trajectories are read from a hdf5 file possibly containing different
    sets (group) of balls. If the group does not exists or the trajectory index
    in the specified group does not exists, a KeyError is raised.
    """

    def __init__(self, index: int, group: str, hdf5_path: pathlib.Path = None):
        with context.RecordedBallTrajectoris(hdf5_path) as rbt:
            if not group in rbt.get_groups():
                raise KeyError(
                    "failed to find the group {} in {}".format(group, hdf5_path)
                )
            indexes = rbt.get_indexes(group)
            if not index in indexes:
                max_index = max([int(i) for i in indexes])
                raise KeyError(
                    "failed to find the index {} "
                    "in the group {} of the file {} "
                    "(max index: {})".format(index, group, hdf5_path, max_index)
                )
            self._stamped_trajectory = rbt.get_stamped_trajectory(
                group, index, direct=True
            )
        self._index = index

    def get_one(self) -> StampedTrajectory:
        """
        Returns a trajectory 
        """
        return self._stamped_trajectory

    def get(self, nb_trajectories: int) -> StampedTrajectories:
        """
        Returns a list of trajectories, all idendical.
        """
        return [self._stamped_trajecotries] * nb_trajectories


class RandomRecordedTrajectory(TrajectoryGetter):

    """
    Class for getting one of the pre-recorded ball trajectory,
    randomly selected. The trajectories are read from an hdf5 file.
    If the specified trajectories group in the file is not found,
    a KeyError is raise.
    """

    def __init__(self, group: str, hdf5_path: pathlib.Path = None):
        self._ball_trajectories = context.ball_trajectories.BallTrajectories(
            group, hdf5_path=hdf5_path
        )

    def get_one(self) -> StampedTrajectory:
        """
        Returns a trajectory 
        """
        return self._ball_trajectories.random_trajectory()

    def get(self, nb_trajectories: int) -> StampedTrajectories:
        """
        Returns a sequence of trajectories, all differents.
        If nb_trajectories is too high (i.e. there are not so many different recorded
        trajectories), a ValueError is thrown.
        """
        return self._ball_trajectories.get_different_random_trajectories(
            nb_trajectories
        )
