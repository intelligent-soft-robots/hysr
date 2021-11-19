import typing
import context

Position = typing.Tuple[float, float, float]

_trajectory_reader_g = context.BallTrajectories()


class TrajectoryGetter:
    def get_one(self) -> typing.Tuple[Position]:
        raise NotImplementedError(
            str("Subclasses of TrajectoryGetter" "must implement the get_one method")
        )

    def get(self) -> typing.Sequence[typing.Sequence[Position]]:
        raise NotImplementedError(
            str("Subclasses of TrajectoryGetter" "must implement the get method")
        )

    def get_sample_rate(self) -> float:
        raise NotImplementedError(
            str(
                "Subclasses of TrajectoryGetter"
                "must implement the method "
                "get_sample_rate"
            )
        )


class LineTrajectory(TrajectoryGetter):

    """
    Class for getting a line trajectory, i.e. trajectories of position
    going from the specified starting point to the specified end point
    over the specified duration, considering the ball is moving from one
    position to the next at the specified rate.
    """

    def __init__(
        self,
        start_position: Position,
        end_position: Position,
        duration: float,
        sampling_rate: float,
    ):
        self._start = start_position
        self._end = end_position
        self._duration = duration
        self._rate = sampling_rate

    def get_one(self) -> typing.Tuple[Position]:
        """
        Returns a trajectory of 3d points
        """
        duration_ms = self._duration * 1e3
        trajectory_points = context.duration_line_trajectory(
            self._start, self._end, duration_ms, sampling_rate=self._rate
        )
        return trajectory_points

    def get(self, nb_trajectories: int) -> typing.Sequence[typing.Sequence[Position]]:
        """
        Returns a list of trajectories, all idendical.
        """
        return [self.get_one()] * nb_trajectories

    def get_sample_rate(self) -> float:
        """
        Returns the sampling rate of the trajectories
        returned by the get and get_one methods (in seconds)
        """
        return self._rate


class IndexedRecordedTrajectory(TrajectoryGetter):

    """
    Class for getting the pre-recorded ball trajectory corresponding
    to the provided index.
    If the provided index is too high (i.e. such recorded trajectory index
    does not exist), an IndexError is thrown).
    """

    def __init__(self, index: int):
        global _trajectory_reader_g
        if index >= _trajectory_reader.size():
            raise IndexError(
                str(
                    "Trajectory index {} not supported " "(index supported up to {})"
                ).format(index, _trajectory_reader.size())
            )
        self._index = index

    def get_one(self) -> typing.Sequence[Position]:
        """
        Returns a trajectory of 3d points
        """
        global _trajectory_reader_g
        trajectory_points = _trajectory_reader_g.get_trajectory(self._index)
        return trajectory_points

    def get(self, nb_trajectories: int) -> typing.Sequence[typing.Sequence[Position]]:
        """
        Returns a list of trajectories, all idendical.
        """
        return [self.get_one()] * nb_trajectories

    def get_sample_rate(self) -> float:
        """
        Returns the sampling rate of the trajectories
        returned by the get and get_one methods (in seconds)
        """
        global _trajectory_reader_g
        return float(_trajectory_reader_g.get_sampling_rate_ms()) * 1e3


class RandomRecordedTrajectory(TrajectoryGetter):

    """
    Class for getting one of the pre-recorded ball trajectory,
    randomly selected
    """

    def __init__(self):
        pass

    def get_one(self) -> typing.Sequence[Position]:
        """
        Returns a trajectory of 3d points
        """
        global _trajectory_reader_g
        _, trajectory_points = _trajectory_reader_g.random_trajectory()
        return trajectory_points

    def get(self, nb_trajectories: int) -> typing.Sequence[typing.Sequence[Position]]:
        """
        Returns a list of trajectories, all differents.
        If nb_trajectories is too high (i.e. there are not so many different recorded
        trajectories), a ValueError is thrown.
        """
        global _trajectory_reader_g
        if nb_trajectories >= _trajectory_reader.size():
            raise ValueError(
                str(
                    "Requested number of trajectories {} not supported ",
                    "(there are only {} recorded trajectories)",
                ).format(index, _trajectory_reader.size())
            )
        return _trajectory_readrer_g.get_different_random_trajectories(nb_trajectories)

    def get_sample_rate(self) -> float:
        """
        Returns the sampling rate of the trajectories
        returned by the get and get_one methods (in seconds)
        """
        global _trajectory_reader_g
        return float(_trajectory_reader_g.get_sampling_rate_ms()) * 1e3
