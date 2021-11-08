import time, threading, typing, pam_mujoco


class ParallelBursts:
    """
    An instance of ParallelBurst will for each (o80) handle given as
    argument spawn a thread. When the burst method of the instance is called,
    all handles will be requested to call their burst function, in parallel.
    For this to work, the handle must correspond to backend that are running
    in bursting mode.

    Args:
        handles: list of handles
        wait: wait duration (in second) of the thread loops
    """

    def __init__(
        self, handles: typing.Sequence[pam_mujoco.MujocoHandle], wait: float = 0.0001
    ):

        self._handles = handles
        self._size = len(handles)
        self._wait = wait

        # the threads will run for as long _running is True
        self._running = True

        # after call to burst: for monitoring which handle
        # finished to burst and which one did not finish
        self._burst_done = [False] * self._size

        # how many bursts the handles should perform.
        # Also used as a signal for the handles that they
        # should burst (burst not None -> handles must
        # burst)
        self._nb_bursts = None

        # there is more than 1 handle, so we need threads
        # to have them running in parallel
        if self._size > 1:
            self._lock = threading.Lock()
            self._threads = [
                threading.Thread(target=self._run, args=(index,))
                for index in range(self._size)
            ]
            for thread in self._threads:
                thread.start()

        # only 1 handle, no thread required
        else:
            self._lock = None
            self._threads = None

    def _run(self, index: None) -> None:
        """
        method ran by each thread.
        index : index of the handle in the
                self._handles list
        """
        while self._running:
            with self._lock:
                # checking if burst is requested
                # nb_bursts is None : nothing requested
                # nb_burst>0 and burst_done : burst already done
                # nb_burst>0 and not burst_done : must burst !
                nb_bursts = self._nb_bursts
                burst_done = self._burst_done[index]
            if nb_bursts is not None and not burst_done:
                # bursting
                self._handles[index].burst(nb_bursts)
                with self._lock:
                    # to avoid bursting again for the
                    # same request
                    self._burst_done[index] = True
            else:
                # no bursting request
                time.sleep(self._wait)

    def burst(self, nb_bursts: int) -> None:
        """
        Requests all handles to burst
        Args:
            nb_bursts: number of requested bursts (i.e. number
                       of iteration) per handle
        """

        # there is only 1 handle, simply having it burst
        if self._size == 1:
            self._handles[0].burst(nb_bursts)

        # several handles running in parallel
        else:
            with self._lock:
                # None of the handles started bursting yet
                self._burst_done = [False] * self._size
                self._nb_bursts = nb_bursts
            all_bursts_done = False
            while not all_bursts_done:
                # waiting for all handles to be done with
                # bursting
                time.sleep(self._wait)
                with self._lock:
                    all_bursts_done = all(self._burst_done)
            with self._lock:
                # all handles finished bursting,
                # reset
                self._burst_done = [False] * self._size
                self._nb_bursts = None

    def stop(self) -> None:
        """
        Stopping all threads
        """
        if self._size > 1:
            self._running = False
            for thread in self._threads:
                thread.join()

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
