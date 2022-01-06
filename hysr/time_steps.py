class TimeSteps:
    """ For managing hysr time steps

    Attributes
    ----------
    mujoco:
      mujoco time step, in seconds
    algorithm:
      RL algorithm time step, in seconds
    pressure_robot: 
      length of a control iteration of the pressure robot,
      for a mujoco accelerated simulated robot it should the same as
      the mujoco time step, for pseudo-real or real robot, it should 
      be the o80 control period
    mujoco_per_algo: int
      number of mujoco steps per algo steps

    Raises
    ------
    ValueError:
      if algorithm divided by mujoco has a reminder
    """

    def __init__(self, mujoco: float, algo: float, pressure_robot: float):

        self.mujoco = mujoco
        self.algo = algo
        self.pressure_robot = pressure_robot

        if algo % mujoco != 0:
            error = str("algo time step divised by mujoco time " "but {}%{}={}").format(
                algo, mujoco, algo % mujoco
            )
            raise ValueError(error)

        self.mujoco_per_algo: int = int(algo / mujoco)
