import pam_mujoco

class PressureControl:

    def __init__(self,
                 robot_handle : pam_mujoco.MujocoHandle,
                 simulations : typing.Sequence[pam_mujoco.MujocoHandle]):

        
