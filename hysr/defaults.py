import typing, pathlib


class Defaults:

    # scene
    position_robot: typing.Tuple[float, float, float] = (0.435, 0.1175, -0.0025)
    orientation_robot: str = "-1 0 0 0 -1 0"
    position_table: typing.Tuple[float, float, float] = (0.1, 0, -0.44)
    orientation_table: str = "-1 0 0 0 -1 0"

    # mujoco
    mujoco_period: float = 0.002  # seconds

    # pam robot config file
    pam_config: typing.Dict[str, pathlib.Path] = {
        "real": pathlib.Path("/opt/mpi-is/pam_interface/pam.json"),
        "sim": pathlib.Path("/opt/mpi-is/pam_interface/pam_sim.json"),
    }
    muscle_model = pathlib.Path("/opt/mpi-is/pam_models/hill.json")
