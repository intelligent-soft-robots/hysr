import typing, pathlib, inspect
import pam_models
import pam_interface
import pam_mujoco


def _get_default(f: typing.Callable[..., typing.Any], arg_name: str) -> typing.Any:
    # e.g.
    # def f(a: str="example"): pass
    # _get_default(f,"a") # returns "example.
    return inspect.signature(f).parameters[arg_name].default


class Defaults:

    # scene
    position_robot: typing.Tuple[float, float, float] = _get_default(
        pam_mujoco.MujocoRobot.__init__, "position"
    )
    orientation_robot: str = _get_default(
        pam_mujoco.MujocoRobot.__init__, "orientation"
    )
    position_table: typing.Tuple[float, float, float] = _get_default(
        pam_mujoco.MujocoTable.__init__, "position"
    )
    orientation_table: str = _get_default(
        pam_mujoco.MujocoTable.__init__, "orientation"
    )

    # mujoco
    mujoco_period: float = 0.002  # seconds

    # pam robot config file
    pam_config: typing.Dict[pam_mujoco.RobotType, typing.Dict["str", pathlib.Path]] = {}

    pam_config[pam_mujoco.RobotType.PAMY1] = {
        "real": pathlib.Path(pam_interface.Pamy1DefaultConfiguration.get_path(False)),
        "sim": pathlib.Path(pam_interface.Pamy1DefaultConfiguration.get_path(True)),
    }

    pam_config[pam_mujoco.RobotType.PAMY2] = {
        "real": pathlib.Path(pam_interface.Pamy2DefaultConfiguration.get_path(False)),
        "sim": pathlib.Path(pam_interface.Pamy2DefaultConfiguration.get_path(True)),
    }

    muscle_model = pam_models.get_default_config_path()
