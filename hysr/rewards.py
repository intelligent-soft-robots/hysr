import math
import pathlib
import typing
import toml
from . import config
from . import hysr_types
from . import hysr_control


class Rewards:
    def __init__(self) -> None:
        pass

    def compute(episode: hysr_control.Episode) -> hysr_types.MultiRewards:
        raise NotImplementedError()

    def reset(self) -> None:
        pass


def rewards_factory(factory_class: hysr_types.FactoryClass) -> Rewards:
    return config.instantiate(factory_class, Rewards)


def rewards_from_toml_content(toml_string: str) -> Rewards:

    try:
        t = toml.loads(toml_string)
    except toml.TomlDecodeError as tde:
        raise ValueError(f"rewards factory: invalid toml content: {tde}")

    config.import_packages("rewards", t)

    factory_class = config.read_factory_class("rewards_factory", t)

    return rewards_factory(factory_class)


def rewards_from_toml_file(file_path: pathlib.Path) -> Rewards:

    if not file_path.is_file():
        raise FileNotFoundError(
            f"rewards factory: failed to find toml configuration file: {file_path}"
        )

    content = file_path.read_text()

    try:
        rewards = rewards_from_toml_content(content)
    except ValueError as ve:
        raise ValueError(
            f"failed to use configuration file {file_path} to instantiate rewards: {ve}"
        )

    return rewards
