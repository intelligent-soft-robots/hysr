import collections
import typing
import importlib
import inspect
from . import hysr_types


def import_packages(label: str, config: typing.Dict[str,typing.Any])->None:
    """
    config is a dictionary, which may have an "imports" key. If the related a value is a string, 
    then attempt to import a package of this name. If the value is an iterable, then attempt to import
    all packages named after then entries of the iterable. 
    If there is no key "imports", does nothing.
    """
    try:
        imports = config["imports"]
    except KeyError:
        return
    if isinstance(imports,str):
        imports = (imports,)
    if not isinstance(imports, collections.abc.Iterable):
        raise ValueError(
            "hysr control factory: the toml configuration has an unexpected value for imports (should be iterable of string)"
        )
    for import_ in imports:
        try:
            imported = importlib.import_module(str(import_))
        except ModuleNotFoundError as e:
            raise ValueError(
                f"{label}: failed to import module {import_}: " f"{e}"
            )

def _get_class(class_path: str) -> typing.Type:
    """
    class_path: something like "package.subpackage.module.class_name".
    Imports package.subpackage.module and returns the class.
    """

    # class_path is only the name of the class, which is thus expected
    # to be in global scope
    if "." not in class_path:
        try:
            class_ = globals()[class_path]
        except KeyError:
            raise ValueError(
                f"class {class_path} could not be found in the global scope"
            )

    # importing the package the class belongs to
    to_import, class_name = class_path.rsplit(".", 1)
    try:
        imported = importlib.import_module(to_import)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"failed to import {to_import} (needed to instantiate {class_path}): {e}"
        )

    # getting the class
    try:
        class_ = getattr(imported, class_name)
    except AttributeError:
        raise ValueError(
            f"class {class_name} (provided path: {class_path}) could not be found"
        )

    return class_

ExpectedSuperclass = typing.TypeVar("ExpectedSuperclass", bound=object)
def instantiate(
    factory_class: hysr_types.FactoryClass,
    expected_superclass: typing.Type[ExpectedSuperclass],
) -> ExpectedSuperclass:
    """
    Uses the information of the factory class (i.e. class name and
    arguments) to import the required package and instantiate
    the class. Also checks the class is a subclass of expected
    super class (ValueError raised if not).
    """
    class_name, args, kwargs = factory_class

    # getting the class from its path
    # e.g. MainSim from 'hysr.MainSim'
    class_ = _get_class(class_name)

    # checking the class is indeed a class
    if not inspect.isclass(class_):
        raise ValueError(f"{class_name} is not a class (it is a {type(class_name)})")

    # checking the class is a subclass
    if not issubclass(class_, expected_superclass):
        raise ValueError(
            f"{class_name} provided as the subclass to use of "
            f"{expected_superclass}, but {class_name} is not a subclass of "
            f"{expected_superclass}"
        )

    # reading the arguments requested by class_
    parameters = inspect.signature(class_.__init__).parameters
    nb_all_args = len(parameters) - 1  # -1: self argument
    valid_kwargs = {
        k: v.default for k, v in parameters.items() if v.default != inspect._empty
    }
    nb_kwargs = len(valid_kwargs)
    nb_args = nb_all_args - nb_kwargs

    # checking we have the expected number of args
    if nb_args != len(args):
        raise ValueError(
            f"the class {class_name} request {nb_args} "
            f"arguments, but {len(args)} provided"
        )

    # checking we have the expected of kwargs
    if len(kwargs) > nb_kwargs:
        raise ValueError(
            f"the class {class_name} takes at most {nb_kwargs} "
            f"key-words arguments, but {len(kwargs)} provided"
        )

    # checking all kwargs are accepted by class_
    for kwarg_name in kwargs.keys():
        valid = list(valid_kwargs.keys())
        if kwarg_name not in valid:
            raise ValueError(f"{kwarg_name} is not a known kwarg name for {class_name}")

    # all seems ok, instantiating
    try:
        instance = class_(*args, **kwargs)
    except Exception as e:
        raise Exception(f"failed to instantiate {class_name}: {e}")

    return instance


def read_factory_class(
    label: str, d: typing.Dict[str, typing.Any]
) -> hysr_types.FactoryClass:
    """
    Instantiate a FactoryClass from provided dictionary.
    The dictionary should have the key "class" (string), and optionaly
    the keys "args" and "kwargs". If 'args' and 'kwargs' are respectively
    a list and a dict, they are returned as such. Else, they are evaluated.
    Note: it is assumed that the packages required for this evaluation
    have already been imported.
    Label is an arbitrary string used in the message of raised exceptions
    """

    required_keys = ("class",)
    optional_keys = ("args", "kwargs")
    for rk in required_keys:
        if rk not in d.keys():
            raise ValueError(
                f"{label}: failed to find required key {rk} in toml configuration"
            )
    for k in d.keys():
        if k not in required_keys + optional_keys:
            raise ValueError(
                f"{label}: the toml configuration has an unexpected key: {k}"
            )
    class_ = str(d["class"])
    if "args" in d.keys():
        if isinstance(d["args"], str):
            try:
                args = eval(d["args"])
            except NameError as ne:
                raise ValueError(f"{label}: failed to evaluate args {d['args']}: {ne} ")
        else:
            args = d["args"]
        if not isinstance(args, collections.abc.Sequence):
            raise ValueError(
                f"{label}: the toml configuration has an unexpected value for: args (should be a sequence)"
            )
    else:
        args = []
    if "kwargs" in d.keys():
        if isinstance(d["kwargs"], str):
            kwargs = eval(d["kwargs"])
        else:
            kwargs = d["kwargs"]
    else:
        kwargs = {}

    return (class_, args, kwargs)
