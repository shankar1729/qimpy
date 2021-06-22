from typing import Union, TypeVar, Type


ClassType = TypeVar('ClassType')


def construct(Class: Type,
              params: Union[ClassType, dict, None],
              object_name: str, **kwargs) -> ClassType:
    """Construct object in QimPy heirarchy.
    Specifically, construct object of type `Class` from `params` and `kwargs`
    if `params` is a dict, and just from `kwargs` if `params` is None.
    Any hyphens in keys within `params` are replaced with _ for convenience.
    Otherwise check that `params` is already of type `Class`, and if not,
    raise an error clearly stating what all types `object_name` can be."""

    # Try all the valid possibilities:
    if isinstance(params, dict):
        return Class(**kwargs, **dict_input_cleanup(params))
    if params is None:
        return Class(**kwargs)
    if isinstance(params, Class):
        return params

    # Report error with canonicalized class name:
    module = Class.__module__
    module_elems = ([] if module is None else (
        [elem for elem in module.split('.')
         if not elem.startswith('_')]))  # drop internal module names
    module_elems.append(Class.__qualname__)
    class_name = '.'.join(module_elems)
    raise TypeError(object_name + ' must be dict or ' + class_name)


def dict_input_cleanup(params: dict) -> dict:
    """Clean-up dict for use in constructors.
    This is required eg. for dicts from YAML to make sure keys are compatible
    with passing as keyword-only arguments to constructors. Most importantly,
    replace hyphens (which look nicer) in all keys to underscores internally,
    so that they become valid identifiers within the code"""
    return dict((k.replace('-', '_'), v) for k, v in params.items())
