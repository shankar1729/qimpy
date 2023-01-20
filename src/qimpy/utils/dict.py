"""Utilities to manipulate dictionaries used as input to constructors."""
__all__ = ["key_cleanup", "flatten", "unflatten", "merge"]


def key_cleanup(params: dict) -> dict:
    """Clean-up dictionary keys for use in constructors.
    This is required eg. for dicts from YAML to make sure keys are compatible
    with passing as keyword-only arguments to constructors. Currently, this
    replaces hyphens (which look nicer) in keys to underscores internally,
    so that they become valid identifiers within the code."""
    return dict((k.replace("-", "_"), v) for k, v in params.items())


def flatten(d: dict, _key_prefix: tuple = tuple()) -> dict:
    """Convert nested dict `d` to a flat dict with tuple keys.
    Input `_key_prefix` is prepended to the keys of the resulting dict,
    and is used internally for recursively flattening the dict."""
    result = {}
    for key, value in d.items():
        flat_key = _key_prefix + (key,)
        if isinstance(value, dict):
            result.update(flatten(value, flat_key))
        else:
            result[flat_key] = value
    return result


def unflatten(d: dict) -> dict:
    """Unpack tuple keys in `d` to a nested dictionary.
    (Inverse of :func:`flatten`.)"""
    result: dict = {}
    for key_tuple, value in d.items():
        assert isinstance(key_tuple, tuple)
        target = result  # where to add value
        for key in key_tuple[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]  # traverse down each key in tuple
        target[key_tuple[-1]] = value
    return result


def merge(d_list: list[dict]) -> dict:
    """Merge a list of nested dictonaries `d_list`.
    The dictionaries are processed in order, with each dictionary overriding
    values associated with keys present in previous dictionaries."""
    result = {}
    for d in d_list:
        result.update(flatten(d))
    return unflatten(result)
