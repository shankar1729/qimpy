"""YAML wrappers handling includes and environment substitution."""
__all__ = ['load', 'dump']

import qimpy as qp
import yaml
import os
from typing import Tuple


def load(filename: str, already_included: tuple = tuple()) -> dict:
    """Load input from `filename` in YAML format to a nested dict.
    Handles environment substitution and processes `include` keys.
    Keep track of `already_included` filenames to prevent cyclic includes,
    when recursively processing include directives."""
    with open(filename) as f:
        result = yaml.safe_load(os.path.expandvars(f.read()))
    return _process_includes(result, already_included + (filename,))


def dump(d: dict) -> str:
    """Convert nested dictionary to YAML-format string."""
    return yaml.dump(d, default_flow_style=None)


def _process_includes(d: dict, already_included: tuple) -> dict:
    """Recursively process `include` directives in nested dictionary."""
    # Process any includes in inner dictionaries recursively:
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _process_includes(value, already_included)
    # Process include at current level:
    include_names = d.pop('include', [])
    if include_names:
        if isinstance(include_names, str):
            include_names = [include_names]  # convert single str to List[str]
        d_list = []
        for include_name in include_names:
            if include_name in already_included:
                raise RecursionError('Cyclic include '
                                     f'{" > ".join(already_included)}'
                                     f' > {include_name}')
            d_list.append(load(include_name, already_included))
        d_list.append(d)  # current dict is last (highest priority)
        d = qp.utils.dict.merge(d_list)
    return d
