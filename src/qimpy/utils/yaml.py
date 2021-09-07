"""YAML wrappers handling includes and environment substitution."""
__all__ = ['load', 'dump']

import qimpy as qp
import yaml
import os


def load(filename: str) -> dict:
    """Load input from `filename` in YAML format to a nested dict.
    Handles environment substitution and processes `include` keys."""
    with open(filename) as f:
        result = qp.utils.dict.key_cleanup(
            yaml.safe_load(  # yaml parse to dict
                os.path.expandvars(f.read())))  # environment substitution
    return result
