"""Utilities to manipulate dictionaries used as input to constructors."""
__all__ = ['key_cleanup', 'flatten', 'unflatten', 'merge']


def key_cleanup(params: dict) -> dict:
    """Clean-up dictionary keys for use in constructors.
    This is required eg. for dicts from YAML to make sure keys are compatible
    with passing as keyword-only arguments to constructors. Currently, this
    replaces hyphens (which look nicer) in keys to underscores internally,
    so that they become valid identifiers within the code."""
    return dict((k.replace('-', '_'), v) for k, v in params.items())
