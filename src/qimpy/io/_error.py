from __future__ import annotations


class InvalidInputException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def check_only_one_specified(**kwargs) -> None:
    """Check that exactly one of `kwargs` is not None. If not, raise an exception."""
    n_specified = sum((0 if x is None else 1) for x in kwargs.values())
    if n_specified != 1:
        names = ", ".join(kwargs.keys())
        raise InvalidInputException(f"Exactly one of {names} must be specified")


class CheckpointOverrideException(InvalidInputException):
    def __init__(self, var_name: str) -> None:
        super().__init__(
            f"Cannot override parameter '{var_name}' when reading from checkpoint"
        )
