from __future__ import annotations


class InvalidInputException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CheckpointOverrideException(InvalidInputException):
    def __init__(self, var_name: str) -> None:
        super().__init__(
            f"Cannot override parameter '{var_name}' when reading from checkpoint"
        )
