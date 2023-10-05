from __future__ import annotations

class InvalidInputException(Exception):
    def __init__(self, message):
        super().__init__(message)

class CheckpointOverrideException(InvalidInputException):
    def __init__(self, value):
        super().__init__(f"Cannot override parameter {value} if reading from checkpoint")

