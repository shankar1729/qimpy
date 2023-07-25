from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic


Twait = TypeVar("Twait", covariant=True)  #: generic return type of `Waitable`


class Waitable(Protocol[Twait]):
    """Generic protocol for objects with a `wait` method.
    Useful as a return type for asynchronous communication or compute functions.
    The function returns a `Waitable` object, with the actual results returned later
    by the `wait` method."""

    def wait(self) -> Twait:
        """Return the actual results of the asynchronous operation, once complete."""


@dataclass
class Waitless(Generic[Twait]):
    """Trivial (identity) `Waitable` for when result is immediately ready."""

    result: Twait

    def wait(self) -> Twait:
        return self.result
