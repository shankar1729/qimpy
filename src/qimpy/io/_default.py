from __future__ import annotations
from typing import Generic, Union, TypeVar
from dataclasses import dataclass


T = TypeVar("T")


@dataclass(frozen=True)
class Default(Generic[T]):
    """Typed default value for a function argument.
    Use as a sentinel to specify a default value, instead of None.
    This allows passing in a default value, and keeping track of whether
    the argument was explicitly passed in or a default within the function.
    """

    value: T  #: The underlying default value


WithDefault = Union[T, Default[T]]  #: Type alias for a type or its default value


def cast_default(item: WithDefault[T]) -> T:
    """Cast an optional default to retain only the value."""
    if isinstance(item, Default):
        return item.value
    else:
        return item


def test_default(param: WithDefault[bool] = Default(False)) -> None:
    is_default = isinstance(param, Default)
    value = cast_default(param)
    print(f"param = {value} was {'' if is_default else 'not '}specified as a default")
