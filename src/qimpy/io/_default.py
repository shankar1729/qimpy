from typing import Generic, Union, TypeVar
from dataclasses import dataclass


T = TypeVar("T")


@dataclass(frozen=True)
class Default(Generic[T]):
    value: T


OptionalDefault = Union[T, Default[T]]


def unwrap_default(item: OptionalDefault[T]) -> T:
    if isinstance(item, Default):
        return item.value
    else:
        return item


def test_default(param: OptionalDefault[bool] = Default(False)) -> None:
    is_default = isinstance(param, Default)
    value = unwrap_default(is_default)
    print(f"param = {value} was {'' if is_default else 'not '}specified as a default")
