from typing import Generic, Union, TypeVar
from dataclasses import dataclass

T = TypeVar("T", covariant=True)

@dataclass(frozen=True)
class Default(Generic[T]):
    value: T

def unwrap_default(item: Union[T,Default[T]]) -> T:
    if isinstance(item, Default):
        return item.value
    else:
        return item

def test_default() -> bool:
    a = Default(False)
    reveal_type(a)
    return unwrap_default(a)

