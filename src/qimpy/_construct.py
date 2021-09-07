from __future__ import annotations
import qimpy as qp
from typing import Optional, List, Union, TypeVar, Type, NamedTuple, final


ClassType = TypeVar('ClassType')
ConstructableType = TypeVar('ConstructableType', bound='Constructable')
ConstructableType2 = TypeVar('ConstructableType2', bound='Constructable')


class ConstructOptions(NamedTuple):
    """Options passed through `__init__` of all Constructable objects."""
    rc: qp.utils.RunConfig  #: Current run configuration
    parent: Optional[qp.Constructable] = None  #: Parent in heirarchy
    attr_name: str = ''  #: Attribute name of object within parent
    checkpoint_in: Optional[qp.utils.Checkpoint] = None \
        #: If present, load data from this checkpoint file during construction


class Constructable:
    """Base class of dict-constructable and serializable objects
    in QimPy heirarchy."""
    __slots__ = ('parent', 'children', 'path', 'rc', 'checkpoint_in')
    parent: Optional[qp.Constructable]  #: Parent object in heirarchy (if any)
    children: List[qp.Constructable]  #: Child objects in heirarchy
    path: str  #: Object's absolute path in heirarchy (includes trailing /)
    rc: qp.utils.RunConfig  #: Current run configuration
    checkpoint_in: Optional[qp.utils.Checkpoint] \
        #: If present, load data from this checkpoint file during construction

    def __init__(self, co: ConstructOptions, **kwargs):
        self.rc = co.rc
        self.parent = co.parent
        self.children = []
        self.path = ('/' if (co.parent is None)
                     else (co.parent.path + co.attr_name + '/'))
        self.checkpoint_in = co.checkpoint_in

    @final
    def save_checkpoint(self, checkpoint: qp.utils.Checkpoint) -> None:
        """Save `self` and all children in heirarchy to `checkpoint`.
        Override `_save_checkpoint` to implement the save functionality."""
        # Save quantities in self:
        saved = self._save_checkpoint(checkpoint)
        if saved:
            qp.log.info(f'  {self.path} <- {", ".join(saved)}')
        # Recur down the heirarchy:
        for child in self.children:
            child.save_checkpoint(checkpoint)

    def _save_checkpoint(self, checkpoint: qp.utils.Checkpoint) -> List[str]:
        """Override to save required quantities to `self.path`
        within `checkpoint`. Return names of objects saved (for logging)."""
        return []

    def _checkpoint_has(self, object_name: str) -> bool:
        """Return whether input checkpoint exists and contains `object_name`
        at `self.path`"""
        return ((self.checkpoint_in is not None)
                and ((self.path + object_name) in self.checkpoint_in))

    def construct(self, attr_name: str, cls: Type[ConstructableType],
                  params: Union[ConstructableType, dict, None],
                  attr_version_name: str = '', **kwargs) -> None:
        """Construct child object `self`.`attr_name` of type `cls`.
        Specifically, construct object from `params` and `kwargs`
        if `params` is a dict, and just from `kwargs` if `params` is None.
        Any '-' in the keys of `params` are replaced with '_' for convenience.
        Otherwise check that `params` is already of type `cls`, and if not,
        raise an error clearly stating the types `attr_name` can be.

        Optionally, `attr_version_name` overrides `attr_name` used in the
        error, which may be useful when the same `attr_name` could be
        initialized by several versions eg. `kpoints` in :class:`Electrons`
        could be `k-mesh` (:class:`Kmesh`) or `k-path` (:class:`Kmesh`).
        """

        # Try all the valid possibilities:
        co = ConstructOptions(parent=self, attr_name=attr_name, rc=self.rc,
                              checkpoint_in=self.checkpoint_in)
        if isinstance(params, dict):
            result = cls(**kwargs, **qp.utils.dict.key_cleanup(params), co=co)
        elif params is None:
            result = cls(**kwargs, co=co)
        elif isinstance(params, cls):
            result = params
            Constructable.__init__(result, co=co)
        else:
            # Report error with canonicalized class name:
            module = cls.__module__
            module_elems = ([] if module is None else (
                [elem for elem in module.split('.')
                 if not elem.startswith('_')]))  # drop internal module names
            module_elems.append(cls.__qualname__)
            class_name = '.'.join(module_elems)
            a_name = (attr_version_name if attr_version_name else attr_name)
            raise TypeError(f'{a_name} must be dict or {class_name}')

        # Add as an attribute and child in heirarchy:
        setattr(self, attr_name, result)
        self.children.append(result)
