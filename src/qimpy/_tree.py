from __future__ import annotations
import qimpy as qp
from typing import List, Union, TypeVar, Type, final


ClassType = TypeVar("ClassType")
TreeNodeType = TypeVar("TreeNodeType", bound="TreeNode")
TreeNodeType2 = TypeVar("TreeNodeType2", bound="TreeNode")


class TreeNode:
    """Base class of objects in tree for construction and checkpointing.
    Provides functionality to set-up tree heirarchy based on input dicts,
    such as from YAML files, and to output to checkpoints such as in HDF5
    files preserving the same tree structure."""

    child_names: List[str]  #: Names of attributes with child objects.

    def __init__(self, **kwargs):
        self.child_names = []

    @final
    def save_checkpoint(self, cp_path: qp.utils.CpPath) -> None:
        """Save `self` and all children in hierarchy to `cp_path`.
        Override `_save_checkpoint` to implement the save functionality."""
        if cp_path.checkpoint is not None:
            # Save quantities in self:
            saved = self._save_checkpoint(cp_path)
            if saved:
                qp.log.info(f'  {cp_path.path} <- {", ".join(saved)}')
            # Recur down the hierarchy:
            for child_name in self.child_names:
                getattr(self, child_name).save_checkpoint(cp_path.relative(child_name))

    def _save_checkpoint(self, cp_path: qp.utils.CpPath) -> List[str]:
        """Override to save required quantities to `cp_path`.
        Return names of objects saved (for logging)."""
        return []

    def add_child(
        self,
        attr_name: str,
        cls: Type[TreeNodeType],
        params: Union[TreeNodeType, dict, None],
        checkpoint_in: qp.utils.CpPath,
        attr_version_name: str = "",
        **kwargs,
    ) -> None:
        """Construct child object `self`.`attr_name` of type `cls`.
        Specifically, construct object from `params` and `kwargs`
        if `params` is a dict, and just from `kwargs` if `params` is None.
        During construction, object and its children will load data from
        `checkpoint_in`, if it contains a loaded checkpoint file.
        Any '-' in the keys of `params` are replaced with '_' for convenience.
        Otherwise check that `params` is already of type `cls`, and if not,
        raise an error clearly stating the types `attr_name` can be.

        Optionally, `attr_version_name` overrides `attr_name` used in the
        error, which may be useful when the same `attr_name` could be
        initialized by several versions eg. `kpoints` in :class:`Electrons`
        could be `k-mesh` (:class:`Kmesh`) or `k-path` (:class:`Kmesh`).
        """
        if params is None:
            params = {}  # Logic below can focus on dict vs cls now.

        # Try all the valid possibilities:
        if isinstance(params, dict):
            result = cls(
                **kwargs,
                **qp.utils.dict.key_cleanup(params),
                checkpoint_in=checkpoint_in.relative(attr_name),
            )
        elif isinstance(params, cls):
            result = params
        else:
            # Report error with canonicalized class name:
            module = cls.__module__
            module_elems = (
                []
                if module is None
                else ([elem for elem in module.split(".") if not elem.startswith("_")])
            )  # drop internal module names
            module_elems.append(cls.__qualname__)
            class_name = ".".join(module_elems)
            a_name = attr_version_name if attr_version_name else attr_name
            raise TypeError(f"{a_name} must be dict or {class_name}")

        # Add as an attribute and child in hierarchy:
        setattr(self, attr_name, result)
        self.child_names.append(attr_name)
