from __future__ import annotations
import qimpy as qp
from typing import Union, TypeVar, Type, final


ClassType = TypeVar("ClassType")
TreeNodeType = TypeVar("TreeNodeType", bound="TreeNode")


class TreeNode:
    """Base class of objects in tree for construction and checkpointing.
    Provides functionality to set-up tree heirarchy based on input dicts,
    such as from YAML files, and to output to checkpoints such as in HDF5
    files preserving the same tree structure."""

    child_names: list[str]  #: Names of attributes with child objects.
    _attr_version_map: dict[str, str]  #: Version of children having variants (if any)

    def __init__(self, **kwargs):
        self.child_names = []

    @final
    def save_checkpoint(
        self, cp_path: qp.utils.CpPath, context: qp.utils.CpContext
    ) -> None:
        """Save `self` and all children in hierarchy to `cp_path`.
        Here, `context` helps identify why/when this checkpoint is being called,
        e.g. at a geometry step, or at the end of the simulation.
        Override `_save_checkpoint` to implement the save functionality."""
        if cp_path.checkpoint is not None:
            # Mark selected variants of children, if required:
            if hasattr(self, "_attr_version_map"):
                attrs = cp_path.attrs
                for attr_name, attr_version_name in self._attr_version_map.items():
                    attrs[attr_name] = attr_version_name
            # Save quantities in self:
            saved = self._save_checkpoint(cp_path, context)
            if saved:
                qp.log.info(f'  {cp_path.path} <- {", ".join(saved)}')
            # Recur down the hierarchy:
            for child_name in self.child_names:
                getattr(self, child_name).save_checkpoint(
                    cp_path.relative(child_name), context
                )

    def _save_checkpoint(
        self, cp_path: qp.utils.CpPath, context: qp.utils.CpContext
    ) -> list[str]:
        """Override to save required quantities to `cp_path`.
        Return names of objects saved (for logging)."""
        return []

    def add_child(
        self,
        attr_name: str,
        cls: Type[TreeNodeType],
        params: Union[TreeNodeType, dict, str, None],
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
        For such cases, use `add_child_one_of` instead, which wraps `add_child`
        and handles the selection of which version of the child to use.

        Finally, this routine supports a special case when `params` is str.
        In this case, `params` becomes a `dict` mapping that str to `{}`.
        This is a useful shortcut for a child which has one of many sub-objects.
        The str specifies the name of the sub-object with default parameters.
        The full dict version must be used instead to specify non-default values.
        Typically, the child will use `add_child_one_of` for its initialization.
        This is convenient to simplify syntax for geometry, thermostat etc.
        """
        if params is None:
            params = {}  # Logic below can focus on dict vs cls now.

        if isinstance(params, str):
            params = {params: {}}  # Shortcut for sub-object with default parameters.

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

    def add_child_one_of(
        self,
        attr_name: str,
        checkpoint_in: qp.utils.CpPath,
        *args: ChildOptions,
        have_default: bool,
    ) -> None:
        """Invoke `add_child` on one of several child options in `args`.
        At most one of the child options should have a `params` that is not None.
        If `have_default`, create child with the first class and default `params`.
        Otherwise, at least one of the child options should have a non-None `params`.

        If loading from `checkpoint_in`, this will search for attribute named
        `attr_name` in the checkpoint and use that to select the child if none
        are specified.  If parameters for a different child are specified,
        that takes precedence and checkpoint_in will be suppressed for that
        child's initialization (as the data within would be incompatible).
        """
        # Check checkpoint for child version it contains if any:
        attr_version_name_checkpoint = (
            checkpoint_in.attrs[attr_name]
            if (checkpoint_in and (attr_name in checkpoint_in.attrs))
            else ""
        )

        # Check argument list:
        arg_options = [arg for arg in args if (arg.params is not None)]
        if len(arg_options) > 1:
            arg_option_names = ", ".join(arg.attr_version_name for arg in arg_options)
            raise ValueError(f"Cannot use more than one of {arg_option_names}")
        if not (arg_options or have_default or attr_version_name_checkpoint):
            arg_names = ", ".join(arg.attr_version_name for arg in args)
            raise ValueError(f"At least one of {arg_names} must be specified")

        # Determine child based on arguments, checkpoint or default:
        if arg_options:
            arg_sel = arg_options[0]  # parameters explicitly specified
        elif attr_version_name_checkpoint:
            for arg in args:
                if arg.attr_version_name == attr_version_name_checkpoint:
                    arg_sel = arg  # version selected by checkpoint
                    break
        else:
            arg_sel = args[0]  # default

        # Remember selected variant for writing to checkpoint:
        if not hasattr(self, "_attr_version_map"):
            self._attr_version_map = {}
        self._attr_version_map[attr_name] = arg_sel.attr_version_name

        # Prevent loading data from inconsistent checkpoint:
        if arg_sel.attr_version_name != attr_version_name_checkpoint:
            checkpoint_in = qp.utils.CpPath()

        self.add_child(
            attr_name,
            arg_sel.cls,
            arg_sel.params,
            checkpoint_in,
            arg_sel.attr_version_name,
            **arg_sel.kwargs,
        )

    class ChildOptions:
        """Arguments to `qimpy.TreeNode.add_child`.
        Used to specify multiple option lists in `qimpy.TreeNode.add_child_one_of`.
        """

        def __init__(
            self,
            attr_version_name: str,
            cls: Type[TreeNodeType],
            params: Union[TreeNodeType, dict, None],
            **kwargs,
        ) -> None:
            self.attr_version_name = attr_version_name
            self.cls = cls
            self.params = params
            self.kwargs = kwargs
