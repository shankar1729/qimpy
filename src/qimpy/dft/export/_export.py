from __future__ import annotations
from typing import Protocol, Optional, Union

from qimpy import TreeNode, dft
from qimpy.io import CheckpointPath, CheckpointContext
from . import BGW


class Exporter(Protocol):
    """Class requirements to use as a geometry action."""

    def export(self, system: dft.System) -> None:
        ...


class Export(TreeNode):
    """Export data for other codes."""

    bgw: BGW
    exporters: list[Exporter]

    def __init__(
        self,
        *,
        system: dft.System,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        bgw: Optional[Union[dict, BGW]] = None,
    ) -> None:
        """Specify one or more export formats.

        Parameters
        ----------
        bgw
            :yaml:`BerkeleyGW export.`
        """
        super().__init__()
        self.exporters = []

        if bgw is not None:
            self.add_child("bgw", BGW, bgw, checkpoint_in, system=system)
            self.exporters.append(self.bgw)

    def __call__(self, system: dft.System):
        """Run selected geometry action."""
        for exporter in self.exporters:
            exporter.export(system)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        # TODO: add attributes for each exporter
        return list(attrs.keys())
