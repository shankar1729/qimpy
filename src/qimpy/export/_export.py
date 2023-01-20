from __future__ import annotations
import qimpy as qp
from typing import Protocol, Optional, Union


class Exporter(Protocol):
    """Class requirements to use as a geometry action."""

    def export(self, system: qp.System) -> None:
        ...


class Export(qp.TreeNode):
    """Export data for other codes."""

    bgw: qp.export.BGW
    exporters: list[Exporter]

    def __init__(
        self,
        *,
        system: qp.System,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        bgw: Optional[Union[dict, qp.export.BGW]] = None,
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
            self.add_child("bgw", qp.export.BGW, bgw, checkpoint_in, system=system)
            self.exporters.append(self.bgw)

    def __call__(self, system: qp.System):
        """Run selected geometry action."""
        for exporter in self.exporters:
            exporter.export(system)
