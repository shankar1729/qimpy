from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy import log, rc, TreeNode
from qimpy.io import (
    CheckpointPath,
    CheckpointContext,
    fmt,
    cast_tensor,
    TensorCompatible,
    InvalidInputException,
    check_only_one_specified,
)
from ._lattice_systems import get_Rbasis


class Lattice(TreeNode):
    """Real and reciprocal space lattice vectors"""

    Rbasis: torch.Tensor  #: Real-space lattice vectors (in columns)
    Gbasis: torch.Tensor  #: Reciprocal-space lattice vectors (in columns)
    volume: float  #: Unit cell volume

    # Gradient / stress:
    compute_stress: bool  #: Whether to compute and report stress
    grad: torch.Tensor  #: Lattice gradient of energy := dE/dRbasis @ Rbasis.T
    _requires_grad: bool  #: Internal flag to control collection of lattice gradients
    strain_rate: Optional[torch.Tensor]  #: Strain rate (for lattice-movable dynamics)

    movable: bool  #: Whether lattice can be moved in geometry relaxation / dynamics
    move_scale: torch.Tensor  #: Scale factors to precondition / constrain lattice move

    periodic: tuple[bool, bool, bool]  #: Whether each direction is periodic
    center: torch.Tensor  #: Center (fractional coords) for non-periodic directions

    def __init__(
        self,
        *,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        system: Optional[dict] = None,
        vectors: Optional[TensorCompatible] = None,
        Rbasis: Optional[TensorCompatible] = None,
        scale: TensorCompatible = 1.0,
        compute_stress: bool = False,
        movable: bool = False,
        move_scale: TensorCompatible = (1.0, 1.0, 1.0),
        periodic: tuple[bool, bool, bool] = (True, True, True),
        center: TensorCompatible = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialize from lattice vectors or lengths and angles.
        Either specify a lattice `system` with any required lengths, angles and
        modifications, or a set of `vectors`, or as the basis matrix `Rbasis`.
        Exactly one among `system`, `vectors` and `Rbasis` must be specified.

        Note that `vectors` amounts to specifying the vectors in rows in a way
        that may be common in other codes/interfaces, while `Rbasis` amounts to
        specifying them in columns, in the form they are stored internally.

        Optionally, `scale` lattice vectors by a single or separate factors.

        Parameters
        ----------
        system
            :yaml:`Specify crystal system and geometry parameters.`
            This must be a dictionary matching one of the following patterns
            (denoted using YAML input-file syntax, and where [ ] indicates optional):

            .. code-block:: text

                system:
                  name: cubic  # all angles are 90 degrees
                  a: <value>  # each lattice vector length in bohrs
                  [modification: body-centered | face-centered]

                system:
                  name: tetragonal  # all angles are 90 degrees
                  a: <value>  # first two lattice vector lengths in bohrs
                  c: <value>  # third lattice vector length in bohrs
                  [modification: body-centered]

                system:
                  name: orthorhombic  # all angles are 90 degrees
                  a: <value>  # first lattice vector length in bohrs
                  b: <value>  # second lattice vector length in bohrs
                  c: <value>  # third lattice vector length in bohrs
                  [modification: body-centered | face-centered | base-centered]

                system:
                  name: hexagonal  # angles are 90, 90 and 120 degrees
                  a: <value>  # first two lattice vector lengths in bohrs
                  c: <value>  # third lattice vector length in bohrs

                system:
                  name: rhombohedral
                  a: <value>  # each lattice vector length in bohrs
                  alpha: <value>  # in radians, all angles equal

                system:
                  name: monoclinic
                  a: <value>  # first lattice vector length in bohrs
                  b: <value>  # second lattice vector length in bohrs
                  c: <value>  # third lattice vector length in bohrs
                  beta: <value>  # angle between a and c in radians, rest 90 degrees
                  [modification: base-centered]

                system:
                  name: triclinic
                  a: <value>  # first lattice vector length in bohrs
                  b: <value>  # second lattice vector length in bohrs
                  c: <value>  # third lattice vector length in bohrs
                  alpha: <value>  # angle between b and c in radians
                  beta: <value>  # angle between a and c in radians
                  gamma: <value>  # angle between a and b in radians

            Note that all lengths are in bohrs and angles are in radians.
            In the input file, use units like Angstrom or nm explicitly for lengths,
            and deg explicitly for angles, if needed.
            If the optional modification is unspecified or None (null in yaml),
            the unmodified Bravais lattice is selected.

        vectors
            :yaml:`Three lattice vectors, each with (x, y, z) in bohrs.`
            The input is essentially a 3 x 3 matrix, with the vectors in rows.
        Rbasis
            :yaml:`Real-space basis vectors in columns.`
            Overall, the 3 x 3 transformation from fractional to Cartesian coordinates.
        scale
            :yaml:`Scale factor for lattice vectors.` Either a single number
            that uniformly scales all lattice vectors or separate factor
            :math:`[s_1, s_2, s_3]` for each lattice vector.
        compute_stress
            :yaml:`Whether to compute and report stress.`
            Enable to report stress regardless of whether lattice is `movable`.
            (Stresses are always computed when lattice is `movable`.)
        movable
            :yaml:`Whether to move lattice during geometry relaxation / dynamics.`
        move_scale
            :yaml:`Scale factor for moving each lattice vector.`
            Set to zero for some directions to constrain lattice relaxation
            or dynamics. Can also adjust the magnitude to precondition lattice
            motion relative to the ions (internal coordinates).
        periodic
            :yaml:`Whether each lattice direction is periodic.`
            Set to False for some directions for lower-dimensional / no periodicity.
        center
            :yaml:`Center of cell for periodicity break along non-periodic directions.`
            In fractional coordinates, and values along periodic directions are
            irrelevant.
        """
        super().__init__()
        log.info("\n--- Initializing Lattice ---")

        # Get unscaled lattice vectors:
        check_only_one_specified(system=system, vectors=vectors, Rbasis=Rbasis)
        if Rbasis is not None:
            self.Rbasis = cast_tensor(Rbasis)
        elif system is not None:
            self.Rbasis = get_Rbasis(**system).to(rc.device)
        else:
            assert vectors is not None
            self.Rbasis = cast_tensor(vectors).T
            if self.Rbasis.shape != (3, 3):
                raise InvalidInputException("vectors must be a 3 x 3 matrix")

        # Apply scale:
        scale_vector = cast_tensor(scale).flatten()
        assert len(scale_vector) in (1, 3)
        self.Rbasis = scale_vector[None, :] * self.Rbasis

        self.movable = movable
        self.compute_stress = compute_stress or self.movable
        self.periodic = periodic
        self.center = cast_tensor(center)
        self.move_scale = cast_tensor(move_scale)
        if checkpoint_in:
            self.strain_rate = checkpoint_in.read_optional("strain_rate")
            stress = checkpoint_in.read_optional("stress")  # converted to grad below
        else:
            self.strain_rate = None
            stress = None

        # Compute dependent quantities:
        self.update(self.Rbasis, report_change=False)
        self.report(report_grad=False)
        check_perpendicular(self.Rbasis, self.periodic)
        self.requires_grad_(False, clear=True)  # initialize gradient
        if stress is not None:
            self.grad = self.volume * stress

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["Rbasis"] = self.Rbasis.to(rc.cpu)
        attrs["compute_stress"] = self.compute_stress
        attrs["movable"] = self.movable
        attrs["periodic"] = self.periodic
        attrs["center"] = self.center.to(rc.cpu)
        attrs["move_scale"] = self.move_scale.to(rc.cpu)
        saved_list = list(attrs.keys())
        if self.strain_rate is not None:
            saved_list.append(cp_path.write("strain_rate", self.strain_rate))
        if self.compute_stress:
            saved_list.append(cp_path.write("stress", self.stress.detach()))
        return saved_list

    def update(
        self,
        Rbasis: torch.Tensor,
        report_change: bool = True,
        center: Optional[torch.Tensor] = None,
    ) -> None:
        """Update lattice vectors and dependent quantities.
        If `report_change` is True, report the relative change of lattice and volume.
        """
        Gbasis = (2 * np.pi) * torch.linalg.inv(Rbasis.T)
        volume = abs(torch.linalg.det(Rbasis).item())
        if report_change:
            change_Rbasis = torch.linalg.norm(Rbasis - self.Rbasis) / torch.linalg.norm(
                self.Rbasis
            )
            change_volume = (volume - self.volume) / self.volume
            log.info(
                f"Relative change in Rbasis: {change_Rbasis:e}"
                f" and volume: {change_volume:e}"
            )
        self.Rbasis = Rbasis
        self.Gbasis = Gbasis
        self.volume = volume
        if center is not None:
            if report_change:
                change_center = (center - self.center).norm()
                log.info(f"RMS change in center (fractional): {change_center:e}")
            self.center = center

    def report(self, report_grad: bool) -> None:
        """Report lattice vectors, and optionally stress if `report_grad`."""
        log.info(
            f"Rbasis (real-space basis [a0] in columns):\n{fmt(self.Rbasis)}\n"
            "Gbasis (reciprocal-space basis [1/a0] in columns):\n"
            f"{fmt(self.Gbasis)}"
            f"\nUnit cell volume: {self.volume}"
            f"\nPeriodicity: {self.periodic} with center: {fmt(self.center)}"
        )
        if report_grad and self.compute_stress:
            log.info(f"Stress [Eh/a0^3]:\n{fmt(self.stress)}")

    @property
    def invRbasis(self) -> torch.Tensor:
        """Inverse of `Rbasis`."""
        return self.Gbasis.T * (0.5 / np.pi)  # reuse the already computed inverse

    @property
    def invRbasisT(self) -> torch.Tensor:
        """Inverse transpose of `Rbasis`."""
        return self.Gbasis * (0.5 / np.pi)  # reuse the already computed inverse

    @property
    def invGbasis(self) -> torch.Tensor:
        """Inverse of `Gbasis`."""
        return self.Rbasis.T * (0.5 / np.pi)  # use the existing inverse

    @property
    def invGbasisT(self) -> torch.Tensor:
        """Inverse transpose of `Gbasis`."""
        return self.Rbasis * (0.5 / np.pi)  # use the existing inverse

    @property
    def stress(self) -> torch.Tensor:
        """Cartesian stress tensor [in Eh/a0^3] (3 x 3).
        Converted from `grad`, which should already have been calculated.
        """
        return self.grad / self.volume

    @property
    def requires_grad(self) -> bool:
        """Return whether gradient with respect to this object is needed."""
        return self._requires_grad

    def requires_grad_(self, requires_grad: bool = True, clear: bool = False) -> None:
        """Set whether gradient with respect to this object is needed..
        If `clear`, also clear previous gradient / set to zero as needed.
        """
        self._requires_grad = requires_grad
        if clear:
            if requires_grad:
                self.grad = torch.zeros_like(self.Rbasis)
            else:
                self.grad = torch.full_like(self.Rbasis, np.nan)


def check_perpendicular(
    Rbasis: torch.Tensor, periodic: tuple[bool, ...], ORTHO_TOL: float = 1e-8
) -> None:
    """Check and raise error if lattice diretcions in `Rbasis` with differing
    `periodic` are not perpendicular to each other beyond tolerance `ORTHO_TOL`."""
    metric = (Rbasis.T @ Rbasis).to(rc.cpu).numpy()
    inv_lengths = 1.0 / np.sqrt(np.diag(metric))
    cos_theta = inv_lengths * metric * inv_lengths[:, None]
    violations = []
    for i in range(3):
        j = (i + 1) % 2
        if (periodic[i] != periodic[j]) and (abs(cos_theta[i, j]) > ORTHO_TOL):
            violations.append((i, j))
    if violations:
        raise ValueError(
            f"Periodic/non-periodic direction pair(s) {violations} not perpendicular"
        )
