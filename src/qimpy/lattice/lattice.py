from typing import Optional, Union, Sequence

import numpy as np
import torch

from qimpy import log, rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext, fmt
from .lattice_systems import get_Rbasis


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

    def __init__(
        self,
        *,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        system: Optional[str] = None,
        modification: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        vector1: Optional[Sequence[float]] = None,
        vector2: Optional[Sequence[float]] = None,
        vector3: Optional[Sequence[float]] = None,
        scale: Optional[Union[float, Sequence[float]]] = None,
        compute_stress: Optional[bool] = None,
        movable: Optional[bool] = None,
        move_scale: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize from lattice vectors or lengths and angles.
        Either specify a lattice `system` and optional `modification`,
        along with any corresponding required lengths (`a`, `b`, `c`)
        and angles (`alpha`, `beta`, `gamma`), or explicitly specity
        all three lattice vectors `vector1`, `vector2` and `vector3`.
        Optionally, `scale` lattice vectors by a single or separate factors.

        Parameters
        ----------
        system
            :yaml:`Specify crystal system and geometry parameters.`
            Options include:

            * cubic (specify `a`),
            * tetragonal (specify `a`, `c`)
            * orthorhombic (specify `a`, `b`, `c`)
            * hexagonal (specify `a`, `c`)
            * rhombohedral (specify `a`, `alpha`)
            * monoclinic (specify `a`, `b`, `c`, `beta`)
            * triclinic (specify `a`, `b`, `c`, `alpha`, `beta`, `gamma`)

        modification
            :yaml:`Specify modification of lattice.`
            Options include:

            * body-centered (only for orthorhombic, tetragonal or cubic)
            * face-centered (only for orthorhombic or cubic)
            * base-centered (only for monoclinic)

        a
            :yaml:`First lattice vector length in bohrs.`
        b
            :yaml:`Second lattice vector length in bohrs.`
        c
            :yaml:`Third lattice vector length in bohrs.`
        alpha
            :yaml:`Angle between b and c in degrees.`
        beta
            :yaml:`Angle between c and a in degrees.`
        gamma
            :yaml:`Angle between a and b in degrees.`
        vector1
            :yaml:`First lattice vector (x1, y1, z1) in bohrs.`
        vector2
            :yaml:`Second lattice vector (x2, y2, z2) in bohrs.`
        vector3
            :yaml:`Third lattice vector (x3, y3, z3) in bohrs.`
        scale
            :yaml:`Scale factor for lattice vectors.` Either a single number
            that uniformly scales all lattice vectors or separate factor
            :math:`[s_1, s_2, s_3]` for each lattice vector.
        compute_stress
            :yaml:`Whether to compute and report stress.`
            Enable to report stress regardless of whether lattice is `movable`.
            Defaults to False if unspecified.
            (Stresses are always computed when lattice is `movable`.)
        movable
            :yaml:`Whether to move lattice during geometry relaxation / dynamics.`
            Defaults to False if unspecified.
        move_scale
            :yaml:`Scale factor for moving each lattice vector.`
            Set to zero for some directions to constrain lattice relaxation
            or dynamics. Can also adjust the magnitude to precondition lattice
            motion relative to the ions (internal coordinates).
            Defaults to (1, 1, 1) if unspecified.
        """
        super().__init__()
        log.info("\n--- Initializing Lattice ---")

        if checkpoint_in:
            attrs = checkpoint_in.attrs
            self.compute_stress = attrs["compute_stress"]
            self.movable = attrs["movable"]
            self.move_scale = checkpoint_in.read("move_scale")
            self.strain_rate = checkpoint_in.read_optional("strain_rate")
            self.Rbasis = checkpoint_in.read("Rbasis")
            stress = checkpoint_in.read_optional("stress")  # converted to grad below
        else:
            self.compute_stress = False
            self.movable = False
            self.move_scale = torch.ones(3, device=rc.device)
            self.strain_rate = None
            stress = None

            # Get unscaled lattice vectors:
            if system:
                self.Rbasis = get_Rbasis(
                    system, modification, a, b, c, alpha, beta, gamma
                )
            else:
                # Direct specification of lattice vectors:
                def check_vectors(**kwargs):
                    for key, value in kwargs.items():
                        if value is None:
                            raise KeyError(key + " must be specified")
                        try:
                            np.array(value, dtype=float).reshape(3)
                        except ValueError:
                            raise ValueError(key + " must contain 3 numbers")

                check_vectors(vector1=vector1, vector2=vector2, vector3=vector3)
                self.Rbasis = torch.tensor([vector1, vector2, vector3]).T

            # Apply scale if needed:
            if scale and (not checkpoint_in):
                scale_vector = torch.tensor(scale).flatten()
                assert len(scale_vector) in (1, 3)
                self.Rbasis = scale_vector[None, :] * self.Rbasis

        # Compute dependent quantities:
        self.update(self.Rbasis.to(rc.device), report_change=False)
        self.report(report_grad=False)
        self.requires_grad_(False, clear=True)  # initialize gradient
        if stress is not None:
            self.grad = self.volume * stress

        # Optionally override optimization / constraints settings:
        if movable is not None:
            self.movable = movable
            self.compute_stress = self.compute_stress or movable
        if compute_stress is not None:
            self.compute_stress = compute_stress or self.movable
        if move_scale is not None:
            self.move_scale = torch.tensor(move_scale, device=rc.device)
            assert self.move_scale.shape == (3,)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["compute_stress"] = self.compute_stress
        attrs["movable"] = self.movable
        saved_list = ["compute_stress", "movable"]
        saved_list.append(cp_path.write("move_scale", self.move_scale))
        if self.strain_rate is not None:
            saved_list.append(cp_path.write("strain_rate", self.strain_rate))
        saved_list.append(cp_path.write("Rbasis", self.Rbasis))
        if self.compute_stress:
            saved_list.append(cp_path.write("stress", self.stress.detach()))
        return saved_list

    def update(self, Rbasis: torch.Tensor, report_change: bool = True) -> None:
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

    def report(self, report_grad: bool) -> None:
        """Report lattice vectors, and optionally stress if `report_grad`."""
        log.info(
            f"Rbasis (real-space basis [a0] in columns):\n{fmt(self.Rbasis)}\n"
            "Gbasis (reciprocal-space basis [1/a0] in columns):\n"
            f"{fmt(self.Gbasis)}"
            f"\nUnit cell volume: {self.volume}"
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
