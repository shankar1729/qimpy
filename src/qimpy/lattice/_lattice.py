import qimpy as qp
import numpy as np
import torch
from ._lattice_systems import get_Rbasis
from typing import Optional, Union, Sequence


class Lattice(qp.TreeNode):
    """Real and reciprocal space lattice vectors"""

    __slots__ = ("Rbasis", "Gbasis", "volume")
    Rbasis: torch.Tensor  #: Real-space lattice vectors (in columns)
    Gbasis: torch.Tensor  #: Reciprocal-space lattice vectors (in columns)
    volume: float  #: Unit cell volume

    def __init__(
        self,
        *,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
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
            :math:`[s_1, s_2, s_3]` for each lattice vector. boo
        """
        super().__init__()
        qp.log.info("\n--- Initializing Lattice ---")

        # Get unscaled lattice vectors:
        if system:
            self.Rbasis = get_Rbasis(system, modification, a, b, c, alpha, beta, gamma)
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
        if scale:
            scale_vector = torch.tensor(scale).flatten()
            assert len(scale_vector) in (1, 3)
            self.Rbasis = scale_vector[None, :] * self.Rbasis
        qp.log.info(
            f"Rbasis (real-space basis in columns):\n{qp.utils.fmt(self.Rbasis)}"
        )
        self.Rbasis = self.Rbasis.to(qp.rc.device)

        # Compute reciprocal lattice vectors:
        self.Gbasis = (2 * np.pi) * torch.linalg.inv(self.Rbasis.T)
        qp.log.info(
            "Gbasis (reciprocal-space basis in columns):\n{qp.utils.fmt(self.Gbasis)}"
        )

        # Compute unit cell volume:
        self.volume = abs(torch.linalg.det(self.Rbasis).item())
        qp.log.info(f"Unit cell volume: {self.volume}")

    def update(self, Rbasis: torch.Tensor) -> None:
        """Update lattice vectors and dependent quantities"""
        Gbasis = (2 * np.pi) * torch.linalg.inv(self.Rbasis.T)
        volume = abs(torch.linalg.det(self.Rbasis).item())
        change_Rbasis = torch.linalg.norm(Rbasis - self.Rbasis) / torch.linalg.norm(
            self.Rbasis
        )
        change_volume = (volume - self.volume) / self.volume
        qp.log.info(
            f"Relative change in Rbasis: {change_Rbasis:e}"
            f" and volume: {change_volume:e}"
        )
        self.Rbasis = Rbasis
        self.Gbasis = Gbasis
        self.volume = volume
