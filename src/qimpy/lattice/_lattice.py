import qimpy as qp
import numpy as np
import torch
from ._lattice_systems import get_Rbasis
from typing import Optional, Union, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig


class Lattice(qp.Constructable):
    """Real and reciprocal space lattice vectors"""

    __slots__ = ('rc', 'Rbasis', 'Gbasis', 'volume')
    rc: 'RunConfig'  #: Current run configuration
    Rbasis: torch.Tensor  #: Real-space lattice vectors (in columns)
    Gbasis: torch.Tensor  #: Reciprocal-space lattice vectors (in columns)
    volume: float  #: Unit cell volume

    def __init__(self, *, rc: 'RunConfig', co: qp.ConstructOptions,
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
                 scale: Optional[Union[float, Sequence[float]]] = None
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
            Specify lattice by crystal system and required geometry
            parameters. Options include:

            * cubic (specify `a`),
            * tetragonal (specify `a`, `c`)
            * orthorhombic (specify `a`, `b`, `c`)
            * hexagonal (specify `a`, `c`)
            * rhombohedral (specify `a`, `alpha`)
            * monoclinic (specify `a`, `b`, `c`, `beta`)
            * triclinic (specify `a`, `b`, `c`, `alpha`, `beta`, `gamma`)

        modification
            Specify modification of lattice:

            * body-centered (only for orthorhombic, tetragonal or cubic)
            * face-centered (only for orthorhombic or cubic)
            * base-centered (only for monoclinic)

        a
            First lattice vector length in :math:`a_0`
        b
            Second lattice vector length in :math:`a_0`
        c
            Third lattice vector length in :math:`a_0`
        alpha
            Angle between `b` and `c` in degrees
        beta
            Angle between `c` and `a` in degrees
        gamma
            Angle between `a` and `b` in degrees
        vector1
            First vector :math:`[x_1, y_1, z_1]` (in :math:`a_0`)
        vector2
            Second vector :math:`[x_2, y_2, z_2]` (in :math:`a_0`)
        vector3
            Third vector :math:`[x_3, y_3, z_3]` (in :math:`a_0`)
        scale
            Single scale factor for all lattice vectors, or separate factor
            :math:`[s_1, s_2, s_3]` for each lattice vector
        """
        super().__init__(co=co)
        self.rc = rc
        qp.log.info('\n--- Initializing Lattice ---')

        # Get unscaled lattice vectors:
        if system:
            self.Rbasis = get_Rbasis(system, modification, a, b, c,
                                     alpha, beta, gamma)
        else:
            # Direct specification of lattice vectors:
            def check_vectors(**kwargs):
                for key, value in kwargs.items():
                    if value is None:
                        raise KeyError(key + ' must be specified')
                    try:
                        v = np.array(value, dtype=float).reshape(3)
                    except ValueError:
                        raise ValueError(key + ' must contain 3 numbers')
            check_vectors(vector1=vector1, vector2=vector2, vector3=vector3)
            self.Rbasis = torch.tensor([vector1, vector2, vector3]).T

        # Apply scale if needed:
        if scale:
            scale_vector = torch.tensor(scale).flatten()
            assert(len(scale_vector) in (1, 3))
            self.Rbasis = scale_vector[None, :] * self.Rbasis
        qp.log.info('Rbasis (real-space basis in columns):\n'
                    + rc.fmt(self.Rbasis))
        self.Rbasis = self.Rbasis.to(rc.device)

        # Compute reciprocal lattice vectors:
        self.Gbasis = (2*np.pi) * torch.linalg.inv(self.Rbasis.T)
        qp.log.info('Gbasis (reciprocal-space basis in columns):\n'
                    + rc.fmt(self.Gbasis))

        # Compute unit cell volume:
        self.volume = abs(torch.linalg.det(self.Rbasis).item())
        qp.log.info(f'Unit cell volume: {self.volume}')

    def update(self, Rbasis: torch.Tensor) -> None:
        """Update lattice vectors and dependent quantities"""
        Gbasis = (2*np.pi) * torch.linalg.inv(self.Rbasis.T)
        volume = abs(torch.linalg.det(self.Rbasis).item())
        change_Rbasis = (torch.linalg.norm(Rbasis - self.Rbasis)
                         / torch.linalg.norm(self.Rbasis))
        change_volume = (volume - self.volume) / self.volume
        qp.log.info(f'Relative change in Rbasis: {change_Rbasis:e}'
                    f' and volume: {change_volume:e}')
        self.Rbasis = Rbasis
        self.Gbasis = Gbasis
        self.volume = volume
