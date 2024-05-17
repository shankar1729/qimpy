from __future__ import annotations
from typing import Sequence, Callable, Optional, Union, Protocol

import torch
import numpy as np

from qimpy import log, rc
from qimpy.profiler import StopWatch
from qimpy.io import Checkpoint, CheckpointPath, Unit, InvalidInputException
from qimpy.mpi import ProcessGrid
from .. import Material, fermi
from . import PackedHermitian, RelaxationTime, Lindblad, Light, PulseB


class DynamicsTerm(Protocol):
    """Definition of a coherent/incoherent term within AbInitio's rho_dot."""

    constant_params: dict[str, torch.Tensor]  #: Constant values of parameters

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        """Initialize spatially-dependent fields / parameter sweeps per patch.
        Implement the part of `Material.initialize_fields` for current term.
        The `params` account for `constant_params` first and then override
        by any specified spatial dependent / parameter sweep values."""

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """Compute drho/dt given rho, with input and output in Schrodinger picture.
        Only compute one half of the result; hermitian conjugate is added later."""


class AbInitio(Material):
    """Ab initio material specification."""

    T: float
    mu: float
    rotation: torch.Tensor
    P: torch.Tensor  #: Momentum matrix elements
    S: Optional[torch.Tensor]  #: Spin matrix elements
    L: Optional[torch.Tensor]  #: Angular momentum matrix elements
    B: Optional[torch.Tensor]  #: Constant applied external field
    evecs: Optional[torch.Tensor]  #: Unitary rotations w.r.t data due to B, if any
    scattering: DynamicsTerm  #: scattering functional
    light: Light  #: light-matter interactions
    pulseB: PulseB  #: magnetic field pulses
    dynamics_terms: dict[str, DynamicsTerm]  #: all active drho/dt contributions

    def __init__(
        self,
        *,
        fname: str,
        T: float,
        mu: float = 0.0,
        rotation: Sequence[Sequence[float]] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        orbital_zeeman: Optional[bool] = None,
        B: Optional[Sequence[float]] = None,
        relaxation_time: Optional[Union[RelaxationTime, dict]] = None,
        lindblad: Optional[Union[Lindblad, dict]] = None,
        light: Optional[Union[Light, dict]] = None,
        pulseB: Optional[Union[PulseB, dict]] = None,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        write_observables: list[str] = ["q", "S"],
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        fname
            :yaml:`File name to load materials data from.`
        T
            :yaml:`Temperature.`
        mu
            :yaml:`Chemical potential in equilbrium.`
        rotation
            :yaml:`3 x 3 rotation matrix from material to simulation frame.`
        orbital_zeeman
            :yaml:`Whether to include L matrix elements in Zeeman coupling.`
            The default None amounts to using L if available in the data.
        relaxation_time
            :yaml:`Relaxation-time approximation to scattering.`
            Multiple scattering types specified will all contirbute independently.
        lindblad
            :yaml:`Ab-initio lindblad scattering.`
            Multiple scattering types specified will all contirbute independently.
        light
            :yaml:`Light-matter interaction (coherent / Lindblad).`
        pulseB
            :yaml:`Magnetic field pulses.`
        """
        self.comm = process_grid.get_comm("k")
        self.mu = mu
        self.rotation = torch.tensor(rotation, device=rc.device)
        self.write_observables = write_observables
        watch = StopWatch("Dynamics.read_checkpoint")
        with Checkpoint(fname) as data_file:
            attrs = data_file.attrs
            spinorial = bool(attrs["spinorial"])
            haveL = bool(attrs["haveL"])
            if orbital_zeeman is None:
                useL = haveL
            else:
                useL = orbital_zeeman
                if useL and not haveL:
                    raise InvalidInputException(
                        f"L not available in {fname} for orbital-zeeman"
                    )
            if T > (Tmax := float(attrs["Tmax"])) * (1 + 1e-6):
                raise InvalidInputException(f"{T = } exceeds {Tmax = }")
            self.T = T
            wk = 1 / float(attrs["nkTot"])
            nk, n_bands = data_file["E"].shape
            log.info(f"Initializing AbInitio material with {nk = } and {n_bands = }")
            super().__init__(
                wk=wk,
                nk=nk,
                n_bands=n_bands,
                n_dim=3,
                checkpoint_in=checkpoint_in,
                process_grid=process_grid,
            )

            self.k[:] = self.read_scalars(data_file, "k")
            self.E[:] = self.read_scalars(data_file, "E")
            self.P = self.read_vectors(data_file, "P")
            self.S = self.read_vectors(data_file, "S") if spinorial else None
            self.L = self.read_vectors(data_file, "L") if useL else None
            watch.stop()

            # Apply constant magnetic field, if any:
            if B is None:
                self.B = None
                self.evecs = None
            else:
                self.B = torch.tensor(B, device=rc.device)
                assert self.B.shape == (3,)
                H0 = torch.diag_embed(self.E) + self.zeemanH(self.B)
                self.E[:], self.evecs = torch.linalg.eigh(H0)
                self.apply_evecs(self.P)
                self.apply_evecs(self.S)  # skips if None automatically
                self.apply_evecs(self.L)  # skips if None automatically

            self.v = torch.einsum("kibb->kbi", self.P).real
            self.eye_bands = torch.eye(n_bands, device=rc.device)
            self.packed_hermitian = PackedHermitian(n_bands)

            # Zeroth order Hamiltonian and density matrix:
            H0 = torch.diag_embed(self.E.to(torch.complex128))[None]
            self.rho0, _, _ = self.rho_fermi(H0, self.mu)

            # Initialize optional terms in the dynamics
            self.dynamics_terms = {}

            if relaxation_time is not None:
                self.add_child(
                    "relaxation_time",
                    RelaxationTime,
                    relaxation_time,
                    checkpoint_in,
                    ab_initio=self,
                )
                self.dynamics_terms["relaxation_time"] = self.relaxation_time

            if lindblad is not None:
                self.add_child(
                    "lindblad",
                    Lindblad,
                    lindblad,
                    checkpoint_in,
                    ab_initio=self,
                    data_file=data_file,
                )
                self.dynamics_terms["lindblad"] = self.lindblad

            if light is not None:
                self.add_child("light", Light, light, checkpoint_in, ab_initio=self)
                self.dynamics_terms["light"] = self.light

            if pulseB is not None:
                self.add_child("pulseB", PulseB, pulseB, checkpoint_in, ab_initio=self)
                self.dynamics_terms["pulseB"] = self.pulseB

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        self.initialize_fields_local(rho, patch_id, **params)

    def initialize_fields_local(
        self,
        rho: torch.Tensor,
        patch_id: int,
        *,
        pumpB: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        if pumpB is not None:
            H0 = torch.diag_embed(self.E) + self.zeemanH(pumpB)
            rho[:] = self.rho_fermi(H0, self.mu)[0].flatten(-3, -1)

        for term_name, dynamics_term in self.dynamics_terms.items():
            params = dict(**dynamics_term.constant_params)  # default constant values
            prefix = f"{term_name}_"
            # Override with any varying versions:
            for key, value in kwargs.items():
                if key.startswith(prefix):
                    params[key.replace(prefix, "")] = value
            dynamics_term.initialize_fields(params, patch_id)

    def read_scalars(self, data_file: Checkpoint, name: str) -> torch.Tensor:
        """Read quantities that don't transform with rotations from data_file."""
        dset = data_file[name]
        offset = (self.k_division.i_start,) + (0,) * (len(dset.shape) - 1)
        size = (self.nk_mine,) + dset.shape[1:]
        return data_file.read_slice(dset, offset, size)

    def read_vectors(self, data_file: Checkpoint, name: str) -> torch.Tensor:
        """Read quantities that transform as a vector with rotations from data_file.
        The second index is assumed to be the Cartesian index."""
        dset = data_file[name]
        offset = (self.k_division.i_start,) + (0,) * (len(dset.shape) - 1)
        size = (self.nk_mine,) + dset.shape[1:]
        result = data_file.read_slice(dset, offset, size)
        return torch.einsum(
            "ij, kj... -> ki...", self.rotation.to(result.dtype), result
        )

    def apply_evecs(self, M: Optional[torch.Tensor]) -> None:
        """Apply transformation by `evecs` to final two band dimensions.
        For convenience, handles optional tensors = None correctly."""
        if M is not None:
            M[:] = torch.einsum(
                "kba, k...bc, kcd -> k...ad", self.evecs.conj(), M, self.evecs
            )

    def schrodingerV(self, t: float) -> torch.Tensor:
        """Compute unitary rotations from interaction to Schrodinger picture."""
        phase = torch.exp((-1j * t) * self.E)
        return torch.einsum("ka, kb -> kab", phase, phase.conj())

    def zeemanH(self, B: torch.Tensor) -> torch.Tensor:
        """Get Zeeman Hamiltonian due to specified external magnetic fields."""
        g_e = Unit.MAP["g_e"]  # spin gyromagnetic ratio magnitude
        muB_B = (B * Unit.MAP["mu_B"]).to(self.S.dtype)
        H = torch.einsum("...i, kiab -> ...kab", muB_B * g_e * 0.5, self.S)
        if self.L is not None:
            H += torch.einsum("...i, kiab -> ...kab", muB_B, self.L)
        return H

    def rho_fermi(
        self, H: torch.Tensor, mu: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the equilibrium density matrix corresponding to `H`.
        Also return the energies and eigenvectors of `H`."""
        E, V = torch.linalg.eigh(H)
        f = fermi(E, mu, self.T)
        rho = torch.einsum("...ab, ...b, ...cb -> ...ac", V, f, V.conj())
        return self.packed_hermitian.pack(rho), E, V

    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return Contactor(self, n, **kwargs)

    def get_reflector(
        self, n: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:  # absorbing boundary
        return torch.zeros_like

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """Overall drho/dt in interaction picture.
        Input and output rho are in packed (real) form."""
        if not self.dynamics_terms:
            return torch.zeros_like(rho)
        watch = StopWatch("AbInitio.rho_dot_pre")
        rho = rho.unflatten(-1, (self.nk_mine, self.n_bands, self.n_bands))

        # Switch to Schrodinger picture for scattering / coherent evolution:
        ph = self.packed_hermitian
        phase = self.schrodingerV(t)
        rho_I = ph.unpack(rho)  # interaction picture, unpacked to complex
        rho_S = rho_I * phase
        watch.stop()

        # Compute rho_dot (upto an overall +h.c.) in Schrodinger picture:
        rho_dot_S = torch.zeros_like(rho_S)
        for dynamics_term in self.dynamics_terms.values():
            rho_dot_S += dynamics_term.rho_dot(rho_S, t, patch_id)

        # Convert result back to interaction picture:
        watch = StopWatch("AbInitio.rho_dot_post")
        rho_dot_I = rho_dot_S * phase.conj()
        rho_dot_I += rho_dot_I.conj().swapaxes(-1, -2)  # + h.c.
        result = ph.pack(rho_dot_I).flatten(-3, -1)
        watch.stop()
        return result

    def get_observable_names(self) -> list[str]:
        observable_names = []
        if "q" in self.write_observables:
            observable_names.append("q")
        if "S" in self.write_observables:
            observable_names += ["Sx", "Sy", "Sz"]
        if "J" in self.write_observables:
            observable_names += ["Jx", "Jy", "Jz"]
        return observable_names  # charge, components of spin and current operator

    def get_observables(self, t: float) -> torch.Tensor:
        orbs_tuple = ()
        if "q" in self.write_observables:
            Nkbb_mine = np.prod(self.rho0.shape)
            q = torch.ones((1, Nkbb_mine), device=rc.device)  # charge observable
            orbs_tuple += (q,)
        if "S" in self.write_observables or "J" in self.write_observables:
            ph = self.packed_hermitian
            phase = self.schrodingerV(t)
            weight = torch.ones(self.n_bands, self.n_bands, device=rc.device) * 2.0
        if "S" in self.write_observables:
            S_obs = self.S.swapaxes(0, 1)
            S_obs = S_obs * phase[None, :].conj()  # complex conjugate then phase of rho
            S_obs_packed = ph.pack(S_obs)  # packed to real
            # Multiply weight of 2 to off-diagonal only:
            S_obs_packed *= weight.fill_diagonal_(1.0)[None, None, :]
            orbs_tuple += (S_obs_packed.flatten(1, 3),)
        if "J" in self.write_observables:
            J_obs = -self.P.swapaxes(0, 1)*phase[None, :].conj()
            J_obs_packed = ph.pack(J_obs)
            J_obs_packed *= weight.fill_diagonal_(1.0)[None, None, :]
            orbs_tuple += (J_obs_packed.flatten(1, 3),)
        return torch.cat(orbs_tuple, dim=0)


class Contactor:
    """Contact with fixed chemical potential and magnetic field."""

    ab_initio: AbInitio  #: Corresponding AbInitio instance
    rho0_S: torch.Tensor  #: Contact distribution fixed in Schrodinger picture

    def __init__(
        self,
        ab_initio: AbInitio,
        n: torch.Tensor,
        *,
        dmu: float = 0.0,
        Bfield: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.ab_initio = ab_initio
        # Zeroth order Hamiltonian including constant ext field:
        H0 = torch.diag_embed(ab_initio.E) + ab_initio.zeemanH(
            torch.tensor([Bfield]).to(rc.device)
        )
        self.rho0_S, _, _ = ab_initio.rho_fermi(H0, ab_initio.mu + dmu)

    def __call__(self, t: float) -> torch.Tensor:
        """Return interaction-picture contact distribution at time `t`."""
        ab_initio = self.ab_initio
        ph = ab_initio.packed_hermitian
        phase = ab_initio.schrodingerV(t)
        rho0_I = ph.pack(ph.unpack(self.rho0_S) * phase.conj())
        return torch.flatten(rho0_I)
