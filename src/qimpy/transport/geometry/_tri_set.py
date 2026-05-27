from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy import TreeNode, rc, log
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid
from ..material import Material
from . import TensorList, Geometry
from ._dg2d import nodes2d, prolongation_matrix
from ._dg_mesh import load_mesh
from ._dg_mpi import (compute_partition, SpatialPartition, DistributedAdvect,
                      mpi_halo_exchange)


def _global_node_xy(mesh, N: int) -> np.ndarray:
    """Physical coordinates of the order-N DG nodes for every (global) element,
    shape (K, Np, 2). Affine map of the reference nodes by each element's
    vertices; reproducible on any rank from the global mesh alone."""
    r, s = nodes2d(N)
    VX, VY, T = mesh.VX, mesh.VY, mesh.EToV
    va, vb, vc = T[:, 0], T[:, 1], T[:, 2]
    L0 = -(r[:, None] + s[:, None]); L1 = 1 + r[:, None]; L2 = 1 + s[:, None]
    x = 0.5 * (L0 * VX[va][None] + L1 * VX[vb][None] + L2 * VX[vc][None])
    y = 0.5 * (L0 * VY[va][None] + L1 * VY[vb][None] + L2 * VY[vc][None])
    return np.ascontiguousarray(np.stack([x, y], -1).transpose(1, 0, 2))  # (K,Np,2)


def _ref_subtriangulation(N: int) -> np.ndarray:
    """Triangulation of the order-N nodal lattice into N^2 sub-triangles, used to
    render the high-order field within each element (indices into Np nodes)."""
    import matplotlib.tri as mtri
    r, s = nodes2d(N)
    return mtri.Triangulation(r, s).triangles.astype(np.int64)


class TriSet(Geometry):
    """Discontinuous-Galerkin geometry on an externally supplied triangle mesh,
    parallel over momentum and space, with mesh-native output and restart.

    The mesh (external; see _dg_mesh.load_mesh) is decomposed across the process
    grid's real-space communicator: each rank owns a subset of elements plus a
    ghost ring, runs the DG operators locally, and exchanges ghost state each RHS
    so cross-rank faces read the correct neighbor trace (orthogonal to the
    material's momentum parallelization). Observables and the restart state live
    on the DG nodes -- not a resampled grid -- so the checkpoint preserves the
    graded, high-order, boundary-conforming solution. A run checkpointed at order
    N can be continued at a higher order: the order-N nodal polynomial embeds
    exactly in the higher-order basis (prolongation_matrix).
    """

    dg: object
    order: int

    def __init__(
        self,
        *,
        material: Material,
        mesh_file: str,
        contacts: dict[str, Optional[dict]],
        order: int = 3,
        render_grid_spacing: Optional[float] = None,   # accepted, unused (kept for yaml compat)
        save_rho: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Parameters
        ----------
        mesh_file
            :yaml:`Path to an external triangle mesh (.npz) to solve on.`
        contacts
            :yaml:`Dictionary of contact names to parameters (match mesh markers).`
        order
            :yaml:`Polynomial order N of the DG basis on each triangle.`
            May exceed the order of a checkpoint being restarted from.
        save_rho
            :yaml:`Whether to also write the full nodal density at each checkpoint.`
        """
        TreeNode.__init__(self)
        self.material = material
        self.comm = process_grid.get_comm("r")        # spatial communicator
        self.order = order
        self.mesh_file = mesh_file
        self.save_rho = save_rho
        self.contacts = contacts

        mesh = load_mesh(mesh_file)
        self.mesh = mesh
        nparts = self.comm.size
        self._distributed = nparts > 1
        K = len(mesh.EToV)
        part = np.empty(K, np.int32)
        if self.comm.rank == 0:
            part[:] = compute_partition(mesh, nparts)
        self.comm.Bcast(part, root=0)
        self.part = SpatialPartition(mesh, order, part, self.comm.rank)
        if getattr(mesh, "_lattice", None) and not self._distributed:
            self.part.dg.make_periodic(mesh._lattice)
        elif getattr(mesh, "_lattice", None) and self._distributed:
            log.info("tri_set: periodic lattice ignored under spatial "
                     "decomposition (Pr>1); run periodic cases with Pr=1.")
        self.dist = DistributedAdvect(self.part, material, contacts, self.comm)
        self.dg = self.part.dg
        self.adv = self.dist.adv
        self._vx, self._vy = self.dist.vx, self.dist.vy
        self.Nk = self.dist.Nk

        coupling = getattr(self.dist, "coupling", None)
        if coupling is not None:
            # Coupled (modal) flux advects through n.A, not diagonal velocities
            # (which are ~0 here); the max characteristic speed is the spectral
            # radius of n_x Ax + n_y Ay, exposed as coupling.max_speed (= vF for
            # the Fermi-circle operator). Using the ~0 diagonal velocity would
            # make dt_max blow up and the auto-selected step diverge.
            vmax = float(coupling.max_speed)
        else:
            v = material.transport_velocity.detach().cpu().numpy()
            vmax = float(np.hypot(v[:, 0], v[:, 1]).max())
        dt_local = float(0.4 * self.dg.dt_scale / max(vmax, 1e-300))
        self.dt_max = self.comm.allreduce(dt_local, op=MPI.MIN)
        if self._distributed:
            log.info(f"tri_set: spatial decomposition over {nparts} ranks "
                     f"(this rank: {self.part.n_owned} owned + "
                     f"{self.dg.K - self.part.n_owned} ghost elements)")

        self._stash_t: list[float] = []
        self._stash_i: list[int] = []
        self._stash_obs: list[np.ndarray] = []        # per step: (K_owned, Np, n_obs)

        # --- initial state: restart from checkpoint (prolonging order if needed)
        #     or material equilibrium tiled onto the local DG nodes ---
        dtype = material.transport_velocity.dtype
        if not self._restart_density(checkpoint_in, dtype):
            Nkbb = material.rho0.numel()
            rho0_nodal = material.rho0.flatten().to(rc.device).reshape(1, 1, Nkbb) \
                * torch.ones(self.dg.Np, self.dg.K, Nkbb, device=rc.device, dtype=dtype)
            self._rho = self.adv.apply_mass(rho0_nodal)

    def _restart_density(self, checkpoint_in, dtype) -> bool:
        """Load this rank's owned density from a checkpoint, prolonging from the
        checkpoint's DG order to the current order if they differ. Returns True
        if a checkpoint state was loaded."""
        if not checkpoint_in:
            return False
        checkpoint, path = checkpoint_in.relative("dg_rho")
        if checkpoint is None or path not in checkpoint:
            return False
        N_ckpt = int(checkpoint_in.attrs["order"])
        full = np.array(checkpoint[path])             # (K, Np_ckpt, Nk_tot)
        owned = self.part.local2global_elem[:self.part.n_owned]
        k0 = self.material.k_division.i_start
        k1 = self.material.k_division.i_stop
        dens = full[owned][:, :, k0:k1]               # (K_owned, Np_ckpt, Nk_local)
        dens = np.ascontiguousarray(dens.transpose(1, 0, 2))  # (Np_ckpt, K_owned, Nk_local)
        if N_ckpt != self.order:
            P = prolongation_matrix(N_ckpt, self.order)        # (Np, Np_ckpt)
            dens = np.einsum("ij,jkl->ikl", P, dens)           # (Np, K_owned, Nk_local)
            log.info(f"tri_set: restart prolonged order {N_ckpt} -> {self.order}")
        dens_local = torch.zeros(self.dg.Np, self.dg.K, self.Nk,
                                 device=rc.device, dtype=dtype)
        dens_local[:, :self.part.n_owned, :] = torch.from_numpy(dens).to(dens_local)
        self._rho = self.adv.apply_mass(dens_local)
        return True

    # ---- qimpy Geometry contract ----
    @property
    def rho(self) -> TensorList:
        return TensorList([self._rho])

    @rho.setter
    def rho(self, rho_new: TensorList) -> None:
        self._rho = rho_new[0]

    @property
    def density(self) -> torch.Tensor:
        """Nodal density u = M_k^{-1} w recovered from the conservative state."""
        return self.adv.apply_mass_inv(self._rho)

    @density.setter
    def density(self, dens: torch.Tensor) -> None:
        self._rho = self.adv.apply_mass(dens.to(self._rho))

    def contact_currents(self, t: float = 0.0) -> dict:
        """Net outward current through each contact (ammeter reading); positive
        means current flowing out of the device into the contact."""
        return self.dist.contact_currents(self._rho, t)

    def contact_potentials(self) -> dict:
        """Floating electrochemical potential of each voltage-probe contact."""
        return self.dist.contact_potentials()

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        w = rho[0]
        if self._distributed:
            mpi_halo_exchange(self.comm, self.part, w)   # refresh ghost rows
        # local_rhs applies the coupled (modal) boundary/volume flux and any
        # floating-contact update; calling adv.rhs_w directly would drop the
        # coupling (scalar path) and freeze the modal advection.
        spatial = self.dist.local_rhs(w, t)
        u = self.adv.apply_mass_inv(w)
        coll_u = self.material.rho_dot(
            u.reshape(-1, 1, self.Nk), t, id(self)).reshape(u.shape)
        coll = self.adv.apply_mass(coll_u)
        return TensorList([spatial + coll])

    def limit_positivity(self, rho: TensorList) -> TensorList:
        """Enforce a non-negative density (the m=0 channel) via the Zhang-Shu
        scaling limiter; conservative and a no-op where already non-negative.
        Applied after each SSP-RK stage by the time integrator when requested."""
        return TensorList([self.adv.limit_density(rho[0])])


    def update_stash(self, i_step: int, t: float) -> None:
        # observables evaluated at the DG nodes (exact), k-reduced by the material
        obs = self.material.measure_observables(self.density, t)  # (Np, K_local, n_obs)
        obs_owned = obs[:, :self.part.n_owned, :].detach().cpu().numpy()
        self._stash_i.append(i_step); self._stash_t.append(t)
        self._stash_obs.append(np.ascontiguousarray(obs_owned.transpose(1, 0, 2)))

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        mesh, N = self.mesh, self.order
        attrs = cp_path.attrs
        # attrs are reloaded as constructor params on restart, so only write
        # real parameters (yaml-overridable); 'order' lets restart bump the order
        attrs["order"] = N
        attrs["mesh_file"] = self.mesh_file
        n_obs = len(self.material.get_observable_names())
        Kg = len(mesh.EToV); Np = self.dg.Np
        Nk_tot = self.material.k_division.n_tot
        n_stash = len(self._stash_t)

        # mesh + node geometry are global on every rank -> collective write
        saved = [
            cp_path.write("mesh_vertices",
                          torch.from_numpy(np.stack([mesh.VX, mesh.VY], -1))),
            cp_path.write("mesh_triangles",
                          torch.from_numpy(mesh.EToV.astype(np.int64))),
            cp_path.write("node_xy",
                          torch.from_numpy(_global_node_xy(mesh, N))),
            cp_path.write("subtri",
                          torch.from_numpy(_ref_subtriangulation(N))),
            "dg_observables", "dg_rho",
        ]
        cp_path.write_str("contact_names", ",".join(self.contacts.keys()))
        cp_path.write_str("observable_names",
                          ",".join(self.material.get_observable_names()))
        cp_path["t"] = np.array(self._stash_t)
        cp_path["i_step"] = np.array(self._stash_i)

        checkpoint, path = cp_path
        cpc = CheckpointPath(checkpoint, path)
        cpc.create_dataset("dg_observables", (n_stash, Kg, Np, n_obs), np.float64)
        cpc.create_dataset("dg_rho", (Kg, Np, Nk_tot), np.float64)

        # gather owned nodal observables (all stash steps) and the current nodal
        # density (restart state) to the head, by global element id + channel
        owned = np.asarray(self.part.local2global_elem[:self.part.n_owned])
        obs_local = (np.stack(self._stash_obs, axis=0) if n_stash
                     else np.zeros((0, len(owned), Np, n_obs)))   # (n_stash,Ko,Np,nobs)
        dens = self.density.detach().cpu().numpy()                # (Np,K_local,Nk_local)
        dens_owned = np.ascontiguousarray(
            dens[:, :self.part.n_owned, :].transpose(1, 0, 2))     # (Ko,Np,Nk_local)
        k0 = int(self.material.k_division.i_start)
        package = rc.comm.gather((owned, k0, obs_local, dens_owned), root=0)

        if checkpoint is not None and rc.is_head:
            full_obs = np.zeros((n_stash, Kg, Np, n_obs))
            full_rho = np.zeros((Kg, Np, Nk_tot))
            for ids, ks, ob, de in package:
                if n_stash:
                    full_obs[:, ids, :, :] = ob
                full_rho[ids, :, ks:ks + de.shape[2]] = de
            checkpoint.write_slice(checkpoint[f"{path}/dg_observables"],
                                   (0, 0, 0, 0), torch.from_numpy(full_obs))
            checkpoint.write_slice(checkpoint[f"{path}/dg_rho"],
                                   (0, 0, 0), torch.from_numpy(full_rho))
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        return saved
