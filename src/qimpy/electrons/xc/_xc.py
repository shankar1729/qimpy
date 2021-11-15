import qimpy as qp
import numpy as np
import torch
from . import lda, gga
from .functional import Functional, get_libxc_functional_names, FunctionalsLibxc
from typing import List, Dict, Union


N_CUT = 1e-16  # Regularization threshold for densities


class XC(qp.TreeNode):
    """Exchange-correlation functional."""

    __slots__ = ("rc", "_functionals", "need_sigma", "need_lap", "need_tau")
    rc: qp.utils.RunConfig
    _functionals: List[Functional]  #: list of functionals that add up to XC
    need_sigma: bool  #: whether overall functional needs gradient
    need_lap: bool  #: whether overall functional needs laplacian
    need_tau: bool  #: whether overall functional needs KE density

    def __init__(
        self,
        *,
        rc: qp.utils.RunConfig,
        spin_polarized: bool,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        functional: Union[str, List[str]] = "gga-pbe",
    ):
        """Initialize exchange-correlation functional.

        Parameters
        ----------
        functional
            :yaml:`Name or list of names of exchange-correlation functionals.`
            Each entry in the list must be one of the internal functionals:

            {INTERNAL_FUNCTIONAL_NAMES}

            or a Libxc functional name (if available). Run the code with
            functional: 'list' to print the names of available functionals
            including those from Libxc (and exit immediately).
            The names are case insensitive, and may use hyphens or underscores.

            Additionally, for Libxc functionals, a combined xc name will expand
            to separate x and c names for convenience, where appropriate.
            Therefore, 'gga-pbe' (default) will use the internal PBE GGA,
            while 'gga-xc-pbe' will use 'gga_x_pbe' + 'gga_c_pbe' from Libxc.

            Finally, each functional name in the list can have an optional
            "\\*num" suffix (no spaces) to scale the functional by num.
            For example, the specification 'gga-pbe\\*0.5 lda-pz\\*0.5' may be
            used to compute a 50-50 mix of two functionals. Warning: there is
            no normalization or check to make the fractions of exchange or
            correlation to add up to 1.
        """
        super().__init__()
        self.rc = rc
        qp.log.info("\nInitializing XC:")
        if isinstance(functional, str):
            functional = functional.split(" ")

        # Initialize internal and LibXC functional objects:
        self._functionals = []
        libxc_names: Dict[str, float] = {}  # Libxc names with scale factors
        for func_name in functional:
            if "*" in func_name:
                tokens = func_name.split("*")
                func_name = tokens[0]
                scale_factor = float(tokens[1])
            else:
                scale_factor = 1.0
            for func in _get_functionals(func_name, scale_factor):
                if isinstance(func, Functional):
                    self._functionals.append(func)
                else:
                    libxc_names.setdefault(func, 0.0)
                    libxc_names[func] += 1.0
        if libxc_names:
            self._functionals.append(
                FunctionalsLibxc(self.rc, spin_polarized, libxc_names)
            )

        # Collect overall needs:
        self.need_sigma = any(func.needs_sigma for func in self._functionals)
        self.need_lap = any(func.needs_lap for func in self._functionals)
        self.need_tau = any(func.needs_tau for func in self._functionals)

    def __call__(self, n_tilde: qp.grid.FieldH, tau_tilde: qp.grid.FieldH) -> float:
        """Compute exchange-correlation energy and potential.
        Here, `n_tilde` and `tau_tilde` are the electron density and KE density
        (used if `need_tau` is True) in reciprocal space.
        Potentials i.e. gradients with respect to `n_tilde` and `tau_tilde` are set
        within the corresponding `grad` attributes if `required_grad` is True.
        Presently, either both gradients or no gradients should be requested."""
        grid = n_tilde.grid
        watch = qp.utils.StopWatch("XC", grid.rc)
        n_in = (~n_tilde).data
        n_densities = n_in.shape[0]

        # Initialize local spin basis for vector-spin mode:
        if n_densities == 4:
            Mvec = n_in[1:]
            MmagInv = 1.0 / Mvec.norm(dim=0).clamp(min=N_CUT)
            Mhat = Mvec * MmagInv  # regularized unit vector

        # Get required quantities in local-spin basis:
        def from_magnetization(x_in: torch.Tensor) -> torch.Tensor:
            """Transform a density-like quantity `x` from magnetization to up/dn basis.
            First dimension of `x_in` must be the spin dimension."""
            if n_densities == 1:
                return x_in
            x = torch.empty((2,) + x_in.shape[1:], dtype=x_in.dtype, device=x_in.device)
            if n_densities == 4:
                # Broadcast Mhat with any batch dimensions of x_in:
                n_batch = len(x_in.shape) - 4
                Mhat_view = Mhat.view((3,) + (1,) * n_batch + Mhat.shape[1:])
                # Project vector component of x against Mhat:
                xM = (x_in[1:] * Mhat_view).sum(dim=0)
                x[0] = 0.5 * (x_in[0] + xM)
                x[1] = 0.5 * (x_in[0] - xM)
            else:  # n_densities == 2:
                x[0] = 0.5 * (x_in[0] + x_in[1])
                x[1] = 0.5 * (x_in[0] - x_in[1])
            return x

        n = from_magnetization(n_in)
        n_spins = n.shape[0]  # always 1 or 2 (local basis in vector case)
        # --- density gradient:
        if self.need_sigma:
            Dn_in = (~(n_tilde.gradient(dim=1))).data
            Dn = from_magnetization(Dn_in)
            sigma = torch.empty(
                (2 * n_spins - 1,) + n.shape[1:], dtype=n.dtype, device=n.device
            )
            for s1 in range(n_spins):
                for s2 in range(s1, n_spins):
                    sigma[s1 + s2] = (Dn[s1] * Dn[s2]).sum(dim=0)
        else:
            sigma = torch.tensor(0.0, device=n.device)
        # --- laplacian:
        if self.need_lap:
            lap_in = (~(n_tilde.laplacian())).data
            lap = from_magnetization(lap_in)
        else:
            lap = torch.tensor(0.0, device=n.device)
        # --- KE density:
        if self.need_tau:
            tau_in = (~tau_tilde).data
            tau = from_magnetization(tau_in)
        else:
            tau = torch.tensor(0.0, device=n.device)

        # Clamp low densities for numerical stability:
        clamp_sel = torch.where(n < N_CUT)
        n[clamp_sel] = N_CUT

        # Evaluate functionals:
        requires_grad = n_tilde.requires_grad
        assert requires_grad == tau_tilde.requires_grad  # compute all or no gradients
        if requires_grad:
            n.grad = torch.zeros_like(n)
            sigma.grad = torch.zeros_like(sigma)
            lap.grad = torch.zeros_like(lap)
            tau.grad = torch.zeros_like(tau)
        E = 0.0
        for functional in self._functionals:
            E += functional(n, sigma, lap, tau, requires_grad) * grid.dV

        # Gradient propagation for potential:
        def from_magnetization_grad(x: torch.Tensor, x_in: torch.Tensor) -> None:
            """Gradient propagation corresponding to `from_magnetization`.
            Sets the gradient contribution to `x_in.grad` from `x.grad`.
            In vector-spin mode, this also contributes to `n_in.grad`, even
            when `x` is not `n` because `n` determines `Mhat`."""
            if n_densities == 1:
                return
            x_in.grad = torch.empty(
                (n_densities,) + x.shape[1:], dtype=x.dtype, device=x.device
            )
            x_in.grad[0] = 0.5 * (x.grad[0] + x.grad[1])
            x_diff_grad = 0.5 * (x.grad[0] - x.grad[1])
            if n_densities == 4:
                # Broadcast Mhat with any batch dimensions of x_in:
                n_batch = len(x.shape) - 4
                Mhat_view = Mhat.view((3,) + (1,) * n_batch + Mhat.shape[1:])
                # Propagate x_diff_grad (which is Mhat_grad) to x_in.grad:
                x_in.grad[1:] = x_diff_grad * Mhat_view
                # Additional propagation of Mhat_grad to n_in.grad:
                if x_in is not None:
                    x_vec = x_in[1:]
                    M_grad = (
                        x_diff_grad
                        * MmagInv
                        * (x_vec - Mhat_view * (Mhat_view * x_vec).sum(dim=0))
                    )
                    if n_batch:
                        batch_dims = tuple(range(1, 1 + n_batch))
                        M_grad = M_grad.sum(dim=batch_dims)
                    n_in.grad[1:] += M_grad
            else:  # n_densities == 2:
                x_in.grad[1] = x_diff_grad

        if requires_grad:
            n.grad[clamp_sel] = 0.0  # account for any clamping
            from_magnetization_grad(n, n_in)
            # --- contributions from GGA gradients:
            if self.need_sigma:
                Dn.grad = torch.zeros_like(Dn)
                for s1 in range(n_spins):
                    for s2 in range(s1, n_spins):
                        Dn.grad[s1] += sigma.grad[s1 + s2] * Dn[s2]
                        Dn.grad[s2] += sigma.grad[s1 + s2] * Dn[s1]
                from_magnetization_grad(Dn, Dn_in)
                n_tilde.grad -= (~qp.grid.FieldR(grid, data=Dn_in.grad)).divergence(
                    dim=1
                )
            # --- contributions from Laplacian:
            if self.need_lap:
                from_magnetization_grad(lap, lap_in)
                n_tilde.grad += (~qp.grid.FieldR(grid, data=lap_in.grad)).laplacian()
            # --- contributions from KE density:
            if self.need_tau:
                from_magnetization_grad(tau, tau_in)
                tau_tilde.grad += ~qp.grid.FieldR(grid, data=tau_in.grad)
            # --- direct n contributions
            n_tilde.grad += ~qp.grid.FieldR(grid, data=n_in.grad)

        # Collect energy
        if grid.comm is not None:
            E = grid.comm.allreduce(E, qp.MPI.SUM)
        watch.stop()
        return E


INTERNAL_FUNCTIONAL_NAMES = {
    "lda_pz",
    "lda_pw",
    "lda_pw_prec",
    "lda_vwn",
    "lda_teter",
    "gga_pbe",
    "gga_pbesol",
}


# Substitute internal functional names in XC docstring:
if XC.__init__.__doc__:
    XC.__init__.__doc__ = XC.__init__.__doc__.replace(
        "{INTERNAL_FUNCTIONAL_NAMES}",
        "* " + "\n            * ".join(sorted(INTERNAL_FUNCTIONAL_NAMES)),
    )


def _get_functionals(name: str, scale_factor: float) -> List[Union[Functional, str]]:
    """Get list of Functional objects associated with a functional `name`.
    For Libxc functionals, a validated list of strings is returned so that
    all the Libxc evaluations can be consolidated in a single wrapper.
    The Functional objects will be initialized with specified scale factor,
    while the scale factors of the Libxc names must be handled separately."""
    key = name.lower().replace("-", "_")
    if key in INTERNAL_FUNCTIONAL_NAMES:
        # Initialize appropriate combination of internal functionals:
        if key == "lda_pz":
            return [
                lda.X_Slater(scale_factor=scale_factor),
                lda.C_PZ(scale_factor=scale_factor),
            ]
        elif key == "lda_pw":
            return [
                lda.X_Slater(scale_factor=scale_factor),
                lda.C_PW(scale_factor=scale_factor, high_precision=False),
            ]
        elif key == "lda_pw_prec":
            return [
                lda.X_Slater(scale_factor=scale_factor),
                lda.C_PW(scale_factor=scale_factor, high_precision=True),
            ]
        elif key == "lda_vwn":
            return [
                lda.X_Slater(scale_factor=scale_factor),
                lda.C_VWN(scale_factor=scale_factor),
            ]
        elif key == "lda_teter":
            return [lda.XC_Teter(scale_factor=scale_factor)]
        elif key == "gga_pbe":
            return [
                gga.X_PBE(scale_factor=scale_factor, sol=False),
                gga.C_PBE(scale_factor=scale_factor, sol=False),
            ]
        else:  # key == 'gga_pbesol'
            return [
                gga.X_PBE(scale_factor=scale_factor, sol=True),
                gga.C_PBE(scale_factor=scale_factor, sol=True),
            ]
    else:
        # Check LibXC functionals:
        libxc_names = get_libxc_functional_names()
        # --- List available functionals if requested:
        if key == "list":
            qp.log.info(
                "\nAvailable internal XC functionals:\n"
                f"\n{sorted(INTERNAL_FUNCTIONAL_NAMES)}\n"
            )
            if libxc_names:
                qp.log.info(
                    "\nAvailable Libxc functionals:\n"
                    f"\n{np.array(sorted(libxc_names))}\n"
                )
            else:
                qp.log.info("\nLibxc not available.")
            exit()
        # --- Try the specified name directly first:
        if key in libxc_names:
            return [key]
        # --- If not, try spitting xc to x and c functionals:
        if "_xc_" in key:
            key_x = key.replace("_xc_", "_x_")
            key_c = key.replace("_xc_", "_c_")
            if (key_x in libxc_names) and (key_c in libxc_names):
                return [key_x, key_c]
        raise KeyError(f"Unknown XC functional {name}")
