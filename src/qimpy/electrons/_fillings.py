import qimpy as qp
import numpy as np
import torch
import collections
from scipy import optimize
from typing import Optional, Dict, Union, Callable, List, Sequence, cast, \
    TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import Checkpoint, RunConfig
    from ..ions import Ions
    from ._electrons import Electrons
    from .._system import System
    from .. import Energy


SmearingResults = collections.namedtuple('SmearingResults',
                                         ['f', 'f_eig', 'S'])
SmearingFunc = Callable[[torch.Tensor, Union[float, torch.Tensor], float],
                        SmearingResults]


class Fillings(qp.Constructable):
    """Electron occupation factors (smearing)"""
    __slots__ = ('electrons', 'n_electrons',
                 'n_bands_min', 'n_bands', 'n_bands_extra',
                 'smearing', 'sigma', 'mu_constrain', 'M_constrain',
                 'mu', 'B', 'M', 'f', 'f_eig', '_smearing_func')
    electrons: 'Electrons'
    n_electrons: float  #: Number of electrons
    n_bands_min: int  #: Minimum number of bands to accomodate `n_electrons`
    n_bands: int  #: Number of bands to calculate
    n_bands_extra: int  #: Number of extra bands during diagonalization
    smearing: Optional[str]  #: Smearing method name
    sigma: Optional[float]  #: Gaussian width (:math:`2k_BT` for Fermi)
    mu: float  #: Electron chemical potential
    mu_constrain: bool  #: Whether to constrain chemical potential
    B: torch.Tensor  #: Magnetic field (vector in spinorial mode)
    M: torch.Tensor  #: Total magnetization (vector in spinorial mode)
    M_constrain: bool  #: Whether to constrain magnetization
    f: torch.Tensor  #: Electronic occupations
    f_eig: torch.Tensor  #: Derivative of `f` with electronic eigenvalues
    _smearing_func: Optional[SmearingFunc]  #: Smearing function calculator

    def __init__(self, *, co: qp.ConstructOptions,
                 ions: 'Ions', electrons: 'Electrons',
                 charge: float = 0., smearing: str = 'gauss',
                 sigma: Optional[float] = None,
                 kT: Optional[float] = None,
                 mu: float = np.nan,
                 mu_constrain: bool = False,
                 B: Union[float, Sequence[float]] = 0.,
                 M: Union[float, Sequence[float]] = 0.,
                 M_constrain: bool = False,
                 n_bands: Optional[Union[int, str]] = None,
                 n_bands_extra: Optional[Union[int, str]] = None) -> None:
        r"""Initialize occupation factor (smearing) scheme.

        Parameters
        ----------
        charge
            Net charge of electrons + ions in e units, which determines
            n_electrons = ions.Z_tot - charge.
        smearing : {'gauss', 'fermi', 'cold', 'mp1', False}, default: 'gauss'
            Smearing method for setting electron occupations, where 'gauss',
            'fermi', 'cold', 'mp1' select Gaussian, Fermi-Dirac, Cold and
            first order Methfessel-Paxton (MP1) smearing respectively.
            Use False (or None) to disable smearing and keep the electron
            occupations fixed at their initial values.
        sigma
            Width of the smearing function (in :math:`E_h`), corresponding
            to the Gaussian width :math:`\\sigma` in the Gaussian, Cold and
            M-P schemes, and to :math:`2k_BT` in the Fermi-Dirac scheme.
            Defaults to 0.002 if neither `sigma` or `kT` are specified.
        kT
            :math:`k_BT` for Fermi-Dirac occupations, amounting to
            :math:`\\sigma/2` for the other Gaussian-based smearing schemes.
            Defaults to 0.001 if neither `sigma` or `kT` are specified.
            Specify only one of sigma or kT.
        mu
            Electron chemical potential :math:`\mu`. This serves as an initial
            guess (rarely needed) if `mu_constrain` is False, and otherwise,
            it is the required target value to constrain :math:`\mu` to.
        mu_constrain
            Whether to hold chemical potential fixed to `mu` in
            occupation updates: this only matters when `smearing` is not None.
        B
            External magnetic field.
            Must be scalar for non-spinorial and 3-vector for spinorial modes.
            If `M_constrain` is True, then this is only an initial guess as the
            magnetic field then becomes a Legendre multiplier to constrain `M`.
        M
            Total magnetization (only for spin-polarized modes).
            Must be scalar for non-spinorial and 3-vector for spinorial modes.
            This magnetization is assigned to the initial occupations and it
            may change when smearing is present depending on `M_constrain`.
        M_constrain
            Whether to hold magnetization fixed to `M` in occupation updates:
            this only matters when `smearing` is not None.
        n_bands : {'atomic', 'x<scale>', int}, default: 'atomic'
            Number of bands, set to the number of atomic orbitals, or
            specified as a scale relative to the minimum number of bands
            to accommodate electrons ('x1.5' implies 1.5 x n_bands_min).
            Alternately, an integer explicitly sets the number of bands.
        n_bands_extra : {'x<scale>', int}, default: 'x0.1'
            Number of extra bands retained by diagonalizers, necessary to
            converge any degenerate subspaces straddling n_bands. This could
            be specified as a multiple of n_bands e.g. 'x0.1' = 0.1 x n_bands,
            or could be specified as an explicit number of extra bands
        """
        super().__init__(co=co)
        self.electrons = electrons

        # Magnetic field and magnetization mode:
        def check_magnetic(x: Union[float, Sequence[float]],
                           x_name: str) -> torch.Tensor:
            """Ensure that x = M or B is appropriate for spin mode."""
            x_len = (3 if electrons.spinorial else
                     (1 if electrons.spin_polarized else 0))
            if x:
                if not electrons.spin_polarized:
                    raise ValueError(f'{x_name} only allowed for'
                                     f' spin-polarized calculations')
                x_arr = torch.tensor(x, device=self.rc.device,
                                     dtype=torch.double).flatten()
                if x_len != len(x_arr):
                    prefix = ('' if electrons.spinorial else 'non-')
                    raise ValueError(f'{x_name} must have exactly {x_len} '
                                     f'components in {prefix}spinorial mode')
                return x_arr
            else:
                return torch.zeros(x_len, device=self.rc.device)
        self.B = check_magnetic(B, 'B')
        self.M = check_magnetic(M, 'M')
        self.M_constrain = bool(M_constrain)

        # Number of electrons and bands:
        self.n_electrons = ions.Z_tot - charge
        M_mag = (abs(self.M.item()) if (electrons.spin_polarized
                                        and not electrons.spinorial) else 0.)
        self.n_bands_min = int(np.ceil((self.n_electrons + 0.5*M_mag)
                                       * electrons.n_spinor / 2))

        # Smearing:
        self.smearing = smearing.lower() if smearing else None
        self.sigma = None
        if self.smearing:
            self._smearing_func = _smearing_funcs.get(self.smearing, None)
            if self._smearing_func is None:
                raise KeyError('smearing must be None/False or one of '
                               f'{_smearing_funcs.keys()}')
            if sigma and kT:
                raise ValueError('specify only one of sigma or kT')
            self.sigma = float(sigma if sigma  # get from sigma
                               else ((2*kT) if kT  # get from kT
                                     else 0.002))  # default value
            self.n_bands_min += 1  # need at least one extra empty band
        sigma_str = (f'{self.sigma:g} (equivalent kT: {0.5*self.sigma:g})'
                     if self.sigma else str(None))
        qp.log.info(f'n_electrons: {self.n_electrons:g}'
                    f'  n_bands_min: {self.n_bands_min}')
        qp.log.info(f'smearing: {self.smearing}  sigma: {sigma_str}')
        self.mu = float(mu)
        self.mu_constrain = bool(mu_constrain)
        if self.mu_constrain and np.isnan(self.mu):
            raise ValueError('mu must be specified for mu_constrain = True')
        qp.log.info(f'mu: initial: {self.mu}'
                    f'  constrained: {self.mu_constrain}')
        if electrons.spin_polarized:
            qp.log.info(f'M: initial: {self.rc.fmt(self.M)}'
                        f'  constrained: {self.M_constrain}'
                        f'  B: {self.rc.fmt(self.B)}')

        self._initialize_n_bands(ions, n_bands, n_bands_extra)
        self._initialize_f()

    def _initialize_n_bands(self, ions: 'Ions',
                            n_bands: Optional[Union[int, str]] = None,
                            n_bands_extra: Optional[Union[int, str]] = None
                            ) -> None:
        # Determine number of bands:
        if n_bands is None:
            n_bands = 'atomic'
        if isinstance(n_bands, int):
            self.n_bands = n_bands
            assert(self.n_bands >= 1)
            n_bands_method = 'explicit'
        else:
            assert isinstance(n_bands, str)
            if n_bands == 'atomic':
                n_bands_method = 'atomic'
                self.n_bands = ions.n_atomic_orbitals(self.electrons.n_spinor)
                if self.n_bands < self.n_bands_min:
                    qp.log.warning(f'Note: {self.n_bands} atomic orbitals'
                                   f' < n_bands_min = {self.n_bands_min}')
                    self.n_bands = self.n_bands_min
                    n_bands_method = '1*n_bands_min'
            else:
                assert n_bands.startswith('x')
                n_bands_scale = float(n_bands[1:])
                if n_bands_scale < 1.:
                    raise ValueError('<scale> must be >=1 in n_bands')
                self.n_bands = max(1, int(np.ceil(self.n_bands_min
                                                  * n_bands_scale)))
                n_bands_method = n_bands[1:] + '*n_bands_min'
        # --- similarly for extra bands:
        if n_bands_extra is None:
            n_bands_extra = 'x0.1'
        if isinstance(n_bands_extra, int):
            self.n_bands_extra = n_bands_extra
            assert(self.n_bands_extra >= 1)
            n_bands_extra_method = 'explicit'
        else:
            assert(isinstance(n_bands_extra, str)
                   and n_bands_extra.startswith('x'))
            n_bands_extra_scale = float(n_bands_extra[1:])
            if n_bands_extra_scale <= 0.:
                raise ValueError('<scale> must be >0 in n_bands_extra')
            self.n_bands_extra = max(1, int(np.ceil(self.n_bands
                                                    * n_bands_extra_scale)))
            n_bands_extra_method = n_bands_extra[1:] + '*n_bands'
        qp.log.info(
            f'n_bands: {self.n_bands} ({n_bands_method})'
            f'  n_bands_extra: {self.n_bands_extra} ({n_bands_extra_method})')

    def _initialize_f(self) -> None:
        """Create or load initial fillings."""
        el = self.electrons
        k_division = el.kpoints.division
        self.f = torch.zeros((el.n_spins, k_division.n_mine, self.n_bands),
                             device=self.rc.device)
        if self._checkpoint_has('f'):
            qp.log.info("Loading fillings f")
            self.read_band_scalars(cast('Checkpoint', self.checkpoint_in),
                                   self.path + 'f', self.f)
        else:
            # Compute fillings
            qp.log.info("Constructing fillings f to occupy lowest bands")
            # --- Fillings sum for each spin channel:
            f_sums = (np.ones(el.n_spins) * self.n_electrons
                      / (el.w_spin * el.n_spins))
            if el.spin_polarized and (not el.spinorial):
                half_M = 0.5 * self.M.item()
                f_sums[0] += half_M
                f_sums[1] -= half_M
                if f_sums.min() < 0:
                    raise ValueError(f'M = {(2 * half_M):g} too large for'
                                     f' n_electrons = {self.n_electrons:g}')
                if f_sums.max() > self.n_bands:
                    raise ValueError(f'M = {(2 * half_M):g} too large for'
                                     f' n_bands = {self.n_bands:g}')
            # --- Initialize fillings based on sum in each channel:
            for i_spin, f_sum in enumerate(f_sums):
                n_full = int(np.floor(f_sum))  # number of fully filled bands
                self.f[i_spin, :, :n_full] = 1.  # full fillings
                if f_sum > n_full:
                    self.f[i_spin, :, n_full] = (f_sum - n_full)  # left overs
        self.f_eig = torch.zeros_like(self.f)

    def update(self, energy: 'Energy') -> None:
        """Update fillings `f` and chemical potential `mu`, if needed.
        Set corresponding energy components in `energy`.
        """
        if self.sigma is None:  # Fixed fillings
            return
        if np.isnan(self.electrons.deig_max):  # Eigenvalues not yet available
            energy.setdefault('-TS', 0.)  # Ensure correct energy name
            return

        assert self._smearing_func is not None
        el = self.electrons
        w_sk = el.basis.w_sk
        eig = el.eig[..., :self.n_bands]  # don't include extra bands in f

        # Guess chemical potential from eigenvalues if needed:
        if np.isnan(self.mu):
            n_full = int(np.floor(self.n_electrons /
                                  (el.w_spin * el.n_spins)))
            self.mu = qp.utils.globalreduce.min(eig[:, :, n_full],
                                                self.rc.comm_kb)

        # Weights that generate number / magnetization and their targets:
        if el.spin_polarized:
            if el.spinorial:
                w_NM = torch.cat((torch.ones_like(eig)[None, ...],
                                  el.C.band_spin()), dim=0)
                M_len = 3
            else:
                w_NM = torch.tensor([[1, 1], [1, -1]],
                                    device=self.rc.device).view(2, 2, 1, 1)
                M_len = 1
            NM_target = np.concatenate(([self.n_electrons],
                                        self.M.to(self.rc.cpu)))
            mu_B = np.concatenate(([self.mu], self.B.to(self.rc.cpu)))
            i_free = np.where([not self.mu_constrain]
                              + [self.M_constrain] * M_len)[0]
        else:
            w_NM = torch.ones((1, 1, 1, 1), device=self.rc.device)
            NM_target = np.array([self.n_electrons])
            mu_B = np.array([self.mu])
            i_free = np.where([not self.mu_constrain])[0]
        results = {}  # populated with NM and S by compute_NM

        def compute_NM(params):
            """Compute electron number and magnetization from eigenvalues.
            Here params are the entries in mu_B that are being optimized.
            Return error in corresponding NM entries, and its gradient
            with respect to the mu_B being optimized.
            Also store NM, f and entropy in `results` of outer scope.
            """
            mu_B[i_free] = params
            mu_B_t = torch.tensor(mu_B).to(self.rc.device)
            mu_eff = (mu_B_t.view(-1, 1, 1, 1) * w_NM).sum(dim=0)
            f, f_eig, S = self._smearing_func(eig, mu_eff, sigma_cur)
            NM = (w_NM * (w_sk * f)).sum(dim=(1, 2, 3))
            NM_mu_B = -((w_NM[None, ...] * w_NM[:, None, ...])
                        * (w_sk * f_eig)).sum(dim=(2, 3, 4))
            # Collect across MPI and make consistent to machine precision:
            for tensor in (NM, NM_mu_B):
                self.rc.comm_k.Allreduce(qp.MPI.IN_PLACE,
                                         qp.utils.BufferView(tensor),
                                         qp.MPI.SUM)
                self.rc.comm_kb.Bcast(qp.utils.BufferView(tensor))
            self.f = f
            self.f_eig = f_eig
            results['NM'] = NM
            results['S'] = S
            # Compute errors:
            NM_err = NM.to(self.rc.cpu).numpy() - NM_target
            NM_err_mu_B = NM_mu_B.to(self.rc.cpu).numpy()
            return NM_err[i_free], NM_err_mu_B[i_free][:, i_free]

        if len(i_free) == 0:
            compute_NM(np.array([]))  # both mu and B are fixed
        elif not self.M_constrain:
            # Only mu is free: use a stable bracketing algorithm
            # --- Find a bracketing interval:
            def root_func1d(mu: float) -> float:
                return compute_NM([mu])[0]

            def expand_range(sign: int) -> float:
                mu_limit = self.mu
                dmu = sign * sigma_cur
                while sign * root_func1d(mu_limit) <= 0.:
                    mu_limit += dmu
                    dmu *= 2.
                return mu_limit

            sigma_cur = self.sigma
            mu_min = expand_range(-1)
            mu_max = expand_range(+1)
            self.mu = optimize.brentq(root_func1d, mu_min, mu_max)
        else:  # B is free: use a quasi-Newton method
            # Find mu and/or B to match N and/or M as appropriate:
            # --- start with a larger sigma and reduce down for stability:
            eig_diff_max = qp.utils.globalreduce.max(eig.diff(dim=-1),
                                                     self.rc.comm_k)
            sigma_cur = max(self.sigma, min(0.1, eig_diff_max))
            final_step = False
            while not final_step:
                final_step = (sigma_cur == self.sigma)
                xtol = 1e-12 if final_step else (0.1 * sigma_cur)
                res = optimize.root(compute_NM, mu_B[i_free],
                                    jac=True, method='lm',
                                    options={'xtol': xtol, 'ftol': 1e-12})
                sigma_cur = max(self.sigma, 0.5 * sigma_cur)
            if np.max(np.abs(res.fun)) > 1e-10 * self.n_electrons:
                raise ValueError('Density/magnetization constraint failed:'
                                 ' check if n_bands is sufficient or if'
                                 ' mu/B guesses are reasonable.')

        # Update fillings and entropy accordingly:
        energy['-TS'] = (-0.5 * self.sigma) * qp.utils.globalreduce.sum(
            w_sk * results['S'], self.rc.comm_k)
        # --- update n_electrons or mu, depending on which is free
        n_electrons = results['NM'][0].item()
        if self.mu_constrain:
            self.n_electrons = n_electrons
        else:
            self.mu = mu_B[0].item()
        # --- update magnetization or B, depending on which is free
        if el.spin_polarized:
            M = results['NM'][1:]
            if self.M_constrain:
                self.B = torch.from_numpy(mu_B[1:]).to(self.rc.device)
            else:
                self.M = M
            M_str = f'  M: {self.rc.fmt(M, floatmode="fixed", precision=5)}'
        else:
            M_str = ''
        qp.log.info(f'  FillingsUpdate:  mu: {self.mu:.9f}'
                    f'  n_electrons: {n_electrons:.6f}{M_str}')

    def _save_checkpoint(self, checkpoint: 'Checkpoint') -> List[str]:
        self.write_band_scalars(checkpoint, self.path + 'f', self.f)
        return ['f']

    def write_band_scalars(self, checkpoint: 'Checkpoint', path: str,
                           v: torch.Tensor) -> None:
        """Write `v` containing one scalar per band to `path` in `checkpoint`.
        This is useful for writing fillings, eigenvalues etc."""
        el = self.electrons
        k_division = el.kpoints.division
        shape = (el.n_spins, k_division.n_tot, self.n_bands)
        offset = (0, k_division.i_start, 0)
        dset = checkpoint.create_dataset(path, shape=shape,
                                         dtype=self.rc.np_type[v.dtype])
        if self.rc.i_proc_b == 0:
            checkpoint.write_slice(dset, offset, v[:, :, :self.n_bands])

    def read_band_scalars(self, checkpoint: 'Checkpoint', path: str,
                          v: torch.Tensor) -> int:
        """Read one scalar per band from `path` in `checkpoint` into `v`.
        Returns number of bands read, which may be <= `self.n_bands`.
        This is useful for reading fillings, eigenvalues etc."""
        el = self.electrons
        k_division = el.kpoints.division
        dset = checkpoint[path]
        n_bands_in = min(dset.shape[-1], self.n_bands)
        offset = (0, k_division.i_start, 0)
        size = (el.n_spins, k_division.n_mine, n_bands_in)
        v[..., :n_bands_in] = checkpoint.read_slice(dset, offset, size)
        return n_bands_in


def _smearing_fermi(eig: torch.Tensor, mu: Union[float, torch.Tensor],
                    sigma: float) -> SmearingResults:
    """Compute Fermi-Dirac occupations, its energy derivative and entropy.
    Note that sigma is taken as 2 kT to keep width consistent."""
    f = torch.special.expit((mu - eig)/(0.5*sigma))
    f_eig = f * (1 - f) / (-0.5*sigma)
    S = -f.xlogy(f) - (1-f).xlogy(1-f)
    return SmearingResults(f, f_eig, S)


def _smearing_gauss(eig: torch.Tensor, mu: Union[float, torch.Tensor],
                    sigma: float) -> SmearingResults:
    """Compute Gaussian (erfc) occupations, energy derivative and entropy."""
    x = (eig - mu) / sigma
    f = 0.5*torch.erfc(x)
    S = torch.exp(-x*x) / np.sqrt(np.pi)
    f_eig = (-1./sigma) * S
    return SmearingResults(f, f_eig, S)


def _smearing_mp1(eig: torch.Tensor, mu: Union[float, torch.Tensor],
                  sigma: float) -> SmearingResults:
    """Compute first-order Methfessel-Paxton occupations, energy derivative
    and entropy."""
    x = (eig - mu) / sigma
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) - x * gaussian)
    f_eig = (x*x - 1.5) * gaussian / sigma
    S = (0.5 - x*x) * gaussian
    return SmearingResults(f, f_eig, S)


def _smearing_cold(eig: torch.Tensor, mu: Union[float, torch.Tensor],
                   sigma: float) -> SmearingResults:
    """Compute Cold smearing occupations, energy derivative and entropy."""
    x = (eig - mu) / sigma + np.sqrt(0.5)  # note: not centered at mu
    sqrt2 = np.sqrt(2.)
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) + sqrt2*gaussian)
    f_eig = -gaussian * (1 + x*sqrt2) / sigma
    S = gaussian * x * sqrt2
    return SmearingResults(f, f_eig, S)


_smearing_funcs: Dict[str, SmearingFunc] = {
    'fermi': _smearing_fermi,
    'gauss': _smearing_gauss,
    'mp1': _smearing_mp1,
    'cold': _smearing_cold}


if __name__ == '__main__':
    def main():
        """Check derivatives and plot comparison of smearing functions."""
        import matplotlib.pyplot as plt
        torch.set_default_tensor_type(torch.DoubleTensor)
        mu = 0.3
        kT = 0.01
        eig = torch.linspace(mu-20*kT, mu+20*kT, 4001)
        deig = eig[1] - eig[0]
        for name, func in _smearing_funcs.items():
            f, f_eig, S = func(eig, mu, 2*kT)
            f_eig_num = (f[2:] - f[:-2])/(2*deig)
            f_eig_err = (f_eig[1:-1] - f_eig_num).norm() / f_eig_num.norm()
            print(f'{name:>5s}:  Err(f_eig): {f_eig_err:.2e}'
                  f'  integral(S): {S.sum()*deig:f}')
            e = (eig - mu)/kT  # dimensionless energy
            for i, (result, result_name) in enumerate([
                    (f, '$f$'), (f_eig, r'$df/d\varepsilon$'), (S, '$S$')]):
                plt.figure(i)
                plt.plot(e, result, label=name)
                plt.xlabel(r'$(\varepsilon-\mu)/k_BT$')
                plt.ylabel(result_name)
                plt.legend()
        plt.show()
    main()
