from qimpy import TreeNode, log
from qimpy.io import CheckpointPath, CheckpointContext
from .. import Wavefunction


class PlusU(TreeNode):
    """DFT+U correction."""

    U_values: dict[tuple[str, str], float]  #: map specie, orbital -> U value

    def __init__(
        self, *, checkpoint_in: CheckpointPath = CheckpointPath(), **U_values: float
    ) -> None:
        """Initialize from components and/or dictionary of options.

        Parameters
        ----------
        U_values
            :yaml:`Dictionary of U values by species and orbital names.`
            For example, to add U to Cu d and O s and p, the yaml input would be:

            .. code-block:: yaml

                plus_U:
                  Cu d: 2.4 eV
                  O s: 0.1 eV
                  O p: 0.7 eV
        """
        super().__init__()
        self.U_values = {}
        
        self.angular_qnum = {'s':0,'p':1,'d':2, 'f':3}
        
        for key, U in U_values.items():
            specie, orbital = key.split()
            
            assert specie in self.ions.symbols
            
            log.info(f"  +U on {specie}: {U}")
            self.U_values[(specie, orbital)] = float(U)
    
    def rhoAtom_calc(self, specie: str, orbital: str) -> torch.tensor:

        ions = self.ions
        symbols = ions.symbols
        basis = self.electrons.basis
        
        orbital_index = get_atomic_orbital_index(ions,basis)
        orbitals = get_atomic_orbitals(ions,basis)

        ell = self.angular_qnum[orbital]
        
        atom_and_orbital_index = torch.logical_and(orbital_index[:,0] == atom_index, orbital_index[:,2] == ell)
        atom_and_orbital = orbitals[:,:,atom_and_orbital_index,:,:]
        
        KS_orbitals = system.electrons.C
        fillings = system.electrons.fillings.f

        w_spin = 2 // (basis.n_spins * basis.n_spinor)
        wk = basis.wk
        wk = torch.complex(wk,torch.zeros_like(wk))
        
        psi_phi_inner = KS_orbitals.dot_O(atom_and_orbital).wait()
        rho = torch.einsum('k,skbm,skb,sknb -> mn',wk,psi_phi_inner,fillings,dagger(psi_phi_inner))

        return rho

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        for (specie, orbital), U in self.U_values.items():
            attrs[f"{specie} {orbital}"] = U
        return list(attrs.keys())

    def __bool__(self) -> bool:
        return bool(self.U_values)

    def __call__(self, C: Wavefunction) -> Wavefunction:
        """TODO."""
        raise NotImplementedError
