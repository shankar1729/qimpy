from __future__ import annotations
from typing import Optional
import torch

from qimpy import TreeNode, log
from qimpy.math import dagger
from qimpy.io import CheckpointPath, CheckpointContext

from .. import Wavefunction, Fillings  
from ...ions._ions_atomic import get_atomic_orbitals, get_atomic_orbital_index

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
        
        self.angular_qnum = {
            's': 0,
            'p': 1,
            'd': 2,
            'f': 3,
        }
        
        for key, U in U_values.items():
    
            specie, orbital = key.split()
   
            log.info(f"  +U on {specie}: {U}")
            self.U_values[(specie, orbital)] = float(U)

    def _rhoAtom_common(self, C: Wavefunction) -> None:
        
        self.basis = C.basis 
        self.ions = self.basis.ions
        self.n_ions_type = self.ions.n_ions_type
        self.symbols = self.ions.symbols
        
        self.wk = self.basis.wk
        self.wk = torch.complex(self.wk, torch.zeros_like(self.wk))
        self.w_spin = 2 // (self.basis.n_spins * self.basis.n_spinor)

        keys = list(self.U_values.keys())
        self.species = [key[0] for key in keys]
        self.shells = [key[1] for key in keys]


    def _rhoAtom_getOrbitals(self, atom_index: int, orbital: str) -> torch.Tensor:

        orbital_index = get_atomic_orbital_index(self.ions, self.basis)
        orbitals = get_atomic_orbitals(self.ions, self.basis)

        ell = self.angular_qnum[orbital]
        
        atom_and_orbital_index = torch.logical_and(orbital_index[:,0] == atom_index, orbital_index[:,2] == ell)
        atom_and_orbital = orbitals[:,:,atom_and_orbital_index,:,:]

        return atom_and_orbital

    def _rhoAtom_calc(self, species: list[str], shells: list[str], C: Wavefunction, fillings: Fillings) -> torch.tensor:
        
        C.coeff = C.coeff[:, :, :fillings.n_bands, :, :]

        rhoAtom = []
        for specie, orbital in zip(species, shells):
            ion_index = self.symbols.index(specie)
            for n_atoms in range(self.n_ions_type[ion_index]):
                atom_index = sum(self.n_ions_type[:ion_index]) + n_atoms

                atom_and_orbital = self._rhoAtom_getOrbitals(atom_index, orbital)
                psi_phi_inner = C.dot_O(atom_and_orbital).wait()

                rho = torch.einsum('k,skbm,skb,sknb -> mn', self.wk, psi_phi_inner, fillings.f, dagger(psi_phi_inner))

                rhoAtom.append(rho)
        
        rhoAtom = torch.stack(rhoAtom)

        return rhoAtom

    def rhoAtom_computeU(self, C: wavefunction, fillings: Fillings) -> float:
        
        #TODO: external potentials

        self._rhoAtom_common(C)
        rhoAtom = self._rhoAtom_calc(self.species, self.shells, C, fillings)

        Uprefac = []
        for specie, orbital in zip(self.species, self.shells):
            ion_index = self.symbols.index(specie)
            for n_atoms in range(self.n_ions_type[ion_index]):
                Uprefac.append(self.w_spin*0.5*self.U_values[(specie,orbital)])
        Uprefac = torch.tensor(Uprefac)

        Utot = torch.einsum('a,amm -> ',Uprefac,(rhoAtom - rhoAtom @ rhoAtom).real).item()
        self.U_rho = torch.diag(Uprefac) - 2*(torch.einsum('a,amn -> amn', Uprefac, rhoAtom)) 

        return Utot

    def rhoAtom_grad(self, C: Wavefunction) -> Wavefunction:
        
        gradient = C.zeros_like()
        for specie, orbital in zip(self.species, self.shells):
            ion_index = self.symbols.index(specie)
            for n_atoms in range(self.n_ions_type[ion_index]):
                atom_index = sum(self.n_ions_type[:ion_index]) + n_atoms
                
                atom_and_orbitals = self._rhoAtom_getOrbitals(atom_index, orbital)

                psi_phi_inner = C.dot_O(atom_and_orbitals)
                psi_phi_inner = psi_phi_inner.wait()
                gradient.coeff += torch.einsum('k, skbrn, abb, skmb -> skmrn', 1/self.wk, atom_and_orbitals.coeff, self.U_rho, psi_phi_inner)
        
        return gradient


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
