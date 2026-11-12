import pytest
import torch
import torch.distributed as dist

from qimpy.dft import System


@pytest.mark.parametrize("system_name", ["system_Si_perturbed", "system_H2O"])
def test_augmentation(system_name: str, request) -> None:
    """Check compatibility of overlap and density augmentation,
    and gradient correctness of density_grad."""
    system: System = request.getfixturevalue(system_name)
    electrons = system.electrons
    electrons.initialize_wavefunctions(system)
    electrons.update(system, requires_grad=True)

    # Overlap vs density augmentation:
    n_electrons_expected = electrons.fillings.n_electrons
    n_electrons = electrons.n_tilde.integral().item()
    assert abs(n_electrons - n_electrons_expected) < 1e-12

    # Potential correctness:
    C = electrons.C
    f = electrons.fillings.f
    beta_C = C.proj[..., : electrons.fillings.n_bands]
    wf = (f * electrons.basis.w_sk).to(beta_C.dtype)
    n_aug = system.ions.augmentation.density(C, f, need_Mvec=False)
    D_aug = system.ions.augmentation.D_aug
    dE_expected = (electrons.n_tilde.grad ^ n_aug).sum().to(D_aug.dtype)
    dE = torch.einsum("skji, skib, skb, skjb ->", D_aug, beta_C.conj(), wf, beta_C)
    if (group_k := electrons.kpoints.group).size() > 1:
        dist.all_reduce(dE, group=group_k)
    assert torch.allclose(dE, dE_expected, atol=1e-12)
