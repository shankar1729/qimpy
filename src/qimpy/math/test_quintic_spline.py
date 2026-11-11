import torch
import pytest

from qimpy import rc
from qimpy.io import log_config
from .quintic_spline import get_coeff, Interpolator


def get_test_data():
    def f_test(x):
        """Non-trivial test function with correct symmetries"""
        return torch.exp(-torch.sin(0.01 * x * x)) * torch.cos(0.1 * x)

    def f_test_prime(x):
        """Analytical derivative of above."""
        return -torch.exp(-torch.sin(0.01 * x * x)) * (
            torch.sin(0.1 * x) * 0.1
            + torch.cos(0.01 * x * x) * 0.02 * x * torch.cos(0.1 * x)
        )

    dx = 0.1
    x = torch.arange(0.0, 40.0, dx, device=rc.device)
    x_fine = torch.linspace(x.min(), x.max() - 1e-6 * dx, 2001, device=rc.device)
    y = f_test(x)
    y_fine = f_test(x_fine)
    y_prime_fine = f_test_prime(x_fine)
    y_coeff = get_coeff(y)  # blip coefficients
    return dx, x_fine, y_fine, y_prime_fine, y_coeff


@pytest.mark.mpi_skip
def test_interpolator():
    dx, x_fine, y_fine, y_prime_fine, y_coeff = get_test_data()
    assert (y_fine - Interpolator(x_fine, dx, 0)(y_coeff)).norm() < dx**4
    assert (y_prime_fine - Interpolator(x_fine, dx, 1)(y_coeff)).norm() < dx**3


def main():
    """Run test and additionally plot for visual inspection."""
    import matplotlib.pyplot as plt

    log_config()
    rc.init()

    # Plot a single blip function for testing:
    plt.figure()
    coeff = torch.zeros(12)
    coeff[5] = 1
    t = torch.linspace(0.0, 12.0, 101, device=rc.device)
    for deriv in range(5):
        plt.plot(
            t.to(rc.cpu),
            Interpolator(t, 2.0, deriv)(coeff).to(rc.cpu),
            label=f"Deriv: {deriv}",
        )
    plt.axhline(0, color="k", ls="dotted")
    plt.legend()

    # Generate test data:
    dx, x_fine, y_fine, y_prime_fine, y_coeff = get_test_data()

    # Plot results:
    plt.figure()
    plt.plot(
        x_fine.to(rc.cpu),
        y_fine.to(rc.cpu),
        "k--",
        label="Reference data",
        zorder=10,
    )
    plt.plot(
        x_fine.to(rc.cpu),
        y_prime_fine.to(rc.cpu),
        "k:",
        label="Reference derivative",
        zorder=10,
    )
    for deriv in range(5):
        plt.plot(
            x_fine.to(rc.cpu),
            Interpolator(x_fine, dx, deriv)(y_coeff).to(rc.cpu),
            label=f"Interpolant (deriv: {deriv})",
            lw=3,
        )
    plt.axhline(0, color="k", ls="dotted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
