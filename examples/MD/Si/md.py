import torch
import matplotlib.pyplot as plt
import qimpy as qp
from qimpy.io import Unit


def main() -> None:
    qp.io.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()
    torch.manual_seed(1234)

    # Callback function to analyze trajectory:
    def analyze(dynamics: qp.dft.geometry.Dynamics, i_iter: int) -> None:
        energies.append(float(dynamics.system.energy))
        volumes.append(dynamics.system.lattice.volume / Unit.MAP["Angstrom"] ** 3)
        pressures.append(Unit.convert(dynamics.P, "bar").value)
        temperatures.append(Unit.convert(dynamics.T, "K").value)

    energies = []  # to be populated by analyze()
    volumes = []  # to be populated by analyze()
    pressures = []  # to be populated by analyze()
    temperatures = []  # to be populated by analyze()

    # Construct coordinates input:
    positions = [
        [0.000, 0.000, 0.000],
        [0.003, 0.502, 0.501],
        [0.503, 0.001, 0.502],
        [0.501, 0.503, 0.002],
        [0.249, 0.251, 0.248],
        [0.250, 0.748, 0.751],
        [0.751, 0.248, 0.749],
        [0.748, 0.752, 0.249],
    ]
    velocities = [
        [-9.446e-05, +6.233e-05, +6.401e-05],
        [+7.916e-05, -9.725e-06, -9.636e-05],
        [-9.603e-05, -4.774e-05, +1.696e-05],
        [-1.114e-04, -7.478e-05, +1.372e-04],
        [+1.999e-04, +1.307e-04, -4.277e-04],
        [-1.040e-04, -5.010e-05, +1.226e-05],
        [+7.844e-05, -7.251e-06, +1.675e-04],
        [+4.838e-05, -3.422e-06, +1.261e-04],
    ]
    coordinates = [
        ["Si", *tuple(pos), {"v": v}] for pos, v in zip(positions, velocities)
    ]

    system = qp.dft.System(
        lattice=dict(system="cubic", a=float(Unit(5.43, "Å")), movable=True),
        ions=dict(
            pseudopotentials="SG15/$ID_ONCV_PBE.upf",
            coordinates=coordinates,
        ),
        electrons=dict(
            k_mesh=dict(size=[2, 2, 2], offset=[0.5, 0.5, 0.5]),
            basis=dict(ke_cutoff=10.0),
            fillings=dict(smearing=None),
            scf=dict(energy_threshold=1e-6),
        ),
        geometry=dict(
            dynamics=dict(
                dt=float(Unit(2.0, "fs")),
                n_steps=100,
                # thermostat=dict(berendsen=dict(B0=Unit(95.0, "GPa"))),
                thermostat="nose-hoover",
                t_damp_T=Unit(10, "fs"),
                t_damp_P=Unit(100, "fs"),
                report_callback=analyze,
            ),
        ),
        checkpoint="md.h5",
    )
    system.run()
    qp.rc.report_end()
    qp.profiler.StopWatch.print_stats()

    if qp.rc.is_head:
        # Visualize trajectory properties:
        for quantity, ylabel, filename in (
            (energies, "Energy [$E_h$]", "energy.pdf"),
            (volumes, r"Volume [$\AA^3$]", "volume.pdf"),
            (pressures, "$P$ [bar]", "pressure.pdf"),
            (temperatures, "$T$ [K]", "temperature.pdf"),
        ):
            plt.figure()
            plt.xlabel("Time step")
            plt.ylabel(ylabel)
            plt.plot(quantity)
            plt.savefig(filename, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
