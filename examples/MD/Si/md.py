import os
import torch
import matplotlib.pyplot as plt
import qimpy as qp
from qimpy.utils import Unit


def main() -> None:
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()
    torch.manual_seed(1234)

    # Callback function to analyze trajectory:
    def analyze(dynamics: qp.geometry.Dynamics, i_iter: int) -> None:
        energies.append(float(dynamics.system.energy))
        volumes.append(dynamics.system.lattice.volume)

    energies = []  # to be populated by analyze()
    volumes = []  # to be populated by analyze()
    ps_path = "../../../../../JDFTx/build_testing/pseudopotentials/SG15"
    system = qp.System(
        lattice=dict(system="cubic", a=float(Unit(5.43, "Å")), movable=False),
        ions=dict(
            pseudopotentials=os.path.join(ps_path, "$ID_ONCV_PBE.upf"),
            coordinates=[
                ["Si", 0.000, 0.000, 0.000],
                ["Si", 0.003, 0.502, 0.501],
                ["Si", 0.503, 0.001, 0.502],
                ["Si", 0.501, 0.503, 0.002],
                ["Si", 0.249, 0.251, 0.248],
                ["Si", 0.250, 0.748, 0.751],
                ["Si", 0.751, 0.248, 0.749],
                ["Si", 0.748, 0.752, 0.249],
            ],
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
                thermostat="berendsen",
                t_damp_T=Unit(10, "fs"),
                report_callback=analyze,
            ),
        ),
    )
    system.run()
    qp.rc.report_end()
    qp.utils.StopWatch.print_stats()

    if qp.rc.is_head:
        # Visualize trajectory:
        plt.figure()
        plt.xlabel("Time step")
        plt.ylabel("Energy [$E_h$]")
        plt.plot(energies)
        plt.savefig("energy.pdf", bbox_inches="tight")

        plt.figure()
        plt.xlabel("Time step")
        plt.ylabel("Volume [$a_0^3$]")
        plt.plot(volumes)
        plt.savefig("volume.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
