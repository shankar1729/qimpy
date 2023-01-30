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

    # Callback function to analyze trajetcory:
    def analyze(dynamics: qp.geometry.Dynamics, i_iter: int) -> None:
        positions = dynamics.system.ions.positions
        lattice = dynamics.system.lattice
        dpos = positions[1] - positions[0]
        dpos -= (dpos + 0.5).floor()  # periodic wrap (minimum image convention)
        bond_distance = (lattice.Rbasis @ dpos).norm().item()
        bond_distances.append(bond_distance)

    bond_distances = []  # to be populated by analyze()
    ps_path = "../../../../../JDFTx/build_testing/pseudopotentials/SG15"
    system = qp.System(
        lattice=dict(system="cubic", modification="face-centered", a=14.0),
        ions=dict(
            pseudopotentials=os.path.join(ps_path, "$ID_ONCV_PBE.upf"),
            coordinates=[["H", 0.0, 0.0, 0.0], ["H", 0.3, 0.2, 1.4]],
            fractional=False,
        ),
        electrons=dict(
            basis=dict(real_wavefunctions=True),
            fillings=dict(smearing=None),
        ),
        geometry=dict(
            dynamics=dict(
                dt=float(Unit(1.0, "fs")),
                n_steps=200,
                thermostat=dict(berendsen={}),
                t_damp_T=Unit(10, "fs"),
                report_callback=analyze,
            ),
        ),
    )
    system.run()
    qp.rc.report_end()
    qp.utils.StopWatch.print_stats()

    # Visualize trajectory:
    plt.plot(bond_distances)
    plt.xlabel("Time step")
    plt.ylabel("Bond length [$a_0$]")
    plt.savefig("bond-length.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
