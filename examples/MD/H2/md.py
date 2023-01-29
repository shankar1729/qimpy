import os
import qimpy as qp
from qimpy.utils import Unit


def main():
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    ps_path = "../../../../../JDFTx/build_testing/pseudopotentials/SG15"
    system = qp.System(
        lattice=dict(system="cubic", a=6.0),
        ions=dict(
            pseudopotentials=os.path.join(ps_path, "$ID_ONCV_PBE.upf"),
            coordinates=[["H", 0.0, 0.0, 0.0], ["H", 0.1, 0.2, 1.4]],
            fractional=False,
        ),
        electrons=dict(
            basis=dict(real_wavefunctions=True),
            fillings=dict(smearing=None),
        ),
        geometry=dict(
            dynamics=dict(
                dt=float(Unit(0.5, "fs")),
                n_steps=100,
            ),
        ),
    )
    system.run()

    qp.rc.report_end()
    qp.utils.StopWatch.print_stats()


if __name__ == "__main__":
    main()
