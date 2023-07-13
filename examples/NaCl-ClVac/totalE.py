# Calculation analogous to totalE.yaml with Python input:
import os
import numpy as np
import qimpy as qp

qp.io.log_config()  # default set up to log from MPI head alone
qp.log.info("Using QimPy " + qp.__version__)
qp.rc.init()

# Create lattice object explicitly (eg. shared between two systems)
n_sup = 2  # number of unit cells in each dimension
lattice = qp.lattice.Lattice(
    modification="face-centered",
    system="cubic",
    a=10.74,  # bohrs, for unit cell
    scale=n_sup,
)  # scaled for supercell

# Ion parameters:
ps_path = "../../../../JDFTx/build_testing/pseudopotentials/SG15"
coords_mesh_1d = np.arange(n_sup) * (1.0 / n_sup)
coords_mesh = (
    np.stack(np.meshgrid(*((coords_mesh_1d,) * 3), indexing="ij")).reshape(3, -1).T
)
coordinates = [["Na", *tuple(coords + 0.5 / n_sup)] for coords in coords_mesh]
coordinates.extend(
    [["Cl", *tuple(coords)] for coords in coords_mesh if np.linalg.norm(coords)]
)  # omit Cl at (0,0,0)

system = qp.dft.System(
    lattice=lattice,
    ions={
        "pseudopotentials": os.path.join(ps_path, "$ID_ONCV_PBE.upf"),
        "coordinates": coordinates,
    },
    electrons={
        "basis": {"real-wavefunctions": True},
        "xc": {"functional": "gga-xc-pbe"},
    },
)
system.run()

qp.rc.report_end()
qp.profiler.StopWatch.print_stats()
