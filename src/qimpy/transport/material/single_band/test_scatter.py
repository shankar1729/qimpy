from qimpy import rc
from qimpy.io import log_config
from qimpy.mpi import ProcessGrid
from . import SingleBand


def test_scatter():
    process_grid = ProcessGrid(rc.comm, "rk", (-1, 1))
    material = SingleBand(
        process_grid=process_grid,
        lattice=dict(
            periodic=[True, True, False], system=dict(name="hexagonal", a=4.651, c=15)
        ),
        kmesh=[1000, 1000, 1],
        mu=0.01,  # ~ 0.3 eV
        T=0.0002,  # ~ 60 K
        v=0.375,  # Graphene Fermi velocity
        scatter=dict(dE=0.0001, epsilon_bg=1.0, lambda_D=10.0),
    )
    print(material)


if __name__ == "__main__":
    log_config()
    rc.init()
    test_scatter()
