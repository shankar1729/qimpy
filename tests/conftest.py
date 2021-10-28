import qimpy as qp
import pytest


@pytest.fixture(scope="session")
def rc() -> qp.utils.RunConfig:
    """Generate shared RunConfig for simple tests without a complete qp.System.
    This fully initializes the process dimensions with arbitrary task counts."""
    rc = qp.utils.RunConfig()
    rc.provide_n_tasks(0, 1)  # replicas
    rc.provide_n_tasks(1, 2)  # k-points
    rc.provide_n_tasks(2, 2)  # bands/basis
    return rc
