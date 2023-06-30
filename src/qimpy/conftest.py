import pytest
import qimpy as qp


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    """Add timing to test result if passed."""
    outcome = yield
    category, short_letter, verbose_word = outcome.get_result()
    if category == "passed":
        verbose_word = f"{verbose_word} in {report.duration:.2f}s"
        outcome.force_result((category, short_letter, verbose_word))


@pytest.fixture(scope="session", autouse=True)
def init_run_config():
    qp.rc.init()


def pytest_collection_modifyitems(config, items):
    # Modify pytest-mpi to deselect instead of skip mpi/non-mpi tests based on mode:
    with_mpi = config.getoption("--with-mpi")
    deselect_mark = "mpi_skip" if with_mpi else "mpi"
    removed = []
    kept = []
    for item in items:
        if item.get_closest_marker(deselect_mark):
            removed.append(item)
        else:
            kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
