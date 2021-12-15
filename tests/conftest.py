import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    """Add timing to test result if passed."""
    outcome = yield
    category, short_letter, verbose_word = outcome.get_result()
    if category == "passed":
        verbose_word = f"{verbose_word} in {report.duration:.2f}s"
        outcome.force_result((category, short_letter, verbose_word))
