"""Infrastructure tests for the installable package."""

import analytics_foundations


def test_package_imports() -> None:
    assert analytics_foundations.__version__ == "0.1.0"

