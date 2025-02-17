import pytest
from typing import Literal

# Mark all tests in this directory to use anyio for async testing
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="session")
def anyio_backend() -> Literal["asyncio"]:
    """Configure anyio to use asyncio backend for async tests.

    This fixture is required by pytest-anyio and sets asyncio as the
    preferred async backend for all tests.

    Returns:
        Literal["asyncio"]: The async backend to use for testing
    """
    return "asyncio"
