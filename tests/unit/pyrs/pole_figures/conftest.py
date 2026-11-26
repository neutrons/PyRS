"""Unit-test setup: each test picks the worker count from the job size unless it sets HIDRA_PF_WORKERS itself."""

import pytest


@pytest.fixture(autouse=True)
def no_worker_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear HIDRA_PF_WORKERS, so a value set in the shell does not change what the unit tests run."""
    monkeypatch.delenv("HIDRA_PF_WORKERS", raising=False)
