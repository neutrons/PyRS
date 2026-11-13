"""Tests for `pyrs.core.reduction_manager`."""

import numpy as np
import pytest

from pyrs.core.reduction_manager import HB2BReductionManager
from tests.util.workspace_helpers import make_minimal_HidraWorkspace


@pytest.mark.parametrize(
    "sub_runs, sub_run", [((5, 6, 7), 5), ((1, 3, 4), 3)], ids=["not-starting-at-one", "with-a-gap"]
)
@pytest.mark.xfail(
    reason="DD-2: compares against sub-run `n - 1` rather than the previously reduced one",
    raises=IndexError,
    strict=True,
)
def test_setup_reduction_engine_any_sub_run_numbering(sub_runs: tuple, sub_run: int) -> None:
    """Whether to rebuild the 2theta map is decided against sub-run `n - 1`, which
    need not exist -- and, when it does, need not be the one reduced last.
    Recorded under DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    ws = make_minimal_HidraWorkspace(sub_runs=np.array(sub_runs), with_raw_counts=True)
    ws.set_sample_log("2theta", np.array(sub_runs), np.array([30.0, 30.0, 40.0]))

    # Act / Assert -- no exception
    HB2BReductionManager().setup_reduction_engine(ws, sub_run, None)
