"""
Integration tests for batch-by-run reduction via ManualReductionModel.

These tests exercise the full path: parse a run specification, build
(label, nexus_file) job pairs, reduce with reduce_runs(), and verify
that multiple workspaces are stored and switchable.

Requires access to /HFIR/HB2B/IPTS-22731/nexus/.
"""

import os
import tempfile

import numpy as np
import pytest

from pyrs.interface.manual_reduction.manual_reduction_model import ManualReductionModel

NEXUS_DIR = "/HFIR/HB2B/IPTS-22731/nexus"
RUN_A = 1017
RUN_B = 1018
NEXUS_A = os.path.join(NEXUS_DIR, f"HB2B_{RUN_A}.nxs.h5")
NEXUS_B = os.path.join(NEXUS_DIR, f"HB2B_{RUN_B}.nxs.h5")


def hfir_available():
    return os.path.exists(NEXUS_A) and os.path.exists(NEXUS_B)


pytestmark = pytest.mark.skipif(not hfir_available(), reason="HFIR archive not accessible")


# ---------------------------------------------------------------------------
# reduce_runs — integration tests against real HB2B data
#
# The pure-logic tests for parse_run_numbers / is_run_specification that used to
# live here (no HFIR access needed) moved to
# tests/unit/pyrs/interface/test_manual_reduction_runspec.py, alongside the other
# tests of the same two functions — see plans/test-framework.md.
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory(prefix="pyrs_batch_test_") as d:
        yield d


@pytest.fixture()
def model():
    """A fresh ManualReductionModel for each test."""
    return ManualReductionModel()


@pytest.mark.integration
def test_reduce_runs_two_files_returns_two_labels(model, output_dir):
    """reduce_runs on two NeXus files returns exactly two labels."""
    jobs = [(str(RUN_A), NEXUS_A), (str(RUN_B), NEXUS_B)]
    labels = model.reduce_runs(jobs, output_dir, progressbar=None)
    assert labels == [str(RUN_A), str(RUN_B)]


@pytest.mark.integration
def test_reduce_runs_stores_both_workspaces(model, output_dir):
    """Both workspaces are stored and accessible via set_current_run."""
    jobs = [(str(RUN_A), NEXUS_A), (str(RUN_B), NEXUS_B)]
    labels = model.reduce_runs(jobs, output_dir, progressbar=None)

    # Should be able to switch to each label without error
    for label in labels:
        model.set_current_run(label)
        sub_runs = model.get_sub_runs()
        assert len(sub_runs) > 0, f"Run {label} has no sub-runs after reduction"


@pytest.mark.integration
def test_reduce_runs_first_run_is_current_after_reduction(model, output_dir):
    """After reduce_runs, the first run's workspace is active."""
    jobs = [(str(RUN_A), NEXUS_A), (str(RUN_B), NEXUS_B)]
    model.reduce_runs(jobs, output_dir, progressbar=None)

    # Sub-runs for the first run should be immediately available
    sub_runs = model.get_sub_runs()
    assert len(sub_runs) > 0


@pytest.mark.integration
def test_reduce_runs_output_files_created(model, output_dir):
    """A .h5 project file is written for each reduced run."""
    jobs = [(str(RUN_A), NEXUS_A), (str(RUN_B), NEXUS_B)]
    model.reduce_runs(jobs, output_dir, progressbar=None)

    h5_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
    assert len(h5_files) == 2, f"Expected 2 .h5 files, found: {h5_files}"


@pytest.mark.integration
def test_reduce_runs_powder_pattern_is_sensible(model, output_dir):
    """Reduced powder pattern has finite intensities in a plausible 2theta range."""
    jobs = [(str(RUN_A), NEXUS_A)]
    model.reduce_runs(jobs, output_dir, progressbar=None)

    sub_runs = model.get_sub_runs()
    vec_2theta, vec_intensity = model.get_powder_pattern(sub_runs[0])

    assert vec_2theta is not None and len(vec_2theta) > 0
    finite = vec_intensity[np.isfinite(vec_intensity)]
    assert finite.max() > 0, "All intensities are zero or NaN"
    # HB2B typically operates between 50° and 130°
    assert vec_2theta.min() > 40 and vec_2theta.max() < 140


@pytest.mark.integration
def test_reduce_runs_single_run_succeeds(model, output_dir):
    """reduce_runs works with a single-item job list."""
    jobs = [(str(RUN_A), NEXUS_A)]
    labels = model.reduce_runs(jobs, output_dir, progressbar=None)
    assert labels == [str(RUN_A)]
    assert len(model.get_sub_runs()) > 0


@pytest.mark.integration
def test_reduce_runs_empty_job_list(model, output_dir):
    """reduce_runs with no jobs returns an empty list and leaves no workspaces."""
    labels = model.reduce_runs([], output_dir, progressbar=None)
    assert labels == []
