"""
Regenerate `tests/data/HB2B_1017.h5` and `tests/data/HB2B_1017_NoMask_Gold.h5` from the raw run.

The project file from 2019 predates PR #196 (#193): its sub-run logs, the arm angle among them, were averaged across
scan-index boundaries, so sub-runs 1 and 2 recorded 83.33 and 88.75 degrees where the arm stood at 70 and 80 (see
docs/ground_truths.md). It also records no calibration, which #1141 needs for a project to be saved. This script
converts `tests/data/HB2B_1017.nxs.h5` with today's code and reduces it without a mask or a calibration file, so the
file records the nominal calibration of the run's monochromator. It keeps the raw counts, which the powder-pattern
tests reduce.

The gold file is the powder pattern of each sub-run that `tests/integration/test_powder_pattern.py` checks: the raw
counts of the new project file reduced by `HB2BReductionManager` with 1000 bins, no mask and no normalisation by
duration.

Run from the repository root (never collected by pytest)::

    pixi run python tests/scripts/make_HB2B_1017.py
"""

from pathlib import Path

import h5py
import numpy as np

from pyrs.core.instrument_geometry import HidraSetup
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from pyrs.core.reduction_manager import HB2BReductionManager
from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

RAW = Path("tests/data/HB2B_1017.nxs.h5")
PROJECT = Path("tests/data/HB2B_1017.h5")
GOLD = Path("tests/data/HB2B_1017_NoMask_Gold.h5")


def write_project() -> None:
    """Convert and reduce run 1017 into the project file, raw counts included."""
    workspace = NeXusConvertingApp(str(RAW), None).convert()
    reducer = ReductionApp()
    reducer.load_hidra_workspace(workspace)
    reducer.reduce_data(sub_runs=None, instrument_file=None, calibration_file=None, mask=None, num_bins=1000)
    reducer.save_diffraction_data(str(PROJECT), ignore_raw_counts=False)
    # reduction's save writes no instrument geometry (DD-34); record the one the reduction used, which the
    # powder-pattern tests need to reduce the raw counts again
    project = HidraProjectFile(str(PROJECT), HidraProjectFileMode.READWRITE)
    project.write_instrument_geometry(HidraSetup(workspace.get_instrument_setup()))
    project.save(verbose=False)


def write_gold() -> None:
    """Reduce the project file's raw counts as the powder-pattern tests do, and store each sub-run's pattern."""
    workspace = HidraWorkspace("gold")
    project = HidraProjectFile(str(PROJECT), HidraProjectFileMode.READONLY)
    workspace.load_hidra_project(project, load_raw_counts=True, load_reduced_diffraction=False)
    project.close()
    service = HB2BReductionManager()
    service.init_session(session_name="gold", hidra_ws=workspace)
    service.reduce_diffraction_data(
        "gold",
        False,
        1000,
        sub_run_list=None,
        mask=None,
        mask_id=None,
        vanadium_counts=None,
        normalize_by_duration=False,
    )
    with h5py.File(GOLD, "w") as gold:
        for sub_run in workspace.get_sub_runs():
            pattern = service.get_reduced_diffraction_data("gold", sub_run)  # 2theta, intensity, variance
            group = gold.create_group(str(sub_run))
            group.create_dataset("x", data=np.asarray(pattern[0]))
            group.create_dataset("y", data=np.asarray(pattern[1]))


def main() -> None:
    """Write both files and report what the project file records."""
    write_project()
    write_gold()
    project = HidraProjectFile(str(PROJECT), HidraProjectFileMode.READONLY)
    applied = project.read_applied_calibration()
    project.close()
    print(f"{PROJECT}: calibration source {applied.source.value}, hash {applied.content_hash}")
    print(f"{GOLD}: written")


if __name__ == "__main__":
    main()
