"""
pyrs/utilities/NXstress/_runs.py

The run index: every scan point of an entry is identified by `(run_number, scan_point)`.

A scan point is only meaningful *within* its run -- every HB2B run numbers its
scan points from 1 -- so two runs in one entry collide on scan point alone. The
strain/stress workflow is the ordinary case: three directions, each its own run,
each scanning points 1..N. This module supplies the outer index level.

Where a run number comes from:

- the `run_number` sample log, read **per scan point**, so a workspace merged
  from several runs in PyRS (CombineRuns) keeps every run's number;
- otherwise `Config["nxstress.default_run_number"]`, shared by every workspace
  that has no run number of its own.

On disk the index is two row-aligned columns, `SAMPLE_DESCRIPTION/run_number`
beside `scan_point` and `peaks/run_number` beside the peak index's own
`scan_point`, plus one attribute on the `peaks` group recording whether the run
is part of the **workspace key** -- the key that tells one input workspace's rows
from another's (see `BOUNDARY_ATTRIBUTE`).

Typical workflow:

1. `for_scan_points(logs, scan_points=...)` gives the run number of each scan point.
2. `single_run(logs)` asks whether a workspace holds exactly one run.
3. On read, `axis(sample_description)` and `is_boundary(peaks)` recover both.

The workspace key itself, which carries the run beside the discriminator values,
is `_workspace_key.WorkspaceKey`.

A file written before the run index carries neither column nor attribute. It is
read with every scan point at the configured default run number -- its scan
points are already unique, so nothing about the read changes.
"""

import numpy as np
from nexusformat.nexus import NXgroup

from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.utilities import config
from pyrs.utilities.unset import Unset

# Defined in `_definitions` so that `_discriminator` can reserve the column; part of
# this module's surface. The attribute is True exactly when every input workspace
# holds a single run, which is what lets two runs sharing every discriminator value
# -- two runs in one direction -- be read back as two workspaces; False only for an
# entry holding one workspace merged from several runs, read back together.
from ._definitions import RUN_BOUNDARY_ATTRIBUTE as BOUNDARY_ATTRIBUTE, RUN_NUMBER  # noqa: F401


def default_run_number() -> int:
    """The run number of a workspace that records none (`pyrs.utilities.config.default_run_number`)."""
    return config.default_run_number()


def for_scan_points(logs: SampleLogs, *, scan_points: np.ndarray | Unset = Unset) -> np.ndarray:
    """The run number of each scan point.

    Args:
        logs: One workspace's sample logs.
        scan_points: The scan points to look up, a subset of the workspace's own;
            every scan point of the workspace, in order, when not supplied.

    Returns:
        One integer per requested scan point: the `run_number` log's value, or the
        configured default for a workspace without that log.

    Raises:
        ValueError: If the `run_number` log holds a value that is not a whole
            number. Truncating it would be silent, and could make two runs collide.
    """
    points = logs.subruns.raw_copy() if scan_points is Unset else np.asarray(scan_points)
    if RUN_NUMBER not in logs:
        return np.full(len(points), default_run_number(), dtype=np.int64)
    values = np.asarray(logs[RUN_NUMBER] if scan_points is Unset else logs[(RUN_NUMBER, points)], dtype=float)
    if not np.all(np.isfinite(values) & (values == np.round(values))):
        raise ValueError(
            f"NXstress: the `run_number` log holds values that are not whole run numbers: "
            f"{sorted(set(values[~(np.isfinite(values) & (values == np.round(values)))].tolist()))}."
        )
    return values.astype(np.int64)


def workspace_pairs(logs: SampleLogs) -> list[tuple[int, int]]:
    """Every `(run, scan point)` pair of one workspace, in scan-point order."""
    return pairs(for_scan_points(logs), logs.subruns.raw_copy())


def merged_inputs(logss: list[SampleLogs]) -> list[int]:
    """Positions of the workspaces that hold more than one run -- merged from several in PyRS."""
    return [n for n, logs in enumerate(logss) if not single_run(logs)]


def block_runs(peaks: NXgroup, start: int, end: int) -> np.ndarray:
    """The run of each row in a block of a `peaks` group.

    The configured default throughout for an entry written before the run index.
    """
    if RUN_NUMBER not in peaks:
        return np.full(end - start, default_run_number(), dtype=np.int64)
    return np.asarray(peaks[RUN_NUMBER].nxdata[start:end]).astype(np.int64)


def run_numbers(logs: SampleLogs) -> set[int]:
    """The distinct run numbers a workspace's scan points carry."""
    return {int(run) for run in for_scan_points(logs)}


def single_run(logs: SampleLogs) -> bool:
    """Whether a workspace holds exactly one run -- that is, it was not merged from several."""
    return len(run_numbers(logs)) == 1


def is_boundary(peaks: NXgroup) -> bool:
    """Whether a `peaks` group records the run as part of its workspace key.

    False for an entry written before the run index, which records nothing.
    """
    if BOUNDARY_ATTRIBUTE not in peaks.attrs:
        return False
    value = peaks.attrs[BOUNDARY_ATTRIBUTE]
    return bool(np.asarray(getattr(value, "nxvalue", value)))


def axis(sample_description: NXgroup) -> np.ndarray:
    """The run number of every scan point of an entry, aligned with its `scan_point` axis.

    The configured default throughout for an entry written before the run index.
    """
    n_scan = len(sample_description["scan_point"].nxdata)
    if RUN_NUMBER not in sample_description:
        return np.full(n_scan, default_run_number(), dtype=np.int64)
    return np.asarray(sample_description[RUN_NUMBER].nxdata).astype(np.int64)


def pairs(runs: np.ndarray, scan_points: np.ndarray) -> list[tuple[int, int]]:
    """`(run, scan_point)` pairs, as plain integers so they hash and compare."""
    return [(int(run), int(point)) for run, point in zip(runs, scan_points)]
