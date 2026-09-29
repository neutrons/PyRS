"""A5: are the scan-point-family datasets the landed writer emits actually resizable?

Spec 04c's architecture rests on one sentence -- that a tail-append "reduces to
code that already exists", the ``cur = shape[0]; resize(cur+N); arr[cur:] = ...``
shape. ``a4_h5py_nexusformat_append.py`` confirmed that shape works on reopened,
file-backed ``NXfield``s, and 04c's Follow-up 1 F1.1 recorded the design as
sound on that evidence.

That probe built **its own** fixtures, and gave every one of them
``maxshape=(None,)`` + ``chunks``. It therefore established that the mechanism
works *on a resizable dataset* -- which is not the same claim as "the datasets
the NXstress writer emits are resizable". An HDF5 dataset created without
``maxshape`` is contiguous and cannot be extended at all, by any mechanism.

This probe asks the question the other one could not: build a real ``NXentry``
through ``NXstress.write``, reopen it, and **traverse every dataset in it**,
reporting shape / maxshape / chunks for each. Per ``probes/README.md``'s
failure-modes table -- "reading source tells you what is written, not what it
means" -- the artifact is built and walked, not grepped.

Claims under test
-----------------
1. ``04c-nxstress-append.md:92-99`` -- "the common case that actually needs to
   work ... reduces to code that already exists ... No new insertion-position
   machinery is needed for the case this spec actually delivers."
2. ``04c-nxstress-append.md:118-126`` -- "this spec covers the full peak-index
   family and the full scan-point family ... 'grown' means **tail-appended** ...
   added after each dataset's current end, in lockstep across every group in the
   family."
3. ``04c-nxstress-append.md:196-205`` -- the Scope bullets asserting a
   tail-append path for ``_input_data.py``, ``_sample.py``'s per-scan-point logs
   and ``_fit.py::_Diffractogram.init_group``.
4. ``04c-nxstress-append.md:266-278`` -- "``_Diffractogram.init_group`` -- accept
   an existing ``NXdata`` group; tail-append ``diffractogram`` /
   ``diffractogram_errors`` / ``scan_point`` after the current end" and the
   equivalent for ``_sample.py``.

Result when first run, against the writer as 04b left it
--------------------------------------------------------
**86 datasets: 32 extendable, 54 fixed-size.** Every named scan-point-family
dataset but ``detector_counts`` and ``SAMPLE_DESCRIPTION/scan_point`` was
``maxshape=(3,) chunks=None`` -- contiguous, and extendable by nothing::

    /entry/start_time:                           FIXED
    /entry/end_time:                             FIXED
    /entry/input_data/scan_point:                FIXED
    /entry/instrument/monochromator/wavelength:  FIXED
    /entry/SAMPLE_DESCRIPTION/vx, vy, vz:        FIXED
    /entry/SAMPLE_DESCRIPTION/logs/* (all 4):    FIXED
    /entry/FIT/DIFFRACTOGRAM/XAXIS:              FIXED
    /entry/FIT/DIFFRACTOGRAM/diffractogram:      FIXED
    /entry/FIT/DIFFRACTOGRAM/diffractogram_errors: FIXED
    /entry/FIT/DIFFRACTOGRAM/scan_point:         FIXED

while the peak-index family was 22 of 24 extendable (the two exceptions being
the ``title`` and ``center_type`` scalars, which are per-entry and correct as
they are). So the claims above held for the peak-index family and were false
for the scan-point family. See 04c's Follow-up 2 F2.1.

**What it is now.** The 04c PR added ``_definitions.growable`` and applied it at
every one of those sites, so the probe reports all-extendable today -- its value
has shifted from evidence of the finding to a **regression detector**: a
per-scan-point field added later without ``growable`` shows up here as FIXED.
The same guarantee is pinned on every commit by
``tests/unit/pyrs/utilities/NXstress/test_append.py::TestWriterEmitsResizableDatasets``,
which sweeps the entry rather than naming fields; this probe is retained because
it prints the full per-dataset dump the test only asserts over.

Run: ``pixi run python plans/NXstress-prod/probes/a5_scan_point_family_resizable.py``
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.utilities.NXstress.NXstress import NXstress  # noqa: E402
from tests.util.peak_collection_helpers import createPeakCollection as _peak_fixture  # noqa: E402


def _unwrap_fixture(fixture, /, *args, **kwargs):
    """Call a pytest generator-fixture's underlying factory directly."""
    return next(fixture.__wrapped__())(*args, **kwargs)


def _conftest():
    spec = importlib.util.spec_from_file_location(
        "_nxstress_conftest", REPO / "tests" / "unit" / "pyrs" / "utilities" / "NXstress" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def walk(group, prefix: str = "") -> list[tuple[str, tuple, tuple, tuple | None]]:
    """Every dataset under `group`, as (path, shape, maxshape, chunks)."""
    found = []
    for name, item in group.items():
        path = f"{prefix}/{name}"
        if isinstance(item, h5py.Dataset):
            found.append((path, item.shape, item.maxshape, item.chunks))
        elif isinstance(item, h5py.Group):
            found.extend(walk(item, path))
    return found


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: is the landed writer's output extendable, dataset by dataset?")
    print("=" * 78)

    conftest = _conftest()
    ws = _unwrap_fixture(
        conftest.minimal_HidraWorkspace,
        name="probe_ws",
        n_subruns=3,
        with_instrument=True,
        with_masks=True,
        mask_names=("mask_a",),
        with_raw_counts=True,
    )
    peaks = _unwrap_fixture(
        _peak_fixture,
        peak_tag="Fe110",
        peak_profile="PseudoVoigt",
        background_type="Linear",
        wavelength=1.452,
        projectfilename="probe.h5",
        runnumber=1234,
        N_subrun=3,
    )

    tmp = Path(tempfile.mkdtemp(prefix="a5_resizable_"))
    path = tmp / "probe.nxs"
    with NXstress(path, "w") as nxs:
        nxs.write([ws], [[peaks]])

    with h5py.File(path, "r") as f:
        datasets = walk(f["entry"], "/entry")

    extendable = [d for d in datasets if d[2] and d[2][0] is None]
    fixed = [d for d in datasets if not (d[2] and d[2][0] is None)]

    report(
        "every dataset the writer emits, counted by whether its first axis is extendable",
        f"{len(datasets)} dataset(s) total: {len(extendable)} extendable, {len(fixed)} FIXED-SIZE",
    )

    # The scan-point family, named by 04c's own table at :107-126 plus the two
    # entry-level fields that table omits. A tail-append must grow every one.
    scan_point_family = (
        "/entry/start_time",
        "/entry/end_time",
        "/entry/input_data/detector_counts",
        "/entry/input_data/scan_point",
        "/entry/instrument/monochromator/wavelength",
        "/entry/SAMPLE_DESCRIPTION/scan_point",
        "/entry/SAMPLE_DESCRIPTION/vx",
        "/entry/SAMPLE_DESCRIPTION/vy",
        "/entry/SAMPLE_DESCRIPTION/vz",
    )
    by_path = {d[0]: d for d in datasets}
    lines = []
    for name in scan_point_family:
        if name not in by_path:
            lines.append(f"{name}: ABSENT")
            continue
        _, shape, maxshape, chunks = by_path[name]
        verdict = "extendable" if maxshape and maxshape[0] is None else "FIXED"
        lines.append(f"{name}: shape={shape} maxshape={maxshape} chunks={chunks} -> {verdict}")
    report(
        "claims 2-4: is every named scan-point-family dataset tail-appendable?",
        "\n          ".join(lines),
    )

    dgram = [d for d in datasets if "DIFFRACTOGRAM" in d[0]]
    report(
        "claim 4: _Diffractogram's datasets",
        "\n          ".join(
            f"{p}: shape={s} maxshape={m} chunks={c} -> {'extendable' if m and m[0] is None else 'FIXED'}"
            for p, s, m, c in sorted(dgram)
        ),
    )

    logs = sorted(d for d in datasets if d[0].startswith("/entry/SAMPLE_DESCRIPTION/logs/"))
    report(
        "claim 3: _sample.py's retained per-scan-point logs",
        f"{len(logs)} log field(s); "
        + (
            "all FIXED"
            if all(not (m and m[0] is None) for _, _, m, _ in logs)
            else "mixed -- see per-field dump below"
        )
        + "\n          "
        + "\n          ".join(f"{p.rsplit('/', 1)[1]}: maxshape={m} chunks={c}" for p, _, m, c in logs),
    )

    peak_family = sorted(d for d in datasets if d[0].startswith(("/entry/peaks/", "/entry/FIT/peak_parameters/")))
    n_ext = sum(1 for _, _, m, _ in peak_family if m and m[0] is None)
    report(
        "claim 1: by contrast, the PEAK-INDEX family -- the half the claim is true of",
        f"{n_ext} of {len(peak_family)} extendable "
        f"(the {len(peak_family) - n_ext} fixed one(s): "
        f"{[p for p, _, m, _ in peak_family if not (m and m[0] is None)]})",
    )

    print("\n" + "-" * 78)
    print("  A FIXED dataset is contiguous HDF5 and cannot be resized by any mechanism,")
    print("  so a per-scan-point field reported FIXED above can never be appended to.")
    scan_point_fixed = [
        name
        for name in scan_point_family + tuple(p for p, _, _, _ in dgram) + tuple(p for p, _, _, _ in logs)
        if name in by_path and not (by_path[name][2] and by_path[name][2][0] is None)
    ]
    if scan_point_fixed:
        print(f"  THIS RUN FOUND {len(scan_point_fixed)}: {scan_point_fixed}")
        print("  Pass `**growable(rank)` at their `NXfield(...)`. See 04c's Follow-up 2 F2.1.")
    else:
        print("  This run found none: every scan-point-family dataset is extendable.")
    print("-" * 78)

    print(f"\n  (artifact left at {path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
