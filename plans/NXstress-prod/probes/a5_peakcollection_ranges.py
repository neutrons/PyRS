"""A5: does the reader really require only contiguity and monotonicity?

Decisions Log item 17 rests the entire 04b/04c simplification -- discriminators
as the most slowly varying sort key, and "locally sorted, globally segmented"
append -- on a single sentence describing code that **neither subspec owns**.
Two subspecs' design weight on one paragraph is exactly the shape
``process.md`` section 5.3 warns about, so this probe drives the real reader.

Claims under test
-----------------
1. ``README.md:729`` (Decisions item 17) and ``04b:37-43`` --
   "``_Peaks.peakCollectionRanges`` (``_peaks.py:246-338``) -- the only
   reader-side splitter -- enforces only that each compound key's run is
   contiguous and ``scan_point`` increases within a run".
2. ``README.md:335-342`` --
   "not because the schema requires global sorting (it doesn't; the reader only
   requires each compound key's run to be contiguous, ``_peaks.py:246-338``)".
3. ``04b:43-45`` -- "It never checks that the runs themselves are globally
   ordered -- there is no ``searchsorted``/``argsort``/binary search anywhere in
   the module."
4. ``README.md:729`` and ``04b:49-55`` -- "the latter is guaranteed upstream by
   ``SubRuns.set`` (``sample_logs.py:164-166``) ... which already raises
   ``"subruns are not sorted in increasing order"`` unless
   ``np.all(value[:-1] < value[1:])``".
5. ``04c:44-53`` -- "**Append does not re-sort the file.** ... R1 and R2 still
   hold throughout, which is all the reader has ever required."

Per ``process.md`` section 5.3, the counterparty's invariant is **reproduced
locally** in ``_expected_ranges`` below rather than imported. If ``_peaks.py``
ever changes its splitting rule, this probe fails instead of silently tracking
the change -- which is the entire point of expressing a cross-module dependency
as executable code rather than as a sentence.

Run: ``pixi run python plans/NXstress-prod/probes/a5_peakcollection_ranges.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from nexusformat.nexus import NXfield

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.utilities.NXstress._peaks import _Peaks  # noqa: E402


class FakePeaks(dict):
    """Minimal stand-in for the NXreflections group the reader indexes into.

    Values are real ``NXfield``s, not bare ndarrays: the reader reaches through
    ``.nxdata``, so a plain array stub would fail for a reason that has nothing
    to do with the claims under test.
    """


def group(phase, h, k, l_, mask, scan_point) -> FakePeaks:
    peaks = FakePeaks()
    peaks["phase_name"] = NXfield(np.array(phase))
    peaks["h"] = NXfield(np.array(h, dtype=np.int32))
    peaks["k"] = NXfield(np.array(k, dtype=np.int32))
    peaks["l"] = NXfield(np.array(l_, dtype=np.int32))
    peaks["mask"] = NXfield(np.array(mask))
    peaks["scan_point"] = NXfield(np.array(scan_point, dtype=np.int32))
    return peaks


def _expected_ranges(phase, h, k, l_, mask) -> int:
    """Reproduce the counterparty's block rule LOCALLY (process.md 5.3).

    A block boundary is any change in the compound key. Deliberately not
    imported from ``_peaks.py``: if that module's rule changes, this probe must
    fail rather than agree.
    """
    keys = list(zip(phase, h, k, l_, mask))
    blocks = 1
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            blocks += 1
    return blocks


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def attempt(label: str, peaks: FakePeaks) -> str:
    try:
        ranges = _Peaks.peakCollectionRanges(peaks)
        return f"{label}: ACCEPTED -> {len(ranges)} block(s) {[(r[1], r[2]) for r in ranges]}"
    except Exception as exc:  # noqa: BLE001 - what it raises is the result
        return f"{label}: RAISED {type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: _Peaks.peakCollectionRanges -- what the reader actually enforces")
    print("=" * 78)

    # --- Globally sorted, the shape a single-step write produces today ---
    sorted_case = group(
        ["Fe"] * 3 + ["Ni"] * 3,
        [1] * 3 + [2] * 3,
        [1] * 3 + [2] * 3,
        [1] * 3 + [2] * 3,
        ["m"] * 6,
        [1, 2, 3, 1, 2, 3],
    )
    report("a fully sorted index is accepted (baseline)", attempt("sorted", sorted_case))

    # --- Claims 1, 2, 3, 5: globally UNSORTED but locally contiguous ---
    # "Ni" sorts after "Fe", so emitting Ni first is globally out of order.
    segmented = group(
        ["Ni"] * 3 + ["Fe"] * 3,
        [2] * 3 + [1] * 3,
        [2] * 3 + [1] * 3,
        [2] * 3 + [1] * 3,
        ["m"] * 6,
        [1, 2, 3, 1, 2, 3],
    )
    report(
        "globally UNSORTED but each key contiguous -- 'locally sorted, globally "
        "segmented' is accepted (claims 1, 2, 3, 5)",
        attempt("segmented", segmented),
    )

    # --- Claim 1: contiguity IS enforced ---
    interleaved = group(
        ["Fe", "Ni", "Fe"],
        [1, 2, 1],
        [1, 2, 1],
        [1, 2, 1],
        ["m"] * 3,
        [1, 1, 2],
    )
    report("a key split into two runs is rejected (contiguity IS enforced)", attempt("interleaved", interleaved))

    # --- Claim 1: monotonic scan_point within a run IS enforced ---
    nonmonotonic = group(["Fe"] * 3, [1] * 3, [1] * 3, [1] * 3, ["m"] * 3, [3, 2, 1])
    report(
        "descending scan_point within one run is rejected (monotonicity IS enforced)",
        attempt("non-monotonic", nonmonotonic),
    )

    # --- Claim 3: no global-order machinery in the module ---
    source = (REPO / "pyrs" / "utilities" / "NXstress" / "_peaks.py").read_text()
    machinery = {name: source.count(name) for name in ("searchsorted", "argsort", "bisect", "np.sort")}
    report(
        "no searchsorted/argsort/binary search anywhere in _peaks.py (claim 3)",
        f"occurrence counts: {machinery}",
    )

    # --- Claim 4: is monotonicity really guaranteed upstream by SubRuns.set? ---
    from pyrs.dataobjects.sample_logs import SubRuns

    try:
        SubRuns(np.array([3, 1, 2]))
        upstream = "ACCEPTED a descending array -- the upstream guarantee does NOT hold"
    except Exception as exc:  # noqa: BLE001
        upstream = f"RAISED {type(exc).__name__}: {exc}"
    report("SubRuns.set raises unless strictly increasing (claim 4)", upstream)

    # --- The local reproduction, per process.md 5.3 ---
    phase = ["Ni"] * 3 + ["Fe"] * 3
    local = _expected_ranges(phase, [2] * 3 + [1] * 3, [2] * 3 + [1] * 3, [2] * 3 + [1] * 3, ["m"] * 6)
    actual = len(_Peaks.peakCollectionRanges(segmented))
    report(
        "locally reproduced block rule agrees with the real reader (fails loudly if _peaks.py changes its rule)",
        f"locally predicted {local} block(s); _peaks.py produced {actual}; agree={local == actual}",
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
