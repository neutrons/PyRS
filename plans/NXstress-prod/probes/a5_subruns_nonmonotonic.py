"""A5: can the *concatenated* scan-point axis be held by PyRS's own types?

04b mandates plain concatenation in workspace order -- "``concat(ws0.get_sub_runs(),
ws1.get_sub_runs(), ...)`` -- not a merge-and-sort" -- and its ``## Tests``
section specifies the case that makes the distinction bite: workspace A has
scan points ``[1, 3, 5]`` and workspace B has ``[2, 4, 6]``, so their *values*
interleave while their *positions* do not.

Nothing in 04b says what that concatenated axis is, as an object. It matters
because the only reason 04b's read side is simple is that it slices "the
scan-point family's concatenated arrays". If the concatenation has to pass
through a ``SampleLogs`` at any point, on either side, the interleaved case
cannot be represented at all -- and 04b's own specified test would be
unimplementable rather than merely awkward.

Claims under test
-----------------
1. ``04b:49-55`` / Decisions item 17 -- ``SubRuns.set`` raises unless strictly
   increasing. (Re-verified here in the *concatenation* shape, not the
   descending shape ``a5_peakcollection_ranges.py`` already covered: those are
   different failure modes and only this one is load-bearing for 04b's merge.)
2. **Unstated by 04b, and the reason this probe exists** -- whether the
   concatenated axis survives ``SubRuns`` construction at all, and therefore
   whether a merged ``HidraWorkspace`` can exist as an intermediate.
3. ``_sample.py:162`` -- ``_Sample.sampleLogsFromNexus`` does
   ``logs.subruns = SubRuns(scan_point)`` over the whole on-disk array, so a
   multi-workspace entry read *before* splitting hits claim 2 directly.
4. ``04b:336-339`` -- "reuse the merge logic ``HidraWorkspace.append_hidra_project``
   already implements in-memory (``pyrs/core/workspaces.py:574``)": whether that
   method takes an in-memory workspace, and whether it preserves scan points.

Per ``probes/README.md``'s failure-modes table, each claim reports the resulting
*state* rather than only the presence or absence of an exception -- "accepted"
is not "harmless", and for claim 4 the accepted-but-renumbered case is exactly
the silent one.

Run: ``pixi run python plans/NXstress-prod/probes/a5_subruns_nonmonotonic.py``
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.core.workspaces import HidraWorkspace  # noqa: E402
from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns  # noqa: E402


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def describe(arr: np.ndarray) -> str:
    """Characterise a whole array, never its first few values."""
    a = np.asarray(arr)
    return (
        f"n={a.size} min={a.min()} max={a.max()} ptp={np.ptp(a)} "
        f"distinct={len(np.unique(a))} strictly_increasing={bool(np.all(a[:-1] < a[1:]))}"
    )


def attempt(label: str, fn) -> str:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - what it raises is the result
        return f"{label}: RAISED {type(exc).__name__}: {exc}"
    return f"{label}: ACCEPTED -> {value}"


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: the concatenated scan-point axis vs. SubRuns / SampleLogs")
    print("=" * 78)

    a = np.array([1, 3, 5])
    b = np.array([2, 4, 6])
    concatenated = np.concatenate([a, b])

    report(
        "04b's own specified test case, as an array (not as its first few values)",
        f"A={a.tolist()} B={b.tolist()} -> concatenated {concatenated.tolist()}; {describe(concatenated)}",
    )

    # --- Claims 1 and 2: can the concatenation be a SubRuns? ---
    report(
        "SubRuns accepts each input workspace's own axis (both are strictly increasing)",
        f"{attempt('A', lambda: SubRuns(a).raw_copy().tolist())}; "
        f"{attempt('B', lambda: SubRuns(b).raw_copy().tolist())}",
    )
    report(
        "SubRuns accepts the workspace-order CONCATENATION of the two (claims 1, 2)",
        attempt("concatenated", lambda: SubRuns(concatenated).raw_copy().tolist()),
    )

    # A non-interleaving pair, to show the rejection is about the values and
    # not about concatenation per se -- otherwise the finding reads as broader
    # than it is.
    c = np.array([7, 8, 9])
    report(
        "the rejection is value-dependent, not intrinsic to concatenating",
        attempt("A ++ [7,8,9]", lambda: SubRuns(np.concatenate([a, c])).raw_copy().tolist()),
    )

    # `append_subruns` bypasses `set` -- worth pinning, because it means the
    # guard is a property of one entry point, not of the type.
    def via_append() -> str:
        s = SubRuns(a)
        s.append(b)
        return f"{s.raw_copy().tolist()} ({describe(s.raw_copy())})"

    report(
        "SubRuns.append bypasses the guard that SubRuns.set enforces",
        attempt("append", via_append),
    )

    # --- Claim 3: what the real reader does with such an axis ---
    def sample_logs_from_concatenated() -> str:
        logs = SampleLogs()
        logs.subruns = SubRuns(concatenated)
        return f"{logs.subruns.raw_copy().tolist()}"

    report(
        "_sample.py:162's `logs.subruns = SubRuns(scan_point)` over an unsplit axis (claim 3)",
        attempt("sampleLogsFromNexus shape", sample_logs_from_concatenated),
    )

    # --- Claim 4: is append_hidra_project reusable as 04b's merge? ---
    sig = inspect.signature(HidraWorkspace.append_hidra_project)
    src = inspect.getsource(HidraWorkspace.append_hidra_project)
    accepts_workspace = "HidraProjectFile" not in src
    renumbers = "np.arange(" in src and "append_subruns" in src
    report(
        "append_hidra_project takes an in-memory HidraWorkspace (claim 4, part a)",
        f"signature {sig}; type-checked against HidraProjectFile in body: "
        f"{'HidraProjectFile' in src}; accepts a workspace: {accepts_workspace}",
    )
    renumber_line = next((ln.strip() for ln in src.splitlines() if "np.arange(" in ln), "<not found>")
    report(
        "append_hidra_project preserves each input's own scan points (claim 4, part b)",
        f"renumbers via `{renumber_line}`: {renumbers} -- "
        f"so appended subruns become 1..N of the merged workspace, not their original values",
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
