"""A5: is a rejected append really a no-op, and does each precondition really fire?

04c states that "the classification/conflict check runs against **all** affected
groups before any resize/append call is made ... so a rejected (``RuntimeError``)
and an unsupported (``NotImplementedError``) append are both true no-ops -- the
on-disk entry is left byte-for-byte unchanged either way" (``:175-181``).

``a4_h5py_nexusformat_append.py`` confirmed the *mechanism* this rests on -- that
merely opening ``'rw'`` and raising is byte-neutral -- and said so explicitly:
"It does **not** establish atomicity across a *partial* append, which is why
this spec's ... ordering requirement is load-bearing and must survive
implementation." The coverage matrix recorded 04c's dispatch as A5-**uncovered**
for the stated reason that none of it existed.

It exists now, so this probe measures the thing the earlier one could not: it
builds a real entry through ``NXstress.write``, drives every rejection path, and
compares the file's sha256 before and after each. A check that runs too late
shows up here as a changed digest, not as a missing exception -- which is
exactly how the ordering requirement fails in practice, and how it did fail
twice during implementation.

Claims under test
-----------------
1. ``04c-nxstress-append.md:175-181`` -- every rejection leaves the entry
   byte-for-byte unchanged.
2. ``04c-nxstress-append.md:159-166`` -- Case B raises ``NotImplementedError``
   and does **not** invalidate the instance.
3. ``04c-nxstress-append.md:167-173`` -- an exact duplicate raises
   ``RuntimeError``, and ``:183-189`` -- it *does* invalidate the instance.
4. ``04c-nxstress-append.md:140-158`` -- Case A's two preconditions (at least
   one ``PeakCollection``; a discriminator scheme already established).
5. ``04c-nxstress-append.md:44-53`` -- the appended file is "locally sorted,
   globally segmented", and the reader accepts it.
6. Follow-up 3 F3.1 -- the refusals that ``tail_append`` would otherwise make
   during mutation (a two-theta or pixel-count disagreement, a mask-set
   disagreement, a non-resizable dataset) are hoisted into the pre-flight, so
   they too are byte-neutral. **A regression here does not raise differently --
   it raises identically and changes the file**, which is why every case is
   measured by digest rather than by exception type.

Run: ``pixi run python plans/NXstress-prod/probes/a5_append_preconditions.py``
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.utilities import config as _config  # noqa: E402
from pyrs.utilities.NXstress.NXstress import NXstress  # noqa: E402
from tests.util.peak_collection_helpers import createPeakCollection as _peak_fixture  # noqa: E402


def _unwrap_fixture(fixture, /, *args, **kwargs):
    """Call a pytest generator-fixture's underlying factory directly."""
    return next(fixture.__wrapped__())(*args, **kwargs)


def _conftest():
    """The repo's own NXstress fixtures. A probe that reimplements them tests itself."""
    spec = importlib.util.spec_from_file_location(
        "_nxstress_conftest", REPO / "tests" / "unit" / "pyrs" / "utilities" / "NXstress" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFTEST = _conftest()


def make(direction: str, points: tuple, peak_tag: str = "Fe110", **ws_kwargs):
    sub_runs = np.array(points)
    ws = _unwrap_fixture(
        CONFTEST.minimal_HidraWorkspace, with_instrument=True, with_masks=True, sub_runs=sub_runs, **ws_kwargs
    )
    ws.set_sample_log("direction", sub_runs, np.array([direction] * len(sub_runs)))
    collection = _unwrap_fixture(
        _peak_fixture,
        peak_tag=peak_tag,
        peak_profile="Gaussian",
        background_type="Linear",
        wavelength=1.486,
        projectfilename="probe.h5",
        runnumber=1,
        N_subrun=len(sub_runs),
        sub_runs=sub_runs,
    )
    return ws, [collection]


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def fresh(tmp: Path, name: str) -> Path:
    """A one-workspace entry, written the ordinary way."""
    path = tmp / name
    ws, peaks = make("11", (1, 2, 3))
    with NXstress(path, "w") as nxs:
        nxs.write([ws], [peaks])
    return path


def attempt(path: Path, ws, peaks, **kwargs) -> tuple[str, bool]:
    """Try an append; return (outcome, file-unchanged)."""
    before = digest(path)
    outcome = "NO EXCEPTION -- the append was accepted"
    try:
        with NXstress(path, "a", **kwargs) as nxs:
            nxs.write([ws], [peaks])
    except Exception as error:  # noqa: BLE001 -- the type and message ARE the finding
        outcome = f"{type(error).__name__}: {str(error).splitlines()[0]}"
    return outcome, digest(path) == before


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: 04c's conflict classification, measured against a real entry")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="a5_precond_"))
    override = tmp / "override.yml"
    override.write_text("nxstress:\n  discriminator_fields: ['direction']\n")
    _config.Config.loadEnv(str(override))

    # --- Claim 5: the supported case, first, so the rejections below mean something ---
    path = fresh(tmp, "roundtrip.nxs")
    ws, peaks = make("22", (4, 5, 6))
    with NXstress(path, "a") as nxs:
        nxs.write([ws], [peaks])
    with NXstress(path, "r") as nxs:
        workspaces, peakss = nxs.read()
    import h5py

    with h5py.File(path, "r") as f:
        order = [v.decode() for v in f["entry/peaks/direction"][()]]
    report(
        "claim 5: a Case-A append round-trips, and the file is locally sorted / globally segmented",
        f"{len(workspaces)} workspace(s) recovered, scan points "
        f"{[w.get_sub_runs().raw_copy().tolist() for w in workspaces]}, directions "
        f"{[w.get_sample_log_value('direction') for w in workspaces]}; "
        f"peak-index discriminator column reads {order} "
        f"(second block FOLLOWS the first -- not re-sorted into it)",
    )

    # --- Claims 1-4: every rejection path, measured by digest ---
    cases = [
        (
            "claim 2: Case B -- more scan points under a key the entry already holds",
            lambda p: attempt(p, *make("11", (4, 5, 6))),
        ),
        (
            "claim 3: exact duplicate -- same key, overlapping scan point",
            lambda p: attempt(p, *make("11", (2, 3, 4))),
        ),
        (
            "claim 4a: Case A precondition 1 -- the new workspace contributes no PeakCollection",
            lambda p: attempt(p, make("22", (4, 5, 6))[0], []),
        ),
        (
            "claim 1: a new key reusing a scan point already in the entry",
            lambda p: attempt(p, *make("22", (3, 4, 5), peak_tag="Si111")),
        ),
        (
            "claim 1: entry_number skipping past the next free number",
            lambda p: attempt(p, *make("22", (4, 5, 6)), entry_number=12),
        ),
        (
            "claim 6: a DIFFERENT two-theta binning -- the case that corrupted a file",
            lambda p: attempt(p, *make("22", (4, 5, 6), n_two_theta=25)),
        ),
        (
            "claim 6: a different detector mask set",
            lambda p: attempt(p, *make("22", (4, 5, 6), mask_names=("mask_a",))),
        ),
    ]
    for label, run in cases:
        case_path = fresh(tmp, f"{abs(hash(label))}.nxs")
        outcome, unchanged = run(case_path)
        report(label, f"{outcome}\n          file byte-for-byte unchanged: {unchanged}")

    # --- Claim 4b: an entry with no discriminator scheme at all ---
    bare = tmp / "bare.nxs"
    plain = _unwrap_fixture(
        CONFTEST.minimal_HidraWorkspace, with_instrument=True, with_masks=True, sub_runs=np.array([1, 2, 3])
    )
    plain_peaks = [
        _unwrap_fixture(
            _peak_fixture,
            peak_tag="Fe110",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.486,
            projectfilename="probe.h5",
            runnumber=1,
            N_subrun=3,
            sub_runs=np.array([1, 2, 3]),
        )
    ]
    empty_override = tmp / "empty.yml"
    empty_override.write_text("nxstress:\n  discriminator_fields: []\n")
    _config.Config.loadEnv(str(empty_override))
    with NXstress(bare, "w") as nxs:
        nxs.write([plain], [plain_peaks])
    _config.Config.loadEnv(str(override))
    outcome, unchanged = attempt(bare, *make("22", (4, 5, 6)))
    report(
        "claim 4b: Case A precondition 2 -- the target entry has no discriminator column",
        f"{outcome}\n          file byte-for-byte unchanged: {unchanged}",
    )

    # --- Claims 2 vs 3: which outcome invalidates the instance? ---
    path = fresh(tmp, "invalidation.nxs")
    case_b_then_a = "?"
    with NXstress(path, "a") as nxs:
        try:
            nxs.write(*[[x] for x in make("11", (4, 5, 6))])
        except NotImplementedError:
            pass
        try:
            ws, peaks = make("22", (7, 8, 9))
            nxs.write([ws], [peaks])
            case_b_then_a = "the later Case-A append SUCCEEDED -- instance stayed usable"
        except Exception as error:  # noqa: BLE001
            case_b_then_a = f"the later Case-A append raised {type(error).__name__}: {error}"

    path = fresh(tmp, "invalidation2.nxs")
    duplicate_then_a = "?"
    with NXstress(path, "a") as nxs:
        try:
            ws, peaks = make("11", (2, 3, 4))
            nxs.write([ws], [peaks])
        except RuntimeError:
            pass
        try:
            ws, peaks = make("22", (7, 8, 9))
            nxs.write([ws], [peaks])
            duplicate_then_a = "the later append SUCCEEDED -- instance was NOT invalidated"
        except RuntimeError as error:
            duplicate_then_a = f"the later append raised RuntimeError: {str(error).splitlines()[0]}"
    report(
        "claims 2-3: Case B leaves the instance usable; a duplicate does not",
        f"after NotImplementedError: {case_b_then_a}\n          after RuntimeError:       {duplicate_then_a}",
    )

    print("\n" + "-" * 78)
    print("  Note what the digests measure. A precondition checked in the wrong place")
    print("  still raises the right exception -- and leaves a changed file behind,")
    print("  because an earlier group has already grown. Two checks failed exactly that")
    print("  way during implementation and were moved into the pre-flight pass; the")
    print("  exception alone would not have shown it.")
    print("-" * 78)
    print(f"\n  (artifacts left under {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
