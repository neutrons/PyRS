"""A5 [RETIRED -- DOES NOT RUN]: what the landed NXstress library did with our data.

RETIRED
-------
**This probe fails by design and is not evidence of a defect.** Retired to
record by the 04b PR (`7799a673`): claim 5 pinned the *pre-04b* ``write``
signature, and 04b changed exactly that, so re-pointing the probe at
``write([ws], [peaks])`` would erase the one thing it is kept for. It now stops
at ``nxs.write(ws, peaks)`` with a pydantic ``ValidationError`` -- the bare
``HidraWorkspace`` against 04b's list signature. **Any other failure is a real
finding.**

Where each claim went (`process.md` 7.5 item 6 -- supersede, do not delete):

==========  ==========================================================
Claim 1     ``a5_discriminator_resolution.py``, ``a5_subruns_nonmonotonic.py``
Claim 2     **no successor probe** -- read-back completeness is now test
            coverage only (``test_workspace_read.py``,
            ``test_multi_workspace.py``)
Claim 3     ``a5_append_preconditions.py`` (``entry_number`` targeting)
Claim 4     ``a5_append_preconditions.py`` -- **and the claim is now FALSE**
Claim 5     nothing; it is history, which is why this file is kept
==========  ==========================================================

Read the claims below as *what was believed when they were written*, not as
current behaviour. Two are now wrong, and are left unedited for the same reason
the probe is: a log preserves what was believed.

The A4/A5 boundary is **which counterparty, not which interpreter**: a claim
about ``h5py`` is A4, a claim about ``pyrs/utilities/NXstress/`` is A5, because
that library is in this repo and has already landed. These are the claims the
GUI hookup will be built on.

Claims under test
-----------------
1. ``README.md:94-97`` -- "Its input/output types are **exactly** the pair
   ``(HidraWorkspace, list[PeakCollection])`` that PeakFittingViewer and
   TextureFittingViewer already produce and consume."
2. ``README.md:363-367`` -- "``NXstress.read`` reconstructs wavelengths, sample
   logs, masks, reduced diffraction data, and peak collections. It does NOT
   reconstruct raw counts unless the optional ``input_data`` group was written."
3. ``README.md:99-102`` -- "Multiple ``NXentry`` groups per file are supported
   (``NXstress.py:156-164``)."
4. ``README.md:114-123`` -- "``_input_data.py:44,63`` and ``NXstress.py:151-152``
   raise on any operation that would extend an existing ``NXentry``. Each write
   must currently be a fresh entry (save-as)."

   **NO LONGER TRUE.** 04c implemented the append path and Decisions row 33
   made ``write()`` dispatch on whether the resolved entry exists, so extending
   an existing ``NXentry`` is now the supported case. See
   ``a5_append_preconditions.py``.
5. ``04b:110-115`` -- the **pre-04b** baseline: "Change ``NXstress.write`` to
   accept ``list[HidraWorkspace]`` in place of a single ``HidraWorkspace``".
   Probed here only to pin what the signature was *at the time*.

   **Superseded.** This is the claim that retired the probe. Note that the one
   statement here that still executes -- ``inspect.signature(NXstress.write)``
   -- reports the *post*-04b signature, so its output now contradicts the claim
   heading it prints under. The record of the old signature is this prose, not
   that measurement.

Deliberately NOT probed, and why
--------------------------------
04b's N-workspace round trip, its discriminator resolution, and 04c's append
path **cannot be probed**: none of that code exists yet. Recording the gap is
the point -- afterwards, "no probe was needed" and "no probe was written" look
identical. Those claims stay A5-**uncovered** until 04b and 04c land, and the
coverage matrix says so rather than implying they were checked.

**All three have since landed and all three are now probed** -- by
``a5_subruns_nonmonotonic.py``, ``a5_discriminator_resolution.py`` and
``a5_append_preconditions.py`` respectively. The paragraph above is kept as the
record of a gap that was stated rather than left implicit; it no longer
describes the present.

Do not run: this probe is retired (see RETIRED above).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
from nexusformat.nexus import nxopen

# See `plans/audit-process/process.md` 7.5 item 9. Absent means "live"; a sweep over
# the directory must skip, or expect the failure of, any probe that sets this.
AUDIT_STATUS = "retired"

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.utilities.NXstress.NXstress import NXstress  # noqa: E402
from tests.util.peak_collection_helpers import createPeakCollection as _peak_fixture  # noqa: E402


def _unwrap_fixture(fixture, /, *args, **kwargs):
    """Call a pytest generator-fixture's underlying factory directly."""
    return next(fixture.__wrapped__())(*args, **kwargs)


def make_workspace(name: str, n_subruns: int = 3, with_raw_counts: bool = False):
    """Build a workspace with the repo's OWN fixture, not a local copy.

    Reusing `tests/unit/pyrs/utilities/NXstress/conftest.py` matters: a probe
    that reimplements the counterparty's setup is testing its own
    reimplementation, and would keep passing after the real fixture changed.
    """
    spec = importlib.util.spec_from_file_location(
        "_nxstress_conftest", REPO / "tests" / "unit" / "pyrs" / "utilities" / "NXstress" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _unwrap_fixture(
        module.minimal_HidraWorkspace,
        name=name,
        n_subruns=n_subruns,
        with_raw_counts=with_raw_counts,
        with_masks=True,
        mask_names=("mask_a",),
    )


def make_peak_collection(peak_tag: str, n_subruns: int = 3):
    """Build a PeakCollection using the repo's own test helper.

    ``createPeakCollection`` is a pytest *fixture*, so calling it directly
    fails. Unwrapping it reuses the real helper rather than duplicating its
    construction logic here -- a probe that reimplements the counterparty is
    testing itself.
    """
    return _unwrap_fixture(
        _peak_fixture,
        peak_tag=peak_tag,
        peak_profile="PseudoVoigt",
        background_type="Linear",
        wavelength=1.452,
        projectfilename="probe.h5",
        runnumber=1234,
        N_subrun=n_subruns,
    )


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: the landed NXstress library's real contract")
    print("=" * 78)

    # --- Claims 1 and 5: what are the signatures TODAY? ---
    write_sig = inspect.signature(NXstress.write)
    read_sig = inspect.signature(NXstress.read)
    report(
        "write takes a single HidraWorkspace today; 04b changes it to a list (claims 1, 5)",
        f"NXstress.write{write_sig}  |  NXstress.read{read_sig}",
    )

    tmp = Path(tempfile.mkdtemp(prefix="a5_nxs_"))

    # --- Claim 2: read-back completeness, WITHOUT raw counts ---
    ws = make_workspace("probe_ws", with_raw_counts=False)
    peaks = [make_peak_collection("Fe 110")]
    path = tmp / "no_raw.nxs"
    with NXstress(path, "w") as nxs:
        nxs.write(ws, peaks)
    with NXstress(path, "r") as nxs:
        back, back_peaks = nxs.read()

    raw_before = dict(getattr(ws, "_raw_counts", {}) or {})
    raw_after = dict(getattr(back, "_raw_counts", {}) or {})
    report(
        "read() reconstructs logs/wavelength/diffraction/peaks, but NOT raw "
        "counts when input_data was never written (claim 2)",
        f"sub_runs={np.asarray(back.get_sub_runs()).tolist()}; "
        f"wavelength={back.get_wavelength(False, False)}; "
        f"sample logs recovered={sorted(back.get_sample_log_names())[:6]}...; "
        f"peak collections={[p.peak_tag for p in back_peaks]}; "
        f"raw counts written={len(raw_before)} recovered={len(raw_after)}",
    )

    # --- Claim 2 again, WITH raw counts ---
    ws2 = make_workspace("probe_ws_raw", with_raw_counts=True)
    path2 = tmp / "with_raw.nxs"
    with NXstress(path2, "w") as nxs:
        nxs.write(ws2, peaks)
    with NXstress(path2, "r") as nxs:
        back2, _ = nxs.read()
    report(
        "...and DOES reconstruct them when the optional input_data group was written (claim 2)",
        f"raw counts written={len(getattr(ws2, '_raw_counts', {}) or {})} "
        f"recovered={len(getattr(back2, '_raw_counts', {}) or {})}",
    )

    # --- Claim 3: multiple NXentry in one file ---
    with NXstress(path, "w") as nxs:
        nxs.write(ws, peaks)
        nxs.write(ws, peaks)
    with nxopen(str(path), "r") as root:
        entries = sorted(root.NXentry and [e.nxname for e in root.NXentry])
    report("multiple NXentry per file are supported (claim 3)", f"entries in one file: {entries}")

    # --- Claim 4: extending an existing NXentry raises ---
    from pyrs.utilities.NXstress._input_data import _InputData

    src = inspect.getsource(_InputData.init_group)
    guard = [ln.strip() for ln in src.splitlines() if "not implemented" in ln.lower()]
    report(
        "any operation extending an existing NXentry raises today (claim 4)",
        f"_input_data.init_group guard: {guard or 'NONE FOUND'}",
    )

    # An earlier draft of this probe asked "does opening in 'a' mode and writing
    # raise?", got "no exception", and was about to report that as a finding
    # against README:114-123. It is not one: mode "a" adds a NEW auto-numbered
    # NXentry rather than extending the existing one, which is exactly what
    # 04c:14-21 says. Count the entries, or the observation misleads.
    with nxopen(str(path), "r") as root:
        before_entries = len([e.nxname for e in root.NXentry])
    try:
        with NXstress(path, "a") as nxs:
            nxs.write(ws, peaks)
        outcome = "no exception raised"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {exc}"
    with nxopen(str(path), "r") as root:
        after_entries = [e.nxname for e in root.NXentry]
    report(
        "what mode 'a' actually does today: does it EXTEND the existing entry, "
        "or add a new one? (claim 4, and 04c:14-21's premise)",
        f"{outcome}; NXentry count {before_entries} -> {len(after_entries)} {sorted(after_entries)}. "
        f"So a write never extends an existing entry -- it appends a new one.",
    )

    print(f"\n(probe files kept under {tmp})")
    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
