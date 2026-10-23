"""A4: does the tail-append mechanism 04c is built on actually work on disk?

Spec 04c's entire design reduces the common append case to "code that already
exists" -- the ``resize(cur+N); arr[cur:] = ...`` shape in ``_Peaks._append_peak``
and ``_fit.py``'s ``_PeakParameters._append_peak``. But every landed use of that
shape runs against **in-memory** ``NXfield`` objects during ``init_group``, before
the file is written. Append runs it against a **reopened, file-backed** group.
Those are different objects with different backing stores, and reading cannot
tell you whether the second works.

Claims under test
-----------------
1. ``04c-nxstress-append.md:92-99`` --
   "the common case that actually needs to work -- 'append a new workspace's
   worth of data' -- reduces to code that already exists (see
   ``_Peaks._append_peak``, ``_fit.py``'s ``_PeakParameters._append_peak``, both
   already written as ``cur = shape[0]; resize(cur+N); arr[cur:] = ...``)".
2. ``04c-nxstress-append.md:251-260`` --
   "reusing the existing ``_append_peak`` ... against an existing on-disk
   ``PeakIndex`` group instead of a freshly ``_init``-ed one ... no change to its
   resize/assign logic, only to what group it's called against."
3. ``09-fit-spectrum-nxstress.md:31-36`` --
   "These fields are currently **zero-sized resizable datasets**
   (``np.empty((0,0))``, ``fillvalue=np.nan``), not pre-shaped NaN arrays -- the
   writer must resize them to the real shape before populating".
4. ``04c-nxstress-append.md:355-359`` --
   "appending to an existing (non-empty) group grows each dataset by exactly the
   new row count, with existing rows byte-for-byte unchanged and new rows
   correctly appended after them."
5. ``04c-nxstress-append.md:175-181`` and ``:330-333`` --
   "The classification/conflict check runs against **all** affected groups before
   any resize/append call is made ... so a rejected (``RuntimeError``) and an
   unsupported (``NotImplementedError``) append are both true no-ops -- the
   on-disk entry is left byte-for-byte unchanged either way."
6. ``04c-nxstress-append.md:151-158`` --
   "Appending a genuinely new, distinguishable workspace to an entry that was
   originally written as a bare ``N == 1``/no-discriminator write isn't possible
   without adding a new on-disk column, which this spec's tail-append design does
   not do (no schema restructuring, only resize-and-append into existing
   datasets)."

The string fields matter independently: ``phase_name`` and ``mask`` are
``h5py.string_dtype(encoding="utf-8")`` variable-length strings, and vlen resize
behaviour is not obviously the same as a fixed-width numeric resize.

Run: ``pixi run python plans/NXstress-prod/probes/a4_h5py_nexusformat_append.py``
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from nexusformat.nexus import NXentry, NXfield, NXreflections, nxopen

# Mirrors pyrs/utilities/NXstress/_definitions.py exactly.
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def CHUNK_SHAPE(rank: int) -> tuple[int, ...]:
    return (1,) * (rank - 1) + (100,)


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def make_file(path: Path) -> None:
    """Build a file the way `_Peaks._init` does: zero-sized, resizable, chunked."""
    with nxopen(str(path), "w") as root:
        root["entry"] = NXentry()
        peaks = NXreflections()
        peaks["scan_point"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1))
        peaks["h"] = NXfield(np.empty((0,), dtype=np.int32), maxshape=(None,), chunks=CHUNK_SHAPE(1))
        peaks["phase_name"] = NXfield(np.empty((0,), dtype=STRING_DTYPE), maxshape=(None,), chunks=CHUNK_SHAPE(1))
        peaks["center"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), fillvalue=np.nan
        )
        # The 2-D zero-sized case spec 09 describes for diffractogram fit fields.
        peaks["fit"] = NXfield(
            np.empty((0, 0), dtype=np.float64), maxshape=(None, None), chunks=CHUNK_SHAPE(2), fillvalue=np.nan
        )
        root["entry"]["PEAKS"] = peaks


def append_rows(path: Path, scan: list[int], hh: list[int], phase: list[str], center: list[float]) -> str:
    """The exact `_append_peak` shape: cur = shape[0]; resize(cur+N); arr[cur:] = ..."""
    n = len(scan)
    with nxopen(str(path), "rw") as root:
        peaks = root["entry"]["PEAKS"]
        cur = peaks["h"].shape[0]
        new_len = cur + n
        peaks["scan_point"].resize((new_len,))
        peaks["h"].resize((new_len,))
        peaks["phase_name"].resize((new_len,))
        peaks["center"].resize((new_len,))
        peaks["scan_point"][cur:] = np.array(scan, dtype=np.int32)
        peaks["h"][cur:] = np.array(hh, dtype=np.int32)
        peaks["phase_name"][cur:] = np.array(phase, dtype=STRING_DTYPE)
        peaks["center"][cur:] = np.array(center, dtype=np.float64)
    return f"{n} row(s) appended at offset {cur}"


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: resize-then-assign on reopened, file-backed NXfields")
    print("=" * 78)
    print(f"\nh5py {h5py.__version__}, numpy {np.__version__}")
    import nexusformat

    print(f"nexusformat {nexusformat.__version__}")

    tmp = Path(tempfile.mkdtemp(prefix="a4_append_"))
    path = tmp / "probe.nxs"

    make_file(path)
    with nxopen(str(path), "r") as root:
        peaks = root["entry"]["PEAKS"]
        shapes = {k: peaks[k].shape for k in ("scan_point", "h", "phase_name", "center", "fit")}
    report("zero-sized resizable datasets are created as such (claim 3)", f"initial shapes: {shapes}")

    # --- Claim 1/2/3: first append, into a genuinely EMPTY resizable dataset ---
    detail = append_rows(path, [1, 2, 3], [1, 1, 1], ["Fe", "Fe", "Fe"], [1.1, 1.2, 1.3])
    with nxopen(str(path), "r") as root:
        peaks = root["entry"]["PEAKS"]
        first = {
            "scan_point": peaks["scan_point"].nxdata.tolist(),
            "phase_name": [s.decode() if isinstance(s, bytes) else s for s in peaks["phase_name"].nxdata],
            "center": peaks["center"].nxdata.tolist(),
        }
    report(
        "resize(cur+N); arr[cur:] = ... works on a ZERO-SIZED reopened dataset (claims 1-3)",
        f"{detail}; read back {first}",
    )

    before_digest = digest(path)
    before_rows = first["scan_point"]

    # --- Claim 4: second append, into a NON-EMPTY on-disk dataset ---
    detail = append_rows(path, [4, 5], [2, 2], ["Ni", "Ni"], [2.1, 2.2])
    with nxopen(str(path), "r") as root:
        peaks = root["entry"]["PEAKS"]
        after = {
            "scan_point": peaks["scan_point"].nxdata.tolist(),
            "phase_name": [s.decode() if isinstance(s, bytes) else s for s in peaks["phase_name"].nxdata],
            "center": peaks["center"].nxdata.tolist(),
        }
    preserved = after["scan_point"][: len(before_rows)] == before_rows
    report(
        "grows by exactly N, existing rows unchanged, new rows after them (claim 4)",
        f"{detail}; scan_point now {after['scan_point']}; "
        f"existing rows preserved: {preserved}; vlen strings: {after['phase_name']}",
    )

    # --- Claim 5: is an aborted append a true no-op on disk? ---
    digest_before_abort = digest(path)
    aborted_with = ""
    try:
        with nxopen(str(path), "rw") as root:
            _ = root["entry"]["PEAKS"]["h"].shape[0]
            raise RuntimeError("simulated conflict detected before any resize")
    except RuntimeError as exc:
        # `except ... as exc` unbinds `exc` at block exit; capture it here.
        aborted_with = str(exc)
    digest_after_abort = digest(path)
    report(
        "opening 'rw' and raising BEFORE any resize leaves the file unchanged (claim 5)",
        f"sha256[:16] before={digest_before_abort} after={digest_after_abort} "
        f"identical={digest_before_abort == digest_after_abort} ({aborted_with})",
    )

    # --- Claim 6: can a NEW column be added to an existing group at all? ---
    try:
        with nxopen(str(path), "rw") as root:
            peaks = root["entry"]["PEAKS"]
            n = peaks["h"].shape[0]
            peaks["direction"] = NXfield(
                np.array(["11"] * n, dtype=STRING_DTYPE), maxshape=(None,), chunks=CHUNK_SHAPE(1)
            )
        with nxopen(str(path), "r") as root:
            added = sorted(root["entry"]["PEAKS"].keys())
        outcome = f"SUCCEEDED -- group now has {added}"
    except Exception as exc:  # noqa: BLE001 - the point is what it raises, if anything
        outcome = f"RAISED {type(exc).__name__}: {exc}"
    report(
        "adding a NEW on-disk column to an existing group -- 04c says its design "
        "does not do this; is it even possible? (claim 6)",
        outcome,
    )

    print(f"\n(probe file kept at {path} for inspection; safe to delete)")
    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    print(f"digest before first append: {before_digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
