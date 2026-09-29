"""A4: can NXstress's string-valued per-scan-point fields be made growable?

To implement 04c's tail-append for the scan-point family, the writer has to stop
emitting those fields as contiguous datasets and start passing
``maxshape=(None,)`` + ``chunks`` (see ``a5_scan_point_family_resizable.py`` for
why). For the numeric fields that is uncontroversial. For the **string-valued**
ones it is not, and this repository has already been bitten twice in that exact
area:

* ``a4_string_log_dtypes.py`` found that a NumPy fixed-width ``<U`` array has no
  h5py conversion path at all and crashes the save outright, which is why
  ``_Sample._writable`` exists; and that the writable variable-length UTF-8
  dtype still reads back as ``bytes``.
* ``a4_h5py_nexusformat_append.py`` confirmed vlen resize works -- but only for
  arrays it created itself as ``np.array([...], dtype=STRING_DTYPE)``.

Neither covers the forms the writer actually uses. ``entry/start_time`` and
``entry/end_time`` are built from a plain Python ``list[str]``
(``NXstress.py:1093-1094``) and handed to ``NXfield`` with no dtype at all, letting
``nexusformat`` choose. ``SAMPLE_DESCRIPTION/logs/*`` are whatever
``_Sample._writable`` returns, which for a bytes log is an ``|S`` array that
passes through untouched. Adding ``chunks`` moves every one of these off h5py's
contiguous path onto its chunked one.

Claims under test
-----------------
1. New, asserted by no document: a ``list[str]`` handed to ``NXfield`` with
   ``maxshape``/``chunks`` can be written at all -- i.e. the dtype
   ``nexusformat`` infers has a chunked h5py path.
2. New: such a field can be reopened and tail-appended, and the **values**
   round-trip (not merely the shape) -- the ``bytes``-vs-``str`` trap from
   ``a4_string_log_dtypes.py``.
3. New: the same, for the ``|S`` bytes arrays ``_Sample._writable`` passes
   through, and for the explicit vlen UTF-8 dtype it converts ``<U`` into.
4. ``04c-nxstress-append.md:355-359`` -- "appending to an existing (non-empty)
   group grows each dataset by exactly the new row count, with existing rows
   byte-for-byte unchanged and new rows correctly appended after them" --
   measured here for the string cases specifically.
5. New: a **fixed-width** ``|S`` column -- which ``_writable`` passes through
   untouched -- is sized by the longest value present *at write time*. Appending
   a longer one afterwards is the case that cannot be caught by "did it raise",
   so it is measured separately and by value.

Run: ``pixi run python plans/NXstress-prod/probes/a4_growable_string_fields.py``
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
from nexusformat.nexus import NXdata, NXentry, NXfield, nxopen

# Mirrors pyrs/utilities/NXstress/_definitions.py exactly.
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def CHUNK_SHAPE(rank: int) -> tuple[int, ...]:
    return (1,) * (rank - 1) + (100,)


def growable(rank: int) -> dict:
    return {"maxshape": (None,) * rank, "chunks": CHUNK_SHAPE(rank)}


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


# The four value forms the NXstress writer actually produces for a string column.
CASES = {
    "list[str] (entry/start_time, NXstress.py:1093)": ["2024-01-15T10:00:00", "2024-01-15T10:01:00"],
    "np.array of |S bytes (_Sample._writable passthrough)": np.array([b"run_a.h5", b"run_b.h5"]),
    "vlen UTF-8 (_Sample._writable converts <U to this)": np.array(["11", "11"], dtype=STRING_DTYPE),
    "bare <U (what _writable exists to prevent)": np.array(["11", "11"]),
}

EXTRA = {
    "list[str] (entry/start_time, NXstress.py:1093)": ["2024-01-15T10:02:00"],
    "np.array of |S bytes (_Sample._writable passthrough)": np.array([b"run_c.h5"]),
    "vlen UTF-8 (_Sample._writable converts <U to this)": np.array(["22"], dtype=STRING_DTYPE),
    "bare <U (what _writable exists to prevent)": np.array(["22"]),
}


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: growable string fields -- write, reopen, tail-append, read back")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="a4_growstr_"))

    for n, (label, values) in enumerate(CASES.items()):
        path = tmp / f"case_{n}.nxs"

        # --- Claim 1: can it be WRITTEN with maxshape + chunks? ---
        try:
            with nxopen(path, "w") as root:
                root["entry"] = NXentry()
                root["entry"]["data"] = NXdata()
                root["entry"]["data"]["field"] = NXfield(values, **growable(1))
            written = "written"
        except Exception as error:  # noqa: BLE001 -- the failure IS the finding
            report(f"claim 1 -- {label}", f"WRITE FAILED: {type(error).__name__}: {error}")
            continue

        with h5py.File(path, "r") as f:
            dset = f["entry/data/field"]
            on_disk = f"dtype={dset.dtype} shape={dset.shape} maxshape={dset.maxshape} chunks={dset.chunks}"
            original = dset[()].tolist()

        # --- Claims 2-4: reopen, tail-append, and check VALUES, not just shape ---
        try:
            with nxopen(path, "rw") as root:
                field = root["entry"]["data"]["field"]
                cur = field.shape[0]
                field.resize((cur + len(EXTRA[label]),))
                field[cur:] = EXTRA[label]
            appended = "appended"
        except Exception as error:  # noqa: BLE001
            report(
                f"claims 1-4 -- {label}",
                f"{written} ({on_disk})\n          APPEND FAILED: {type(error).__name__}: {error}",
            )
            continue

        with h5py.File(path, "r") as f:
            final = f["entry/data/field"][()].tolist()

        preserved = final[: len(original)] == original
        report(
            f"claims 1-4 -- {label}",
            f"{written}, {appended}; {on_disk}\n          "
            f"grew {len(original)} -> {len(final)} row(s); existing rows unchanged: {preserved}\n          "
            f"read back: {final!r}",
        )

    # --- Claim 5: the fixed-width |S trap, which no shape check can see ---
    path = tmp / "case_widening.nxs"
    with nxopen(path, "w") as root:
        root["entry"] = NXentry()
        root["entry"]["data"] = NXdata()
        root["entry"]["data"]["field"] = NXfield(np.array([b"short.h5"]), **growable(1))
    longer = b"a_considerably_longer_filename.h5"
    with nxopen(path, "rw") as root:
        field = root["entry"]["data"]["field"]
        cur = field.shape[0]
        field.resize((cur + 1,))
        field[cur:] = np.array([longer])
    with h5py.File(path, "r") as f:
        dtype, final = f["entry/data/field"].dtype, f["entry/data/field"][()].tolist()
    report(
        "claim 5 -- appending a LONGER value to a fixed-width |S column",
        f"column dtype is {dtype} (sized by the longest value at WRITE time)\n          "
        f"appended {longer!r} ({len(longer)} bytes)\n          "
        f"read back: {final!r}\n          "
        f"round-tripped intact: {final[-1] == longer}"
        + ("" if final[-1] == longer else "  <-- SILENTLY TRUNCATED, no exception raised"),
    )

    print("\n" + "-" * 78)
    print("  Note what 'read back' shows: a value written as `str` comes back as `bytes`.")
    print("  That is the pre-existing round-trip property pinned by a4_string_log_dtypes.py,")
    print("  not something `maxshape`/`chunks` introduces -- but any append that COMPARES")
    print("  an incoming value against one already on disk must decode first.")
    print("-" * 78)
    print(f"\n  (artifacts left under {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
