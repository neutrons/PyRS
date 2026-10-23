"""A4: which string array dtypes can a NeXus field actually hold, and what comes back?

**This is about log VALUES, not log names.** The two are entirely separate
mechanisms and only one of them was ever in question. A log *name* is converted
by `allowed_identifier` (Decisions rows 23 and 25), which is total, injective
and reversible over arbitrary text -- there is no PV-log key it cannot encode,
and claim 5 below re-confirms that rather than taking it on trust. What had
never been checked is the dtype of the *array of values* stored under that
name.

04b's discriminator values are resolved from `HidraWorkspace` sample logs, and
for the case that motivates the whole mechanism -- spec 05's `direction`, with
values `"11"`/`"22"`/`"33"` -- those values are an array of Python strings.
Such an array is NumPy dtype `<U2`, and nothing in the series had checked
whether a `<U` array can be written at all.

It cannot, and the failure is not a NeXus rule: it comes from h5py, which has
no conversion path for NumPy's fixed-width unicode dtype. The existing test
fixtures never hit it because every string log they build (`start_time`,
`end_time`, `Filename`) is deliberately constructed as *bytes*, matching what a
real h5py-backed dataset returns.

Claims under test
-----------------
1. A `<U` string array cannot be written to a NeXus field.
2. A bytes (`|S`) array can -- which is why the existing fixtures work.
3. The h5py variable-length UTF-8 dtype (`FIELD_DTYPE.STRING`, already used for
   `phase_name` and `mask` in `_peaks.py::_init`) can.
4. What each writable form reads back *as*, which decides whether a value
   resolved before a write compares equal to the same value resolved after one.
   A round trip that returns `b"11"` where `"11"` went in is a silent
   discriminator mismatch, not an error.
5. That none of this is a *name* problem: an awkward PV-log key encodes and
   writes fine whatever its values are, so the two mechanisms are independent.
   Claimed explicitly because "a string log cannot be written" invites exactly
   the wrong reading, and because a reader who concluded `allowed_identifier`
   was incomplete would go looking for a defect that is not there.

Reported as resulting state rather than as "accepted"/"rejected" alone, per
`probes/README.md`'s failure-modes table.

Run: ``pixi run python plans/NXstress-prod/probes/a4_string_log_dtypes.py``
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from nexusformat.nexus import NXentry, NXfield, nxopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.utilities.NXstress._definitions import (  # noqa: E402
    FIELD_DTYPE,
    allowed_identifier,
    decode_identifier,
)


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def round_trip(label: str, values: np.ndarray) -> str:
    """Write one field, read it back, and describe both ends."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.nxs"
        try:
            with nxopen(str(path), "w") as root:
                root["entry"] = NXentry()
                root["entry"]["field"] = NXfield(values)
        except Exception as exc:  # noqa: BLE001 - what it raises is the result
            return f"{label}: WRITE RAISED {type(exc).__name__}: {exc}"

        with nxopen(str(path), "r") as root:
            back = root["entry"]["field"].nxdata

    element = back[0] if len(back) else None
    return (
        f"{label}: wrote dtype {values.dtype!s} -> read dtype {np.asarray(back).dtype!s}, "
        f"element {element!r} (type {type(element).__name__}), "
        f"equal to input: {list(back) == list(values)}"
    )


def named_field_written(name: str) -> str:
    """Write one numeric field under `name`, to isolate the name path from the value path."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.nxs"
        try:
            with nxopen(str(path), "w") as root:
                root["entry"] = NXentry()
                root["entry"][name] = NXfield(np.array([1.0, 2.0, 3.0]))
        except Exception as exc:  # noqa: BLE001 - what it raises is the result
            return f"{name!r}: RAISED {type(exc).__name__}"
    return f"{name!r}: OK"


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: string dtypes in a NeXus field, and what a round trip returns")
    print("=" * 78)

    unicode_values = np.array(["11", "22", "33"])
    bytes_values = np.array([b"11", b"22", b"33"])
    vlen_values = np.array(["11", "22", "33"], dtype=FIELD_DTYPE.STRING.value)

    report(
        "a NumPy fixed-width unicode array is writable (claim 1)",
        round_trip("<U", unicode_values),
    )
    report(
        "a bytes array is writable -- why the existing fixtures work (claim 2)",
        round_trip("|S", bytes_values),
    )
    report(
        "the h5py variable-length UTF-8 dtype is writable (claim 3)",
        round_trip("vlen utf-8", vlen_values),
    )

    # --- Claim 4: does a value survive as the same Python type? ---
    report(
        "FIELD_DTYPE.STRING is h5py's vlen str dtype, already used for phase_name/mask",
        f"{FIELD_DTYPE.STRING.value!r}, kind={np.dtype(FIELD_DTYPE.STRING.value).kind!r}, "
        f"metadata={np.dtype(FIELD_DTYPE.STRING.value).metadata}",
    )
    report(
        "a `<U` array converts to the vlen dtype without loss, so coercion is available",
        f"np.asarray(['11','22','33'], dtype=FIELD_DTYPE.STRING.value) -> "
        f"{list(np.asarray(unicode_values, dtype=FIELD_DTYPE.STRING.value))}",
    )

    # --- Claim 5: the name path is unaffected, and is a separate mechanism ---
    awkward = ["HB2B:Mot:sz_real", "2theta", "a b/c$d", "with.dot", "__weird__"]
    encoded = {name: allowed_identifier(name) for name in awkward}
    report(
        "every awkward PV-log NAME encodes and round-trips (claim 5)",
        "; ".join(f"{name!r}->{enc!r} rt={decode_identifier(enc) == name}" for name, enc in encoded.items()),
    )
    report(
        "a field under such a name writes fine when its VALUES are numeric (claim 5)",
        "; ".join(named_field_written(enc) for enc in encoded.values()),
    )
    report(
        "and a field under a PLAIN name still fails when its VALUES are `<U` (claim 5)",
        round_trip("plain name, <U values", unicode_values),
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
