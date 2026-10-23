"""A4: how do you detect a non-resizable field, and what happens if you resize anyway?

``a5_scan_point_family_resizable.py`` establishes that the landed NXstress writer
emits most of the scan-point family as contiguous, fixed-size HDF5 datasets. 04c
must therefore grow the ones it can and **refuse, clearly**, on a file written
before that change -- which needs two things this probe supplies:

1. a way to ask a reopened ``nexusformat`` ``NXfield`` whether its first axis is
   extendable, without reaching past ``nexusformat`` into raw h5py at every call
   site; and
2. knowledge of what ``NXfield.resize`` actually *does* to a fixed-size dataset,
   so the guard is written against the real failure rather than an assumed one.

Per ``probes/README.md``'s failure-modes table -- "'Accepted' is not 'harmless'.
Report the resulting *state*, not just the absence of an exception" -- every
check below prints what the file looks like afterwards, not merely whether
something raised.

Claims under test
-----------------
1. New, asserted by no document yet: a reopened ``NXfield`` exposes enough to
   decide "extendable along axis 0" -- and what the right accessor is.
2. New: ``NXfield.resize`` on a contiguous dataset fails rather than silently
   succeeding, corrupting, or truncating.
3. ``04c-nxstress-append.md:175-181`` -- "a rejected (``RuntimeError``) ...
   append [is a] true no-op". A failed resize part-way through must not be
   reachable, so the pre-flight check in claim 1 has to be complete; this
   measures whether a *failed* resize leaves the dataset untouched, which is the
   backstop if it ever is reached.

Run: ``pixi run python plans/NXstress-prod/probes/a4_nxfield_maxshape_introspection.py``
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np
from nexusformat.nexus import NXdata, NXentry, NXfield, nxopen


def CHUNK_SHAPE(rank: int) -> tuple[int, ...]:
    return (1,) * (rank - 1) + (100,)


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: detecting a non-resizable NXfield, and resizing one anyway")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="a4_maxshape_"))
    path = tmp / "probe.nxs"

    # Two fields written the two ways `pyrs/utilities/NXstress/` writes them today.
    with nxopen(path, "w") as root:
        root["entry"] = NXentry()
        root["entry"]["data"] = NXdata()
        root["entry"]["data"]["fixed"] = NXfield(np.arange(3, dtype=np.float64))
        root["entry"]["data"]["growable"] = NXfield(
            np.arange(3, dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1)
        )

    # --- Claim 1: what can a reopened NXfield tell us? ---
    with nxopen(path, "r") as root:
        fixed = root["entry"]["data"]["fixed"]
        grow = root["entry"]["data"]["growable"]
        attrs = [a for a in ("maxshape", "chunks", "shape", "nxfilemode") if hasattr(fixed, a)]
        detail = []
        for label, field in (("fixed", fixed), ("growable", grow)):
            detail.append(
                f"{label}: "
                + ", ".join(f"{a}={getattr(field, a)!r}" for a in attrs)
                + f", type={type(field).__name__}"
            )
    report(
        "claim 1: does a reopened NXfield expose maxshape, so 'extendable' is decidable?",
        f"NXfield attributes present: {attrs or 'NONE of maxshape/chunks/shape/nxfilemode'}\n          "
        + "\n          ".join(detail),
    )

    # If nexusformat does not surface it, the h5py handle underneath does -- record
    # which accessor a guard must actually use.
    with h5py.File(path, "r") as f:
        h5_detail = (
            f"fixed: maxshape={f['entry/data/fixed'].maxshape} chunks={f['entry/data/fixed'].chunks}  |  "
            f"growable: maxshape={f['entry/data/growable'].maxshape} chunks={f['entry/data/growable'].chunks}"
        )
    report("claim 1 (fallback): the same question asked through raw h5py", h5_detail)

    # --- Claims 2 and 3: resize the fixed one and report the resulting STATE ---
    before = digest(path)
    outcome = "no exception raised"
    try:
        with nxopen(path, "rw") as root:
            root["entry"]["data"]["fixed"].resize((5,))
    except Exception as error:  # noqa: BLE001 -- the exception type IS the finding
        outcome = f"{type(error).__name__}: {error}"
    after = digest(path)

    with h5py.File(path, "r") as f:
        state = (
            f"shape={f['entry/data/fixed'].shape} "
            f"maxshape={f['entry/data/fixed'].maxshape} "
            f"values={f['entry/data/fixed'][()].tolist()}"
        )
    report(
        "claims 2-3: resizing a FIXED-SIZE reopened dataset -- what happens, and what is left behind?",
        f"outcome: {outcome}\n          "
        f"resulting state: {state}\n          "
        f"file sha256[:16] before={before} after={after} identical={before == after}",
    )

    # Control: the same call on the growable one must succeed, so the check above
    # is measuring resizability and not some unrelated breakage.
    control = "no exception raised"
    try:
        with nxopen(path, "rw") as root:
            field = root["entry"]["data"]["growable"]
            cur = field.shape[0]
            field.resize((cur + 2,))
            field[cur:] = np.array([9.0, 9.0])
    except Exception as error:  # noqa: BLE001
        control = f"{type(error).__name__}: {error}"
    with h5py.File(path, "r") as f:
        control_state = f"shape={f['entry/data/growable'].shape} values={f['entry/data/growable'][()].tolist()}"
    report(
        "control: the same resize on the GROWABLE dataset -- the check is measuring resizability",
        f"outcome: {control}\n          resulting state: {control_state}",
    )

    print(f"\n  (artifact left at {path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
