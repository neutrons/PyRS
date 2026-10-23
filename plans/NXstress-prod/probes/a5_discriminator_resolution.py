"""A5: does 04b's discriminator resolver work against the real HidraWorkspace?

04b specifies a bidirectional resolver -- a "get" half used at write time and a
symmetric "set" half used at read time -- in two code blocks, against a class
neither 04b nor this series owns. Every branch of both blocks is a claim about
``HidraWorkspace``'s live surface, and none had been executed.

Claims under test
-----------------
1. ``04b:252-257`` -- ``isinstance(getattr(type(ws), name, None), property)``
   selects a genuine ``@property`` accessor.
2. ``04b:272-279`` -- the ``isinstance(..., property)`` check, rather than a
   bare ``hasattr``, is what stops a discriminator name colliding with an
   *unrelated method* name (04b names ``save_experimental_data``) and returning
   a bound method instead of a value.
3. ``04b:280-283`` -- the fallback ``ws.get_sample_log_value(name)``
   "already returns the single value when every sub-run agrees, and raises
   otherwise. No new constancy-checking code is needed." Both halves, plus what
   it does for a name absent from the logs.
4. ``04b:263-270`` -- the set half's ``prop.fset is not None`` test, "so a
   read-only property with the same name as a discriminator field falls back to
   the log path rather than raising on ``setattr``". Whether any
   ``HidraWorkspace`` property has an ``fset`` at all decides whether that
   branch has a live subject today.
5. ``04b:284-285`` -- the set fallback reuses
   ``HidraWorkspace.set_sample_log(name, sub_runs, values, units="")``.

The resolver is reproduced **locally** here rather than imported: it does not
exist yet, and once it does, a probe importing it would be testing the
implementation instead of the claim about ``HidraWorkspace`` that the
implementation rests on.

Run: ``pixi run python plans/NXstress-prod/probes/a5_discriminator_resolution.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyrs.core.workspaces import HidraWorkspace  # noqa: E402


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def resolve(ws: HidraWorkspace, name: str):
    """04b:252-257's get half, reproduced locally."""
    if name.isidentifier() and isinstance(getattr(type(ws), name, None), property):
        return getattr(ws, name)
    return ws.get_sample_log_value(name)


def apply_(ws: HidraWorkspace, name: str, value) -> None:
    """04b:263-270's set half, reproduced locally."""
    prop = getattr(type(ws), name, None) if name.isidentifier() else None
    if isinstance(prop, property) and prop.fset is not None:
        setattr(ws, name, value)
    else:
        sub_runs = ws.get_sub_runs().raw_copy()
        ws.set_sample_log(name, sub_runs, np.full(len(sub_runs), value))


def attempt(label: str, fn) -> str:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - what it raises is the result
        return f"{label}: RAISED {type(exc).__name__}: {exc}"
    return f"{label}: RETURNED {value!r} (type {type(value).__name__})"


def make_workspace() -> HidraWorkspace:
    ws = HidraWorkspace("probe")
    sub_runs = np.array([1, 2, 3])
    ws.set_sample_log("direction", sub_runs, np.array(["11", "11", "11"]))
    ws.set_sample_log("varies", sub_runs, np.array([1.0, 2.0, 3.0]))
    return ws


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: 04b's discriminator resolver against the real HidraWorkspace")
    print("=" * 78)

    props = {name: obj for name, obj in vars(HidraWorkspace).items() if isinstance(obj, property)}
    settable = {name: obj for name, obj in props.items() if obj.fset is not None}
    report(
        "HidraWorkspace's @property accessors, and which are settable (claim 4)",
        f"{len(props)} properties: {sorted(props)}; settable (fset is not None): {sorted(settable) or 'NONE'}",
    )

    ws = make_workspace()

    # --- Claim 1: a real property is selected by the property test ---
    report(
        "a discriminator name matching a real @property resolves via the property (claim 1)",
        attempt("resolve(ws, 'name')", lambda: resolve(ws, "name")),
    )

    # --- Claim 2: an unrelated *method* name must NOT be mistaken for one ---
    method_name = "save_experimental_data"
    via_hasattr = getattr(ws, method_name, None)
    report(
        "a bare hasattr would mis-select an unrelated method name (claim 2)",
        f"getattr(ws, {method_name!r}) -> {type(via_hasattr).__name__} "
        f"{'(a bound method, NOT a value)' if callable(via_hasattr) else ''}; "
        f"isinstance(getattr(type(ws), name), property) = "
        f"{isinstance(getattr(type(ws), method_name, None), property)}",
    )
    report(
        "the property test makes that name fall through to the log path (claim 2)",
        attempt(f"resolve(ws, {method_name!r})", lambda: resolve(ws, method_name)),
    )

    # --- Claim 3: the SampleLogs fallback, all three ways ---
    report(
        "a constant log resolves to its single value, with no new code (claim 3)",
        attempt("resolve(ws, 'direction')", lambda: resolve(ws, "direction")),
    )
    report(
        "a log that varies across scan points raises (claim 3)",
        attempt("resolve(ws, 'varies')", lambda: resolve(ws, "varies")),
    )
    report(
        "a name present in neither place raises (claim 3)",
        attempt("resolve(ws, 'absent_field')", lambda: resolve(ws, "absent_field")),
    )

    # --- Claims 4 and 5: the set half round-trips through the log fallback ---
    def set_then_get() -> str:
        apply_(ws, "written_back", "22")
        return resolve(ws, "written_back")

    report(
        "the set half's log fallback round-trips through resolve (claims 4, 5)",
        attempt("apply_ then resolve 'written_back'", set_then_get),
    )

    # A settable property is the 05 case; no subject exists today, so stub the
    # shape rather than claim it works on a class that has none.
    class _Stubbed(HidraWorkspace):
        @property
        def direction(self):
            return self._probe_direction

        @direction.setter
        def direction(self, value):
            self._probe_direction = value

    stub = _Stubbed("probe-stub")
    stub.set_sample_log("direction", np.array([1, 2, 3]), np.array(["11", "11", "11"]))

    def stub_round_trip() -> str:
        apply_(stub, "direction", "33")
        return f"property={resolve(stub, 'direction')!r}, log still={stub.get_sample_log_value('direction')!r}"

    report(
        "with a settable property present, both halves use it -- the spec-05 shape (claim 4)",
        attempt("stub round trip", stub_round_trip),
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
