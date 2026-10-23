"""A5: the current behaviour of the landed code spec 04 proposes to change.

Spec 04 is entirely about in-repo NXstress internals, so every one of its
claims is A5 — a claim about how a module in this repository behaves. The first
pass checked its *citations* (A3) and found them accurate, but never executed
the code, leaving A5 partial. This closes that: each claim below is a statement
about what the landed library does **today**, which is what makes 04's proposed
change necessary or not.

Claims under test
-----------------
1. ``04:71-73`` / ``README.md:166-167`` -- "Extend ``allowed_identifier`` to
   cover at minimum ``$``, whitespace, and any other characters disallowed by
   the NXstress/HDF5 group-name rules" — i.e. it does not cover them today.
2. ``04:61-67`` / ``README.md:147-150`` -- "**Mask naming/storage
   inconsistency** … Default vs. named masks are addressed differently in the
   current writer."
3. ``04:77-80`` / ``README.md:134-136`` -- "**Instrument name hardcoded to
   ``"HB2B"``**".
4. ``04:84-88`` -- the rotation-order cross-check, and the question spec 04's
   Follow-up 1 F1.4 left open: *which* convention is the referent. The
   transformation chain written by ``_instrument.py`` is reported here so the
   comparison can be made against something concrete rather than recalled.

Run: ``pixi run python plans/NXstress-prod/probes/a5_nxstress_internals_today.py``
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from nexusformat.nexus import NXfield  # noqa: E402

from pyrs.utilities.NXstress._definitions import allowed_identifier  # noqa: E402


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: what spec 04's targets do today")
    print("=" * 78)

    # --- Claim 1: what does allowed_identifier actually sanitise? ---
    cases = ["a:b", "has space", "dollar$sign", "tab\there", "dot.ok", "slash/bad", "newline\nbad", "unicode-ok"]

    def show(s: str) -> str:
        """Render control characters visibly; ruff targets py311, so keep the
        escaping out of the f-strings below."""
        return s.replace("\t", "\\t").replace("\n", "\\n")

    results = {show(c): show(allowed_identifier(c)) for c in cases}
    unchanged = [show(c) for c in cases if c == allowed_identifier(c) and c != "dot.ok"]
    report(
        "`allowed_identifier` does NOT yet cover `$`, whitespace, or other disallowed characters (claim 1)",
        f"{results}; passed through unchanged: {unchanged}",
    )
    report(
        "…and its whole implementation is a single `:` replacement",
        f"source: {inspect.getsource(allowed_identifier).strip().splitlines()[-1].strip()}",
    )

    # HDF5 itself forbids '/' in a link name -- worth knowing which of the
    # pass-through characters are actually dangerous.
    import h5py
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="a5_ident_"))
    # "Accepted" is not the same as "harmless": report the group's REAL name and
    # the file's top-level keys, or a silent restructuring reads as a pass.
    outcomes = {}
    with h5py.File(tmp / "probe.h5", "w") as f:
        for name in ("dollar$sign", "has space", "slash/bad"):
            try:
                grp = f.create_group(allowed_identifier(name))
                nested = grp.name.strip("/").count("/") > 0
                outcomes[name] = f"accepted, stored at {grp.name!r}" + (
                    " -- SILENTLY NESTED, not one group" if nested else ""
                )
            except Exception as exc:  # noqa: BLE001
                outcomes[name] = f"{type(exc).__name__}: {str(exc)[:46]}"
        top_level = sorted(f.keys())
    report(
        "…and what each actually does to the file (claim 1, sharpened)",
        "\n          ".join(f"{k!r}: {v}" for k, v in outcomes.items())
        + f"\n          top-level groups afterwards: {top_level}"
        + "\n          NOTE: '$' and ' ' are legal HDF5 link names -- it is the NeXus"
        + "\n          convention, not HDF5, that disallows them. '/' is the dangerous"
        + "\n          one: it does not raise, it silently creates a NESTED group.",
    )

    # --- Beyond claim 1: is the conversion many-to-one, and is that detected? ---
    # `_sample.py:127-137` iterates EVERY sample log, converts the key, and
    # writes into a flat NXcollection. If two keys converge, the second
    # overwrites the first -- including the `local_name` attribute that is
    # supposed to preserve the original PV name.
    from nexusformat.nexus import NXcollection

    logs = NXcollection()
    pv_names = ["HB2B:CS:X", "HB2B_CS_X"]
    for key in pv_names:
        logs[allowed_identifier(key)] = NXfield([1.0, 2.0], local_name=key)
    report(
        "the converter is MANY-TO-ONE and nothing checks for collisions (not claimed anywhere -- found by probing)",
        f"distinct PV logs {pv_names} both convert to "
        f"{allowed_identifier(pv_names[0])!r}; groups actually written: "
        f"{list(logs.keys())}; surviving local_name attribute: "
        f"{logs[allowed_identifier(pv_names[0])].attrs.get('local_name')!r} -- "
        f"the first log is gone, with no record that it existed.",
    )

    # --- Claim 2: the default-vs-named mask asymmetry ---
    from pyrs.utilities.NXstress._fit import _Fit

    src = inspect.getsource(_Fit.init_group)
    asymmetry = [ln.strip() for ln in src.splitlines() if "discard" in ln or "DEFAULT_TAG" in ln]
    report(
        "default vs named masks are addressed differently in the writer (claim 2)",
        f"`_Fit.init_group` normalises by dropping `None` and injecting the default tag: "
        f"{asymmetry} — so the on-disk name for the default mask is DEFAULT_TAG while "
        f"the in-memory key is `None`, which is exactly the asymmetry the TODO describes.",
    )

    # --- Claim 3: the hardcoded instrument name ---
    from pyrs.utilities.NXstress._instrument import _Instrument

    inst_src = inspect.getsource(_Instrument.init_group)
    hardcoded = [ln.strip() for ln in inst_src.splitlines() if '"HB2B"' in ln]
    report(
        'instrument name is hardcoded to "HB2B" (claim 3)',
        f"{hardcoded or 'NOT FOUND — claim would be stale'}",
    )

    # --- Claim 4: the transformation chain, in the order actually written ---
    chain = []
    for ln in inst_src.splitlines():
        s = ln.strip()
        if s.startswith("(") and ('"rotation' in s or '"translation' in s or '"distance"' in s or '"two_theta' in s):
            chain.append(s.rstrip(","))
    report(
        "the NXtransformations chain order, for the cross-check spec 04 requires "
        "and its Follow-up F1.4 left open (claim 4)",
        "written in this order, each depending on the previous:\n          " + "\n          ".join(chain),
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
