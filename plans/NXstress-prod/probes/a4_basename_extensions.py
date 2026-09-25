"""A4: how spec 07's two basename branches actually strip extensions.

Spec 07 states a single rule -- "Any extension on a caller-supplied
``project_file_name`` is stripped and ignored" -- and then implements it with two
different stdlib calls. Whether they agree depends entirely on how many dots the
filename carries, which matters because HB2B NeXus inputs are routinely
double-extensioned (``*.nxs.h5``). This was reasoned about in spec 07's
Follow-up 1 F1.3 but not executed; recorded then as an A4 partial, closed here.

Claims under test
-----------------
1. ``07:45-51`` -- "**Extension is never caller-controlled** … only the
   *basename* varies with ``project_file_name`` … Any extension on a
   caller-supplied ``project_file_name`` is stripped and ignored, never
   validated against and never a reason to raise".
2. ``07:105-109`` (the code block) --
   ``basename = os.path.basename(nexus).split(".")[0]`` when
   ``project_file_name is None``, else
   ``basename = os.path.splitext(os.path.basename(project_file_name))[0]``.
3. ``07:141-144`` -- the Verification case: "pass an explicit
   ``project_file_name`` with a mismatched extension (e.g. ``"foo.nxs"`` …);
   confirm the extension is ignored and a correctly-suffixed ``.h5`` file is
   written under basename ``foo``".

Claim 3 is the reason this matters: the test the spec proposes uses a
*single*-extension name, which is precisely the case in which the two branches
agree. It therefore cannot detect the disagreement.

Run: ``pixi run python plans/NXstress-prod/probes/a4_basename_extensions.py``
"""

from __future__ import annotations

import os
import sys

CASES = [
    "HB2B_1234.nxs.h5",
    "HB2B_1234.h5",
    "foo.nxs",
    "foo",
    "run.2024.03.nxs.h5",
]


def auto_derived(name: str) -> str:
    """The `project_file_name is None` branch, verbatim from spec 07."""
    return os.path.basename(name).split(".")[0]


def caller_supplied(name: str) -> str:
    """The explicit-`project_file_name` branch, verbatim from spec 07."""
    return os.path.splitext(os.path.basename(name))[0]


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: spec 07's two basename branches")
    print("=" * 78)
    print(f"\npython {sys.version.split()[0]}\n")

    print(f"  {'input':<24} {'auto-derived':<20} {'caller-supplied':<20} agree?")
    print(f"  {'-' * 24} {'-' * 20} {'-' * 20} ------")
    disagreements = []
    for name in CASES:
        a, b = auto_derived(name), caller_supplied(name)
        agree = a == b
        if not agree:
            disagreements.append((name, a, b))
        print(f"  {name:<24} {a:<20} {b:<20} {'yes' if agree else 'NO'}")

    print(
        f"\n  CLAIM   the two branches implement one rule -- 'the extension is "
        f"stripped and ignored' (claims 1, 2)"
        f"\n  RESULT  they disagree on {len(disagreements)} of {len(CASES)} cases: "
        f"{[(n, a, b) for n, a, b in disagreements]}"
    )

    spec_case = "foo.nxs"
    print(
        f"\n  CLAIM   spec 07's Verification case ({spec_case!r}) detects the difference (claim 3)"
        f"\n  RESULT  auto-derived={auto_derived(spec_case)!r}, "
        f"caller-supplied={caller_supplied(spec_case)!r} -- identical, so the "
        f"proposed test CANNOT detect it. A double-extension name can: "
        f"'HB2B_1234.nxs.h5' -> {auto_derived('HB2B_1234.nxs.h5')!r} vs "
        f"{caller_supplied('HB2B_1234.nxs.h5')!r}."
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
