"""A4: the authoritative NeXus identifier rule, and what it costs us.

Follow-up 4 F4.2 rejected `.` as an escape marker because its legality was
"not verifiable in this repo" -- no `validItemName` rule existed in
`nexusformat` or here, so the only support was an unsourced comment in
`_definitions.py`. Decision 23 then adopted the Python-identifier rule
*specifically* because it needs no external specification.

That premise no longer holds. The rule is published in the NeXus definitions
repository, in `nxdl.xsd` -- not in the application definition, which is why
vendoring `NXstress.nxdl.xml` alone does not bring it along. This probe pins
the rule, and measures what adopting it buys over Decision 23's rule.

Two things it establishes that the Python-identifier rule gets wrong:

* `.` **is** legal in the interior of an identifier, so the unsourced comment
  at `_definitions.py` was right all along;
* a **leading digit** is legal, and `2theta` / `2thetaSetpoint` are real log
  names in `tests/data`. Decision 23's rule encodes them needlessly.

And one constraint nothing in the series had accounted for: names are capped at
**63 characters**, while encoding *lengthens* them.

The authority is a sibling checkout, not a package. When it is not reachable the
probe still runs and says so -- the claim then rests on the recorded provenance
in `docs/developer/source/design/nexus/IO_prototype.rst` rather than on a live
read, and that distinction is reported rather than hidden.

Run: ``pixi run python plans/NXstress-prod/probes/a4_validitemname_rule.py``
"""

from __future__ import annotations

import glob
import os
import pathlib
import re
import sys

import h5py

# The claim, recorded here so the probe fails loudly if upstream ever changes it.
CLAIMED_PATTERN = "[a-zA-Z0-9_]([a-zA-Z0-9_.]*[a-zA-Z0-9_])?"
CLAIMED_MAXLEN = 63

# Provenance of the vendored application definition (see IO_prototype.rst).
NXDL_VERSION = "v2026.01"
NXDL_COMMIT = "004da96ef29bc6b7529e1f8e5d9415cd000bccb8"

SEARCH_ROOTS = [
    os.environ.get("NEXUS_DEFINITIONS", ""),
    "/home/ux0/workspaces/NeXuS-definitions",
    "../NeXuS-definitions",
    "../definitions",
]


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def find_xsd() -> pathlib.Path | None:
    for root in SEARCH_ROOTS:
        if not root:
            continue
        candidate = pathlib.Path(root).expanduser() / "nxdl.xsd"
        if candidate.is_file():
            return candidate
    return None


def parse_rule(xsd_text: str) -> tuple[str | None, int | None]:
    """Extract the pattern and maxLength of the `validItemName` simpleType."""
    block = re.search(
        r'<xs:simpleType\s+name="validItemName">(.*?)</xs:simpleType>',
        xsd_text,
        re.DOTALL,
    )
    if block is None:
        return None, None
    body = block.group(1)
    pattern = re.search(r'<xs:pattern\s+value="([^"]*)"', body)
    maxlen = re.search(r'<xs:maxLength\s+value="(\d+)"', body)
    return (
        pattern.group(1) if pattern else None,
        int(maxlen.group(1)) if maxlen else None,
    )


def real_log_names() -> set[str]:
    """Every dataset name in the sampled real project files."""
    names: set[str] = set()
    for path in sorted(glob.glob("tests/data/*.h5"))[:6]:
        try:
            with h5py.File(path, "r") as handle:

                def walk(group):
                    for key in group:
                        try:
                            if isinstance(group[key], h5py.Group):
                                walk(group[key])
                            else:
                                names.add(key)
                        except Exception:  # noqa: BLE001 - malformed entries are not the subject
                            pass

                walk(handle)
        except Exception:  # noqa: BLE001
            pass
    return names


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: the authoritative NeXus `validItemName` rule")
    print("=" * 78)

    # --- Is the authority reachable at all? ---
    xsd = find_xsd()
    if xsd is None:
        report(
            "the upstream NeXus definitions checkout is reachable",
            "FALSE -- no `nxdl.xsd` found. Set NEXUS_DEFINITIONS to a checkout of "
            "nexusformat/definitions to verify the rule live. The rule below is the "
            f"RECORDED claim ({NXDL_VERSION}, {NXDL_COMMIT[:8]}), not a live read.",
        )
        pattern, maxlen = CLAIMED_PATTERN, CLAIMED_MAXLEN
        verified_live = False
    else:
        pattern, maxlen = parse_rule(xsd.read_text(errors="replace"))
        report(
            "`nxdl.xsd` defines a `validItemName` simpleType with a pattern and a length cap",
            f"pattern={pattern!r} maxLength={maxlen!r}  (read from {xsd})",
        )
        report(
            "the rule matches what this series records",
            f"pattern {'MATCHES' if pattern == CLAIMED_PATTERN else f'DIFFERS -- got {pattern!r}'}; "
            f"maxLength {'MATCHES' if maxlen == CLAIMED_MAXLEN else f'DIFFERS -- got {maxlen!r}'}",
        )
        verified_live = True

    if pattern is None:
        print("\n  cannot continue without a rule")
        return 1

    rule = re.compile(f"^{pattern}$")

    # --- What the rule permits, against what Decision 23 assumed ---
    report(
        "`.` is legal in the interior of an identifier (F4.2 called this unverifiable)",
        f"{bool(rule.match('name.with.dots'))} -- so the unsourced comment in "
        f"`_definitions.py` was correct. Leading/trailing `.`: "
        f"{bool(rule.match('.lead'))}/{bool(rule.match('trail.'))}",
    )
    report(
        "a LEADING DIGIT is legal (Decision 23's Python rule escapes it)",
        f"{bool(rule.match('2theta'))} -- `2theta` and `2thetaSetpoint` are real log "
        f"names, and need no encoding at all under this rule.",
    )
    report(
        "characters this series cared about are rejected",
        ", ".join(f"{c!r}={bool(rule.match('a' + c + 'b'))}" for c in [":", "$", " ", "/", "-", "."]),
    )
    report(
        "`_` is NOT the only legal punctuation (contrast Decision 23)",
        f"legal interior punctuation: {[c for c in '_.$:- @#!' if rule.match('a' + c + 'b')]}",
    )

    # --- The length cap, which nothing in the series had accounted for ---
    names = real_log_names()
    already = {n for n in names if rule.match(n)}
    longest = max(names, key=len) if names else ""
    report(
        "how the rule lands on the REAL namespace",
        f"{len(names)} distinct names sampled from tests/data; {len(already)} already "
        f"valid under this rule, {len(names) - len(already)} need encoding; "
        f"longest name: {longest!r} ({len(longest)} chars, cap {maxlen})",
    )

    # Worst case: every character escapes to 4 chars under the `__XX` form.
    worst = max((len(n) for n in names), default=0) * 4
    report(
        "worst-case encoded length against the cap",
        f"{worst} chars if EVERY character of the longest name escaped, vs cap {maxlen} -- "
        f"{'CAN EXCEED, so the encoder must check' if worst > maxlen else 'cannot exceed'}",
    )

    # --- The vendored application definition, and the fields this PR fills ---
    vendored = pathlib.Path("docs/developer/source/design/nexus/NXstress.nxdl.xml")
    if vendored.is_file():
        text = vendored.read_text(errors="replace")
        required = [f for f in ("sx", "sy", "sz") if re.search(rf'<field name="{f}"[^>]*minOccurs="1"', text)]
        report(
            "the vendored NXstress definition requires peaks/sx,sy,sz",
            f"required fields found: {required}; "
            f"documented as: "
            f"{'sample position in the sample reference frame' if 'sample position in the sample reference frame' in text else 'NOT FOUND'}",
        )
    else:
        report("the NXstress definition is vendored in-repo", f"FALSE -- {vendored} not found")

    print("\n" + "-" * 78)
    print(f"Rule verified against a live checkout: {verified_live}")
    print("Consequence: Decision 23's Python-identifier rule is CONFORMANT but not")
    print("minimal -- it escapes `.` and leading digits that this rule permits.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
