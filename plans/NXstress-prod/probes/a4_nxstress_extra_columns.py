"""A4: does the NXstress application definition permit extra `peaks` columns?

04b's Q1 asks whether ``NXreflections`` tolerates index columns beyond the
schema's own, because the whole discriminator mechanism is one such column. The
question was downgraded from a blocking gate to a tracked follow-up on a
*precedent* argument -- ``_peaks.py::_init`` already writes ``mask`` and
``scan_point``, so the format evidently tolerates it -- with the note that the
real cross-check must wait "once both [the schema doc and the validator] land in
the repo".

The schema half has since landed: Decisions row 27(b) vendored
``NXstress.nxdl.xml`` at NXDL v2026.01. So the precedent argument can now be
replaced by reading the application definition itself. The validator has **not**
landed, and that remains the gap ``a4_nexusformat_validator.py`` tracks.

Claims under test
-----------------
1. 04b's schema-precedent note and ``open-questions/04b`` Q1 -- ``mask`` and
   ``scan_point`` are written to ``peaks`` by ``_peaks.py::_init`` and are *not*
   declared by the schema, so a non-declared column is already shipping.
2. 04b Q1 -- "a discriminator column is the same category of extension
   ``NXreflections`` already tolerates in practice". Under NXDL, an application
   definition lists the *minimum* required content unless it carries
   ``restricts``; this probe reports which of the two ``<definition>`` has.
3. 04b's `## NXstress Changes` collision guard -- the reserved-column list the
   guard is written against must match what ``_peaks.py::_init`` actually
   writes, or the guard protects the wrong set.

This reads the vendored XML and the real ``_init`` output. It does **not** grep
the source for field names: per ``probes/README.md``'s failure-modes table,
reading source tells you what is written, not what the output contains -- so the
group is built and traversed.

Run: ``pixi run python plans/NXstress-prod/probes/a4_nxstress_extra_columns.py``
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SCHEMA = REPO / "docs" / "developer" / "source" / "design" / "nexus" / "NXstress.nxdl.xml"
NXDL_NS = "{http://definition.nexusformat.org/nxdl/3.1}"


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def declared_peak_fields(root: ET.Element) -> dict[str, str]:
    """Field names the schema declares under the `peaks` group, with minOccurs."""
    for group in root.iter(f"{NXDL_NS}group"):
        if group.get("name") == "peaks":
            return {field.get("name"): field.get("minOccurs", "1") for field in group.findall(f"{NXDL_NS}field")}
    raise RuntimeError("no group named 'peaks' in the vendored schema")


def written_peak_fields() -> list[str]:
    """Field names `_peaks.py::_init` really puts on the group, by traversal."""
    from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns
    from pyrs.utilities.NXstress._peaks import _Peaks

    logs = SampleLogs()
    logs.subruns = SubRuns([1, 2, 3])
    return list(_Peaks._init(logs))


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: does the vendored NXstress schema permit extra `peaks` columns?")
    print("=" * 78)

    if not SCHEMA.exists():
        report(
            "the vendored NXstress application definition is present",
            f"ABSENT at {SCHEMA.relative_to(REPO)} -- the absence IS the finding",
        )
        return 1

    root = ET.parse(SCHEMA).getroot()
    report(
        "the vendored schema is present and parses",
        f"{SCHEMA.relative_to(REPO)}, definition name={root.get('name')!r}, "
        f"category={root.get('category')!r}, extends={root.get('extends')!r}",
    )

    # --- Claim 2: application definitions state a minimum unless `restricts` ---
    restricts = root.get("restricts")
    report(
        "the definition does NOT carry `restricts`, so it states a minimum, not a closed set",
        f"restricts={restricts!r}; attributes present: {sorted(k for k in root.attrib if '}' not in k)}",
    )

    # --- Claim 1: the schema's own list vs. what PyRS writes today ---
    declared = declared_peak_fields(root)
    written = written_peak_fields()
    undeclared = [n for n in written if n not in declared]
    unwritten = [n for n in declared if n not in written]

    report(
        "schema-declared `peaks` fields (name -> minOccurs)",
        f"{len(declared)}: {declared}",
    )
    report(
        "fields `_peaks.py::_init` actually writes (built and traversed, not grepped)",
        f"{len(written)}: {written}",
    )
    report(
        "PyRS already ships `peaks` columns the schema does not declare (claim 1)",
        f"undeclared-but-written = {undeclared}; declared-but-unwritten = {unwritten}",
    )

    # --- Claim 3: the collision guard's reserved set must equal what is written ---
    guard_set = {
        "h",
        "k",
        "l",
        "mask",
        "scan_point",
        "center",
        "center_errors",
        "center_type",
        "sx",
        "sy",
        "sz",
        "qx",
        "qy",
        "qz",
        "phase_name",
    }
    report(
        "04b's reserved-column list matches what `_init` writes (claim 3)",
        f"written-not-in-guard = {sorted(set(written) - guard_set)}; "
        f"guard-not-written = {sorted(guard_set - set(written))}; "
        f"equal={guard_set == set(written)}",
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
