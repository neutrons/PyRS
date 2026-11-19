"""
The vendored NeXus definitions: an abbreviated, verbatim copy of `nexusformat/definitions`.

`docs/developer/source/design/nexus/NeXuS-definitions/` holds the NXstress
application definition, the base classes it names, and the NXDL schema language
(`nxdl.xsd`), copied from a release tag. Their provenance is in `IO_prototype.rst`
beside them, and each file's sha256 in `NeXuS-definitions.sha256`. These tests
re-check that record on every commit, instead of leaving it to be re-read:

- every file matches its recorded checksum, and the tree holds exactly the files
  recorded;
- every class the application definition names is present;
- the `validItemName` rule PyRS enforces is the one `nxdl.xsd` defines.

Unit tier: reads only files in the repository.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pyrs.utilities.NXstress._definitions import MAX_IDENTIFIER_LENGTH, VALID_ITEM_NAME

NEXUS_DOCS = Path(__file__).resolve().parents[5] / "docs" / "developer" / "source" / "design" / "nexus"
TREE = NEXUS_DOCS / "NeXuS-definitions"
CHECKSUMS = NEXUS_DOCS / "NeXuS-definitions.sha256"
XSD = "{http://www.w3.org/2001/XMLSchema}"


def recorded_checksums() -> dict[str, str]:
    """`NeXuS-definitions.sha256`, as {path relative to the nexus docs: sha256}."""
    entries = {}
    for line in CHECKSUMS.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        entries[path] = digest
    return entries


@pytest.mark.parametrize("path", sorted(recorded_checksums()))
def test_each_file_matches_its_checksum(path):
    """A vendored file edited by hand, or re-copied from another release, fails here."""
    assert hashlib.sha256((NEXUS_DOCS / path).read_bytes()).hexdigest() == recorded_checksums()[path]


def test_the_tree_holds_exactly_the_recorded_files():
    # Arrange
    on_disk = {str(path.relative_to(NEXUS_DOCS)) for path in TREE.rglob("*") if path.is_file()}

    # Act / Assert
    assert on_disk == set(recorded_checksums())


def test_every_class_nxstress_names_is_vendored():
    """The tree claims to cover what NXstress refers to: each `type=` or `extends=` class is here."""
    # Arrange
    schema = (TREE / "applications" / "NXstress.nxdl.xml").read_text()
    named = set(re.findall(r'(?:type|extends)="(NX[a-z]\w*)"', schema))

    # Act
    missing = sorted(name for name in named if not (TREE / "base_classes" / f"{name}.nxdl.xml").is_file())

    # Assert
    assert named, "no class names found: the pattern no longer matches the schema"
    assert missing == []


def test_the_enforced_name_rule_is_nxdl_xsd_s():
    """`_definitions.VALID_ITEM_NAME` and `MAX_IDENTIFIER_LENGTH` restate `validItemName` from `nxdl.xsd`."""
    # Arrange
    root = ET.parse(TREE / "nxdl.xsd").getroot()
    (rule,) = [element for element in root.iter(f"{XSD}simpleType") if element.get("name") == "validItemName"]

    # Act
    pattern = rule.find(f"{XSD}restriction/{XSD}pattern").get("value")
    max_length = int(rule.find(f"{XSD}restriction/{XSD}maxLength").get("value"))

    # Assert
    assert pattern == VALID_ITEM_NAME
    assert max_length == MAX_IDENTIFIER_LENGTH
