"""`resolve_depends_on`: `NXtransformations` chains resolved by the NeXus rule.

`NXtransformations.nxdl.xml`: a relative `depends_on` "is relative to the group
enclosing the `depends_on` specification". The writers once put
`./transformations/<previous>` on fields that are themselves inside
`transformations/` -- a path naming `transformations/transformations/<previous>`.
Tests that reduced each value to its basename could not see it.
"""

import pytest
from nexusformat.nexus import NXdetector, NXentry, NXfield, NXinstrument, NXroot, NXtransformations, nxload

from pyrs.utilities.NXstress._transformations import resolve_depends_on


def _detector(entry_point: str, links: dict[str, str]) -> NXdetector:
    """A detached detector whose `transformations` holds `links`, name -> `depends_on`."""
    detector = NXdetector()
    transformations = NXtransformations()
    for name, depends_on in links.items():
        field = NXfield(0.0)
        field.attrs["depends_on"] = depends_on
        transformations[name] = field
    detector["transformations"] = transformations
    detector["depends_on"] = entry_point
    return detector


def _names(detector) -> list[str]:
    return [link.nxname for link in resolve_depends_on(detector)]


def test_sibling_names_resolve_in_traversal_order():
    detector = _detector("transformations/c", {"a": ".", "b": "a", "c": "b"})
    assert _names(detector) == ["c", "b", "a"]


def test_entry_point_of_dot_is_an_empty_chain():
    assert _names(_detector(".", {})) == []


def test_dot_and_dot_dot_segments_resolve():
    detector = _detector("./transformations/b", {"a": ".", "b": "../transformations/a"})
    assert _names(detector) == ["b", "a"]


def test_absolute_path_resolves_from_the_root():
    root = NXroot()
    root["entry"] = NXentry()
    root["entry/instrument"] = NXinstrument()
    root["entry/instrument/detector"] = _detector(
        "transformations/b", {"a": ".", "b": "/entry/instrument/detector/transformations/a"}
    )
    assert _names(root["entry/instrument/detector"]) == ["b", "a"]


def test_resolves_in_a_file(tmp_path):
    root = NXroot()
    root["entry"] = NXentry()
    root["entry/detector"] = _detector("transformations/b", {"a": ".", "b": "a"})
    root.save(tmp_path / "chain.nxs")
    assert _names(nxload(tmp_path / "chain.nxs", "r")["entry/detector"]) == ["b", "a"]


def test_old_writer_form_names_nothing():
    """The pre-fix writer's form: a link path written as if from the detector."""
    detector = _detector("./transformations/b", {"a": ".", "b": "./transformations/a"})
    with pytest.raises(RuntimeError, match=r"names nothing.*'transformations' has no 'transformations'"):
        resolve_depends_on(detector)


@pytest.mark.parametrize(
    ("entry_point", "links", "message"),
    [
        ("transformations/b", {"a": ".", "b": "nope"}, "names nothing"),
        ("transformations/b", {"a": "b", "b": "a"}, "cycle"),
        ("transformations/b", {"a": "", "b": "a"}, "is empty"),
        ("", {}, "is empty"),
        ("transformations", {"a": "."}, "names a group"),
        ("transformations/a/x", {"a": "."}, "is not a group"),
        ("../../x", {}, "climbs above the root"),
    ],
)
def test_broken_chain_raises(entry_point, links, message):
    with pytest.raises(RuntimeError, match=message):
        resolve_depends_on(_detector(entry_point, links))


def test_missing_entry_point_raises():
    with pytest.raises(RuntimeError, match="has no `depends_on`"):
        resolve_depends_on(NXdetector())


def test_link_without_depends_on_raises():
    detector = _detector("transformations/a", {})
    detector["transformations/a"] = NXfield(0.0)
    with pytest.raises(RuntimeError, match="'a' has no `depends_on` attribute"):
        resolve_depends_on(detector)
