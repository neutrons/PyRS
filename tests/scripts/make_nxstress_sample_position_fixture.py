#!/usr/bin/env python
"""Build the ``*_with_instrument.h5`` fixtures: real HB2B project files with geometry.

Run by hand; not collected by pytest (``tests/scripts`` is in ``norecursedirs``):

    pixi run python tests/scripts/make_nxstress_sample_position_fixture.py [TARGET ...]

With no ``TARGET``, every fixture that does not exist yet is built, and an existing
one is left alone. Naming a target (``HB2B_1327``, ...) rebuilds it. The targets
are listed in ``FIXTURES``.

Why these fixtures exist
------------------------
Writing NXstress needs **instrument geometry**, and the project files these
fixtures start from have none: each has an ``instrument/geometry setup/detector``
group, but it is **empty** -- the three datasets it should contain are simply
absent, which is why ``get_instrument_setup()`` returns ``None``. So none of them
can be saved as NXstress.

Each fixture copies one real file and fills that group in from
``HB2B_938_peak.h5``, another real HB2B file in this repository, so the values are
real instrument parameters rather than invented ones. Everything else -- sample
logs, reduced diffraction data and fit results -- is copied verbatim, and the
original file is left in place.

* ``HB2B_1327`` -- spec 04's ``## Verification`` asks for a ``.nxs`` written "from
  a real HB2B dataset" with ``peaks/sx,sy,sz`` non-NaN. Every file with finite
  ``vx``/``vy``/``vz`` lacked geometry, and every file with geometry had
  non-finite coordinates (spec 04's Follow-up 7 F7.10); 1327 was chosen for its
  coordinates. A test whose subject were synthesised would be circular, which is
  why the base file was chosen for its coordinates rather than for its geometry.
* ``HB2B_1599_texture`` -- TextureFitting offers *Save as NXstress...* only for a
  workspace with geometry, and no texture project file (reduced data that are
  out-of-plane ``eta_<angle>`` patterns) has any. This one lets the viewer's
  handling of eta fits at NXstress save run on real data.

Peak tags
---------
1327's tags are renamed, ``peakN`` -> ``peak`` + the digit three times, because
``_Peaks._parse_peak_tag`` needs a digit run whose length is a multiple of three
and refuses ``peak0``. This gives phase name ``peak`` with Miller indices (0,0,0)
and (1,1,1). **The indices carry no physical claim** -- the source records
``d reference`` as 1.0 for one peak and NaN for the other, so these tags never
identified a reflection.

The mask
--------
The ``mask`` group is dropped from every fixture. It is 8.4 MB of a typical
source file's 8.5-11 MB, which would breach pre-commit's 8 MB ceiling, and
NXstress generates a default mask when a workspace carries none -- so keeping it
would cost the repository 8 MB per file to exercise nothing these fixtures are for.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests/data"
GEOMETRY_DONOR = DATA / "HB2B_938_peak.h5"

#: pre-commit's `check-added-large-files` ceiling. Named so the refusal message
#: cannot drift from the comparison -- they were two separate literals.
SIZE_CEILING_MB = 8.0

GEOMETRY_PATH = "instrument/geometry setup/detector"
SKIP_TOP_LEVEL = {"mask"}


@dataclass(frozen=True)
class Fixture:
    """One fixture: a real project file, plus geometry.

    Attributes:
        source: The stem of the file in ``tests/data`` it is copied from; the
            fixture is ``<source>_with_instrument.h5`` beside it.
        peak_tag_renames: Peak tags to rename on the way, old -> new.
    """

    source: str
    peak_tag_renames: dict[str, str] = field(default_factory=dict)

    @property
    def source_path(self) -> Path:
        return DATA / f"{self.source}.h5"

    @property
    def target_path(self) -> Path:
        return DATA / f"{self.source}_with_instrument.h5"


FIXTURES = {
    fixture.source: fixture
    for fixture in (
        Fixture("HB2B_1327", peak_tag_renames={"peak0": "peak000", "peak1": "peak111"}),
        Fixture("HB2B_1599_texture"),
    )
}


def main(argv: list[str]) -> int:
    unknown = [name for name in argv if name not in FIXTURES]
    if unknown:
        print(f"unknown target(s) {unknown}; choose from {sorted(FIXTURES)}", file=sys.stderr)
        return 1
    if argv:
        selected = [FIXTURES[name] for name in argv]
    else:
        selected = [fixture for fixture in FIXTURES.values() if not fixture.target_path.exists()]
        if not selected:
            print("every fixture exists already; name a target to rebuild it")
            return 0

    if not GEOMETRY_DONOR.is_file():
        print(f"missing input: {GEOMETRY_DONOR}", file=sys.stderr)
        return 1
    with h5py.File(GEOMETRY_DONOR, "r") as donor:
        geometry = {key: donor[f"{GEOMETRY_PATH}/{key}"][()] for key in donor[GEOMETRY_PATH]}
    if not geometry:
        print(f"{GEOMETRY_DONOR} has no geometry to copy", file=sys.stderr)
        return 1
    print(f"geometry from {GEOMETRY_DONOR.name}: { {k: v.tolist() for k, v in geometry.items()} }")

    failed = 0
    for fixture in selected:
        staging = fixture.target_path.with_name(fixture.target_path.name + ".staging")
        try:
            failed += _build(fixture, geometry, staging)
        finally:
            # A failure anywhere between creating the staging file and moving it must
            # not leave it in `tests/data`.
            staging.unlink(missing_ok=True)
    return 1 if failed else 0


def _build(fixture: Fixture, geometry: dict, staged: Path) -> int:
    if not fixture.source_path.is_file():
        print(f"missing input: {fixture.source_path}", file=sys.stderr)
        return 1

    # Built beside the target, then moved into place only once the size is
    # acceptable. Writing the target directly would truncate a committed fixture
    # before anything had been checked, so a refusal would leave the repository
    # with a destroyed -- or, if the script then removed it, a deleted -- tracked
    # file. Nothing here deletes a file it did not create.
    with h5py.File(fixture.source_path, "r") as src, h5py.File(staged, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        for name in src:
            if name in SKIP_TOP_LEVEL:
                print(f"{fixture.source}: skipping {name!r} (see module docstring)")
                continue
            if name == "peaks":
                peaks = dst.create_group("peaks")
                for key, value in src["peaks"].attrs.items():
                    peaks.attrs[key] = value
                for tag in src["peaks"]:
                    renamed = fixture.peak_tag_renames.get(tag, tag)
                    src.copy(src[f"peaks/{tag}"], peaks, name=renamed)
                    if renamed != tag:
                        print(f"{fixture.source}: renamed peak tag {tag!r} -> {renamed!r}")
                continue
            src.copy(src[name], dst, name=name)

        detector = dst.require_group(GEOMETRY_PATH)
        for key, value in geometry.items():
            if key in detector:
                del detector[key]
            detector.create_dataset(key, data=value)
        print(f"{fixture.source}: populated {GEOMETRY_PATH} with {sorted(geometry)}")

    size_mb = staged.stat().st_size / 1e6
    if size_mb > SIZE_CEILING_MB:
        # Refused before the target is touched, so an existing fixture is left exactly
        # as it was. Only the staging file -- which this script created -- is removed.
        print(
            f"refusing to install {fixture.target_path.name}: {size_mb:.2f} MB exceeds pre-commit's "
            f"{SIZE_CEILING_MB:.0f} MB ceiling.\n  {fixture.target_path} is unchanged.",
            file=sys.stderr,
        )
        return 1

    staged.replace(fixture.target_path)
    print(f"wrote {fixture.target_path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
