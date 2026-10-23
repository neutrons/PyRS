#!/usr/bin/env python
"""Build the NXstress sample-position integration fixture.

Run by hand; not collected by pytest (``tests/scripts`` is in ``norecursedirs``):

    pixi run python tests/scripts/make_nxstress_sample_position_fixture.py

Why this fixture exists
-----------------------
Spec 04's ``## Verification`` asks for a ``.nxs`` written "from a real HB2B
dataset" with ``peaks/sx,sy,sz`` confirmed non-NaN. No file in ``tests/data``
could satisfy that, and the reason is structural rather than accidental --
the requirements are split across two disjoint sets of files:

* writing NXstress at all needs **instrument geometry** and a **peak tag the
  Miller-index parser accepts**;
* non-NaN sample positions additionally need **finite vx/vy/vz**.

Every file with finite ``vx``/``vy``/``vz`` lacks geometry; every file with
geometry has non-finite coordinates, and of those only ``HB2B_938_peak.h5`` also
has a peak tag. Recorded as spec 04's Follow-up 7 F7.10.

What is real here, and what is not
----------------------------------
The quantity under test is left untouched. ``vx``/``vy``/``vz`` -- and every
other sample log, the reduced diffraction data and the fit results -- are copied
verbatim from ``HB2B_1327.h5``. A test whose subject were synthesised would be
circular, which is why the base file was chosen for its coordinates rather than
for its geometry.

Two peripheral things are supplied:

1. **Instrument geometry.** ``HB2B_1327.h5`` *has* an
   ``instrument/geometry setup/detector`` group, but it is **empty** -- the three
   datasets it should contain are simply absent, which is why
   ``get_instrument_setup()`` returns ``None``. They are filled in from
   ``HB2B_938_peak.h5``, another real HB2B file in this repository, so the values
   are real instrument parameters rather than invented ones.

2. **Parseable peak tags.** ``_Peaks._parse_peak_tag`` needs a digit run whose
   length is a multiple of three; ``peak0`` and ``peak1`` fail it. They are
   renamed mechanically -- ``peakN`` becomes ``peak`` + the digit repeated three
   times -- giving phase name ``peak`` with Miller indices (0,0,0) and (1,1,1).
   **These indices carry no physical claim**, and nothing is lost by inventing
   them: the source file records ``d reference`` as 1.0 for one peak and NaN for
   the other, so these tags never identified a reflection in the first place.

The ``mask`` group is dropped. It is 8.4 MB of the source file's 11 MB, which
would breach pre-commit's 8 MB ceiling, and NXstress generates a default mask
when a workspace carries none -- so keeping it would cost the repository 8 MB to
exercise nothing this fixture is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "tests/data/HB2B_1327.h5"
GEOMETRY_DONOR = REPO / "tests/data/HB2B_938_peak.h5"
TARGET = REPO / "tests/data/HB2B_1327_with_instrument.h5"

#: pre-commit's `check-added-large-files` ceiling. Named so the refusal message
#: cannot drift from the comparison -- they were two separate literals.
SIZE_CEILING_MB = 8.0

GEOMETRY_PATH = "instrument/geometry setup/detector"
SKIP_TOP_LEVEL = {"mask"}
PEAK_TAG_RENAMES = {"peak0": "peak000", "peak1": "peak111"}


def main() -> int:
    try:
        return _build()
    finally:
        # A failure anywhere between creating the staging file and moving it must
        # not leave it in `tests/data`.
        staging = TARGET.with_name(TARGET.name + ".staging")
        staging.unlink(missing_ok=True)


def _build() -> int:
    for path in (SOURCE, GEOMETRY_DONOR):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    with h5py.File(GEOMETRY_DONOR, "r") as donor:
        geometry = {key: donor[f"{GEOMETRY_PATH}/{key}"][()] for key in donor[GEOMETRY_PATH]}
    if not geometry:
        print(f"{GEOMETRY_DONOR} has no geometry to copy", file=sys.stderr)
        return 1
    print(f"geometry from {GEOMETRY_DONOR.name}: { {k: v.tolist() for k, v in geometry.items()} }")

    # Built beside the target, then moved into place only once the size is
    # acceptable. Writing `TARGET` directly would truncate the committed fixture
    # before anything had been checked, so a refusal would leave the repository
    # with a destroyed -- or, if the script then removed it, a deleted -- tracked
    # file. Nothing here deletes a file it did not create.
    staged = TARGET.with_name(TARGET.name + ".staging")
    with h5py.File(SOURCE, "r") as src, h5py.File(staged, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        for name in src:
            if name in SKIP_TOP_LEVEL:
                print(f"skipping {name!r} (see module docstring)")
                continue
            if name == "peaks":
                peaks = dst.create_group("peaks")
                for key, value in src["peaks"].attrs.items():
                    peaks.attrs[key] = value
                for tag in src["peaks"]:
                    renamed = PEAK_TAG_RENAMES.get(tag, tag)
                    src.copy(src[f"peaks/{tag}"], peaks, name=renamed)
                    if renamed != tag:
                        print(f"renamed peak tag {tag!r} -> {renamed!r}")
                continue
            src.copy(src[name], dst, name=name)

        detector = dst.require_group(GEOMETRY_PATH)
        for key, value in geometry.items():
            if key in detector:
                del detector[key]
            detector.create_dataset(key, data=value)
        print(f"populated {GEOMETRY_PATH} with {sorted(geometry)}")

    size_mb = staged.stat().st_size / 1e6
    if size_mb > SIZE_CEILING_MB:
        # Refused before `TARGET` is touched, so an existing fixture is left exactly
        # as it was. Only the staging file -- which this script created -- is removed.
        staged.unlink()
        print(
            f"refusing to install {TARGET.name}: {size_mb:.2f} MB exceeds pre-commit's "
            f"{SIZE_CEILING_MB:.0f} MB ceiling.\n  {TARGET} is unchanged.",
            file=sys.stderr,
        )
        return 1

    staged.replace(TARGET)
    print(f"wrote {TARGET} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
