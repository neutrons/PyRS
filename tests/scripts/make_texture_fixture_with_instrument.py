#!/usr/bin/env python
"""Build ``tests/data/HB2B_1599_texture_with_instrument.h5``: a real texture project file with geometry.

Run by hand; not collected by pytest (``tests/scripts`` is in ``norecursedirs``):

    pixi run python tests/scripts/make_texture_fixture_with_instrument.py

An existing fixture is left alone unless ``--rebuild`` is given.

Why this fixture exists
-----------------------
TextureFitting offers *Save as NXstress...* only for a workspace with instrument
geometry, and no texture project file in ``tests/data`` (a file whose reduced data
is out-of-plane ``eta_<angle>`` patterns) has any: its
``instrument/geometry setup/detector`` group exists but is empty. So the viewer's
handling of eta fits at NXstress save could not be exercised on real data.

The fixture copies ``HB2B_1599_texture.h5`` and fills that group in from
``HB2B_938_peak.h5``, another real HB2B file in this repository, so the values are
real instrument parameters rather than invented ones. Everything else -- sample
logs, reduced diffraction data, fit results -- is copied verbatim, and the original
file is left in place.

The ``mask`` group is dropped: it is 8.4 MB of the source's 8.5 MB, which would
breach pre-commit's 8 MB ceiling, and NXstress generates a default mask when a
workspace carries none.

The same recipe builds the strain/stress ``*_with_instrument.h5`` fixtures in a
later PR (``tests/scripts/make_nxstress_sample_position_fixture.py``); when both
exist, this fixture should become one more entry of that script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tests/data"
SOURCE = DATA / "HB2B_1599_texture.h5"
TARGET = DATA / "HB2B_1599_texture_with_instrument.h5"
GEOMETRY_DONOR = DATA / "HB2B_938_peak.h5"

#: pre-commit's `check-added-large-files` ceiling (`--maxkb=8192`).
SIZE_CEILING_MB = 8.0

GEOMETRY_PATH = "instrument/geometry setup/detector"
SKIP_TOP_LEVEL = {"mask"}


def main(argv: list[str]) -> int:
    if TARGET.exists() and "--rebuild" not in argv:
        print(f"{TARGET.name} exists already; pass --rebuild to rebuild it")
        return 0
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

    # Built beside the target and moved into place only once its size is
    # acceptable, so a refusal never truncates or deletes a committed fixture.
    staged = TARGET.with_name(TARGET.name + ".staging")
    try:
        with h5py.File(SOURCE, "r") as src, h5py.File(staged, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for name in src:
                if name in SKIP_TOP_LEVEL:
                    print(f"skipping {name!r} (see module docstring)")
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
            print(
                f"refusing to install {TARGET.name}: {size_mb:.2f} MB exceeds pre-commit's "
                f"{SIZE_CEILING_MB:.0f} MB ceiling.\n  {TARGET} is unchanged.",
                file=sys.stderr,
            )
            return 1
        staged.replace(TARGET)
        print(f"wrote {TARGET} ({size_mb:.2f} MB)")
        return 0
    finally:
        staged.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
