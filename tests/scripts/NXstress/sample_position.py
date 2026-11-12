"""Sample positions and transformation chains, written from a real HB2B dataset.

Shows that:

1. `peaks/sx,sy,sz` hold the `vx`/`vy`/`vz` sample coordinates, not the
   identically named stage logs, and each names its source log;
2. every link of the detector's and the sample's transformation chains is
   reachable from its `depends_on` entry point, each value resolved by the NeXus
   rule (relative to the group enclosing it).

Run from the repository root:

    pixi run python tests/scripts/NXstress/sample_position.py [output-dir]
"""

import h5py
import numpy as np
from nexusformat.nexus import nxload

from _fixtures import TEST_DATA, check, load_project, output_dir, run, section

from pyrs.utilities.NXstress.NXstress import NXstress
from pyrs.utilities.NXstress._peaks import _Peaks
from pyrs.utilities.NXstress._transformations import resolve_depends_on


def as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    source = TEST_DATA / "HB2B_1327_with_instrument.h5"
    path = output_dir() / "sample_position.nxs"

    ws, peaks = load_project(source)
    with NXstress(path, "w") as nx:
        nx.write([ws], [peaks])
    print(f"Wrote {path} from {source.name} ({len(peaks)} peak collections).")

    # The `peaks` rows are one per (peak collection, scan point), in the writer's sort order.
    collections = sorted(peaks, key=_Peaks.PeakIndex.sort_key)
    logs = ws._sample_logs

    def flattened(log_name: str) -> np.ndarray:
        return np.concatenate([np.asarray(logs[(log_name, c.sub_runs.raw_copy())], dtype=float) for c in collections])

    with h5py.File(path, "r") as f:
        section("peaks/sx,sy,sz are the sample coordinates")
        for axis, coordinate in (("sx", "vx"), ("sy", "vy"), ("sz", "vz")):
            written = f[f"entry/peaks/{axis}"][()]
            print(
                f"  peaks/{axis}[:3] = {np.round(written[:3], 3)}   {coordinate}[:3] = "
                f"{np.round(flattened(coordinate)[:3], 3)}   stage log {axis}[:3] = "
                f"{np.round(flattened(axis)[:3], 3)}"
            )
            check(f"peaks/{axis} has no NaN", not np.any(np.isnan(written)))
            check(f"peaks/{axis} equals the {coordinate} log", np.allclose(written, flattened(coordinate)))
            check(f"peaks/{axis} is not the {axis} stage log", not np.allclose(written, flattened(axis)))
            check(
                f"peaks/{axis} records local_name '{coordinate}'",
                as_text(f[f"entry/peaks/{axis}"].attrs["local_name"]) == coordinate,
            )

    # Resolved by the NeXus rule -- each value relative to the group enclosing it --
    # so a chain that does not end in "." raises rather than reporting a partial one.
    tree = nxload(path, "r")
    for label, group_path in (("detector", "entry/instrument/DETECTOR"), ("sample", "entry/SAMPLE_DESCRIPTION")):
        section(f"{label} transformation chain")
        group = tree[group_path]
        transformations = group["transformations"]
        reached = [link.nxname for link in resolve_depends_on(group)]
        print(f"  depends_on -> {' -> '.join(reached)} -> .")
        check(
            f"all {len(transformations)} {label} transformations are reachable",
            set(reached) == set(transformations),
        )


if __name__ == "__main__":
    run(main)
