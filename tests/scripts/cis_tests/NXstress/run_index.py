"""The run index: every scan point identified by `(run_number, scan_point)`.

Shows that:

1. four real HB2B runs -- e11 from 1327 and 1328, e22 from 1331, e33 from 1332 --
   each numbering its scan points from 1, are written to **one** entry and read
   back as four workspaces, each with its own run, direction and scan points;
2. on disk, `run_number` sits beside `scan_point` on `peaks` and
   `SAMPLE_DESCRIPTION`, and `peaks@run_number_is_boundary` records that the run
   is part of each workspace's key;
3. a workspace merged from several runs in PyRS, as CombineRuns makes, is
   written alone with the boundary recorded as False, and reads back as one;
4. an entry written before the run index -- no `run_number` column, no boundary
   record -- still reads, every scan point at the configured default run;
5. a peak tag naming no lattice plane (`peak0`, as every strain/stress file in
   `tests/data` uses) is written with hkl (000), meaning "unknown", and read back
   as exactly the tag it was. (The 1327 fixture's tags were renamed `peak000` and
   `peak111` when it was made; the second parses as phase `peak`, hkl (111).)

Run from the repository root:

    pixi run python tests/scripts/cis_tests/NXstress/run_index.py [output-dir]
"""

from _fixtures import TEST_DATA, check, load_project, output_dir, run, section, synthetic_peaks, synthetic_workspace

import h5py
import numpy as np

from pyrs.utilities.NXstress.NXstress import NXstress
from pyrs.utilities.restorable_property import restore

RUNS = {"11": (1327, 1328), "22": (1331,), "33": (1332,)}


def real_inputs() -> tuple[list, list]:
    """One workspace per run, given the direction it is loaded as, with its fitted peaks."""
    workspaces, peakss = [], []
    for direction, runs in RUNS.items():
        for run_number in runs:
            ws, peaks = load_project(TEST_DATA / f"HB2B_{run_number}_with_instrument.h5")
            restore(ws, "direction", direction)
            workspaces.append(ws)
            peakss.append(peaks)
    return workspaces, peakss


def describe(workspaces, peakss) -> list[tuple]:
    """(direction, run, first..last scan point, count) per workspace read back."""
    rows = []
    for ws, peaks in zip(workspaces, peakss):
        points = ws.get_sub_runs().raw_copy()
        rows.append((ws.direction, int(peaks[0].runnumber), int(points[0]), int(points[-1]), len(points)))
    return rows


def four_runs_in_one_entry(out) -> None:
    section("four real runs, each numbering its scan points from 1, in one entry")
    path = out / "four_runs.nxs"
    workspaces, peakss = real_inputs()
    for ws, peaks in zip(workspaces, peakss):
        points = ws.get_sub_runs().raw_copy()
        print(f"  input: run {int(peaks[0].runnumber)}, scan points {points[0]}..{points[-1]}")
    with NXstress(path, "w") as nx:
        nx.write(workspaces, peakss)

    with NXstress(path, "r") as nx:
        back, back_peaks = nx.read()
    rows = describe(back, back_peaks)
    for direction, run_number, first, last, count in rows:
        print(f"  read back: direction {direction}, run {run_number}, scan points {first}..{last} ({count})")
    expected = [(d, r) for d, runs in RUNS.items() for r in runs]
    check("four workspaces, one per run, under their directions", [(d, r) for d, r, *_ in rows] == expected)
    check("every run's scan points start at 1", all(first == 1 for _, _, first, _, _ in rows))
    check(
        "each workspace keeps its own scan points",
        [count for *_, count in rows] == [len(ws.get_sub_runs()) for ws in workspaces],
    )

    section("on disk")
    with h5py.File(path, "r") as f:
        peaks_group, sample = f["entry/peaks"], f["entry/SAMPLE_DESCRIPTION"]
        runs_on_disk = sorted(set(peaks_group["run_number"][()].tolist()))
        boundary = bool(peaks_group.attrs["run_number_is_boundary"])
        print(f"  peaks/run_number values: {runs_on_disk}")
        print(f"  SAMPLE_DESCRIPTION/run_number: {len(sample['run_number'])} rows, beside scan_point")
        print(f"  peaks@run_number_is_boundary = {boundary}")
        check("peaks carries every run", runs_on_disk == [1327, 1328, 1331, 1332])
        check("SAMPLE_DESCRIPTION has one run per scan point", len(sample["run_number"]) == len(sample["scan_point"]))
        check("the run is part of the workspace key", boundary)

        section("a peak tag naming no lattice plane")
        peak_rows = {
            (name.decode(), (int(h), int(k), int(l_)))
            for name, h, k, l_ in zip(peaks_group["phase_name"][()], peaks_group["h"], peaks_group["k"], peaks_group["l"])
        }
        for name, hkl in sorted(peak_rows):
            print(f"  on disk: phase_name {name!r}, (h, k, l) {hkl}")
        unnamed = {hkl for name, hkl in peak_rows if name in ("peak0", "peak1")}
        check("peak0 and peak1 are written whole as the phase name, with hkl (000)", unnamed == {(0, 0, 0)})
    tags_in = sorted({collection.peak_tag for peaks in peakss for collection in peaks})
    tags_out = sorted({collection.peak_tag for peaks in back_peaks for collection in peaks})
    print(f"  tags written {tags_in}; read back {tags_out}")
    check("every tag reads back exactly as it was", tags_in == tags_out)


def merged_runs(out) -> None:
    section("a workspace merged from several runs in PyRS")
    path = out / "merged_runs.nxs"
    ws = synthetic_workspace([1, 2, 3, 4])
    ws.set_sample_log("run_number", np.arange(1, 5), np.array([1327, 1327, 1328, 1328]))
    with NXstress(path, "w") as nx:
        nx.write([ws], [[synthetic_peaks(ws)]])
    with h5py.File(path, "r") as f:
        boundary = bool(f["entry/peaks"].attrs["run_number_is_boundary"])
    with NXstress(path, "r") as nx:
        back, _ = nx.read()
    runs = back[0].get_sample_log_values("run_number").tolist()
    print(f"  boundary recorded: {boundary}; read back as {len(back)} workspace, runs per point {runs}")
    check("the boundary is recorded as False", boundary is False)
    check("it reads back as one workspace, runs intact", len(back) == 1 and runs == [1327, 1327, 1328, 1328])


def before_the_run_index(out) -> None:
    section("an entry written before the run index")
    path = out / "before_run_index.nxs"
    ws11 = synthetic_workspace([1, 2, 3], direction="11")
    ws22 = synthetic_workspace([4, 5, 6], direction="22")
    with NXstress(path, "w") as nx:
        nx.write([ws11, ws22], [[synthetic_peaks(ws11)], [synthetic_peaks(ws22)]])
    # What an earlier PyRS wrote: no run column anywhere, and no boundary record.
    with h5py.File(path, "a") as f:
        del f["entry/peaks/run_number"], f["entry/SAMPLE_DESCRIPTION/run_number"]
        del f["entry/peaks"].attrs["run_number_is_boundary"]
    with NXstress(path, "r") as nx:
        back, _ = nx.read()
    summary = [(ws.direction, ws.get_sub_runs().raw_copy().tolist()) for ws in back]
    print(f"  read back (direction, scan points): {summary}")
    check("it still splits by direction alone", summary == [("11", [1, 2, 3]), ("22", [4, 5, 6])])


def main() -> None:
    out = output_dir()
    four_runs_in_one_entry(out)
    merged_runs(out)
    before_the_run_index(out)


if __name__ == "__main__":
    run(main)
