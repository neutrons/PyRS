"""Appending under the run index.

Shows that:

1. a new run in a direction the entry already holds is a Case A append: it joins
   the entry as a workspace of its own, though its scan points repeat the first
   run's;
2. each of these is refused before anything is written, leaving the file
   byte-for-byte unchanged:
   - a `(run, scan point)` the entry already holds;
   - an incoming workspace merged from several runs, which has no one run to be
     keyed by;
   - an entry written before the run index, which has no `run_number` column for
     the incoming rows;
3. more scan points for a `(direction, run)` the entry already holds is Case B,
   refused as not supported.

Run from the repository root:

    pixi run python tests/scripts/cis_tests/NXstress/run_index_append.py [output-dir]
"""

import shutil

from _fixtures import check, output_dir, run, section, sha256, synthetic_peaks, synthetic_workspace

import h5py
import numpy as np

from pyrs.utilities.NXstress.NXstress import NXstress


def workspace(points: list[int], *, direction: str, runs: list[int]):
    """A synthetic workspace over `points`, with a `direction` and a per-point `run_number` log."""
    ws = synthetic_workspace(points, direction=direction)
    ws.set_sample_log("run_number", np.asarray(points), np.asarray(runs))
    return ws


def summary(path) -> list[tuple]:
    """(direction, run, scan points) per workspace read back."""
    with NXstress(path, "r") as nx:
        workspaces, _ = nx.read()
    return [
        (ws.direction, int(ws.get_sample_log_values("run_number")[0]), ws.get_sub_runs().raw_copy().tolist())
        for ws in workspaces
    ]


def refused(path, ws, exception: type[BaseException]) -> str | None:
    """Append `ws` on a fresh instance; the first line of the refusal, or None if it was accepted."""
    with NXstress(path, "a") as nx:
        try:
            nx.write([ws], [[synthetic_peaks(ws)]])
        except exception as error:
            return str(error).splitlines()[0]
    return None


def main() -> None:
    out = output_dir()
    path = out / "append_runs.nxs"

    section("Case A: a second run in direction 11")
    first = workspace([1, 2, 3], direction="11", runs=[1327] * 3)
    with NXstress(path, "w") as nx:
        nx.write([first], [[synthetic_peaks(first)]])
    second = workspace([1, 2, 3], direction="11", runs=[1328] * 3)
    with NXstress(path, "a") as nx:
        nx.write([second], [[synthetic_peaks(second)]])
    result = summary(path)
    print(f"  read back (direction, run, scan points): {result}")
    check("two workspaces in direction 11, one per run", result == [("11", 1327, [1, 2, 3]), ("11", 1328, [1, 2, 3])])

    section("refused, the file unchanged")
    before = sha256(path)
    collision = workspace([3, 4], direction="22", runs=[1328] * 2)
    message = refused(path, collision, RuntimeError)
    print(f"  run 1328, scan point 3 again -> RuntimeError: {message}")
    check("a (run, scan point) already in the entry is refused", message is not None)
    check("the file is byte-for-byte unchanged", sha256(path) == before)

    merged = workspace([4, 5], direction="22", runs=[1331, 1332])
    message = refused(path, merged, RuntimeError)
    print(f"  a workspace of runs 1331 and 1332 -> RuntimeError: {message}")
    check("an incoming workspace merged from several runs is refused", message is not None)
    check("the file is byte-for-byte unchanged", sha256(path) == before)

    old = out / "append_before_run_index.nxs"
    shutil.copyfile(path, old)
    # What an earlier PyRS wrote: no run column anywhere, and no boundary record.
    with h5py.File(old, "a") as f:
        del f["entry/peaks/run_number"], f["entry/SAMPLE_DESCRIPTION/run_number"]
        del f["entry/peaks"].attrs["run_number_is_boundary"]
    before_old = sha256(old)
    message = refused(old, workspace([7, 8], direction="33", runs=[1332] * 2), RuntimeError)
    print(f"  onto an entry from before the run index -> RuntimeError: {message}")
    check("an entry written before the run index is refused", message is not None)
    check("that file is byte-for-byte unchanged", sha256(old) == before_old)

    section("Case B: more scan points for a (direction, run) already there")
    message = refused(path, workspace([4, 5], direction="11", runs=[1328] * 2), NotImplementedError)
    print(f"  direction 11, run 1328, scan points 4, 5 -> NotImplementedError: {message}")
    check("Case B is refused as not supported", message is not None)
    check("the file is byte-for-byte unchanged", sha256(path) == before)


if __name__ == "__main__":
    run(main)
