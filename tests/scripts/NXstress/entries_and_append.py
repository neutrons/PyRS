"""Several entries in one file, and appending to an entry.

Shows that:

1. a file can hold several `NXentry` groups; `entry_numbers()` lists them, and
   `read`/`write` select one by its 1-based `entry_number`;
2. an entry number that skips past the next free entry, or is not positive, is
   refused;
3. an append grows an existing entry, and the grown entry reads back with the
   new workspace;
4. a refused append leaves the file byte-for-byte unchanged and makes the
   `NXstress` instance unusable;
5. a single-workspace viewer refuses a two-entry file rather than showing one
   entry. The file is left in the output directory for the GUI steps in
   `plans/PRs/phase-2.md`.

Run from the repository root:

    pixi run python tests/scripts/NXstress/entries_and_append.py [output-dir]
"""

from _fixtures import check, nxstress_config, output_dir, run, section, sha256, synthetic_peaks, synthetic_workspace

from pyrs.interface.nxstress_load import UnsupportedNXstressFile, read_single_workspace
from pyrs.utilities.NXstress.NXstress import NXstress


def raises(exception: type[BaseException], action) -> str | None:
    """The message `action()` raised `exception` with, or None if it did not raise."""
    try:
        action()
    except exception as error:
        return str(error).splitlines()[0]
    return None


def sub_runs(workspaces) -> list[list[int]]:
    return [ws.get_sub_runs().raw_copy().tolist() for ws in workspaces]


def entries(out) -> None:
    section("several entries in one file")
    path = out / "two_entries.nxs"
    first = synthetic_workspace([1, 2, 3])
    second = synthetic_workspace([1, 2, 3, 4])
    with NXstress(path, "w") as nx:
        nx.write([first], [[synthetic_peaks(first)]])
        nx.write([second], [[synthetic_peaks(second, "Fe 211")]], entry_number=2)

    with NXstress(path, "r") as nx:
        numbers = nx.entry_numbers()
        print(f"  entry_numbers() = {numbers}")
        check("the file holds entries 1 and 2", numbers == (1, 2))
        workspaces, peakss = nx.read(entry_number=2)
        print(f"  read(entry_number=2): sub-runs {sub_runs(workspaces)}, peaks {[p.peak_tag for p in peakss[0]]}")
        check("entry 2 holds the second write", sub_runs(workspaces) == [[1, 2, 3, 4]])

        message = raises(ValueError, lambda: nx.read(entry_number=0))
        print(f"  read(entry_number=0) -> ValueError: {message}")
        check("entry number 0 is refused", message is not None)

    with NXstress(path, "a") as nx:
        message = raises(ValueError, lambda: nx.write([first], [[synthetic_peaks(first)]], entry_number=4))
        print(f"  write(entry_number=4) -> ValueError: {message}")
        check("an entry number past the next free one is refused", message is not None)
    with NXstress(path, "r") as nx:
        check("the file still holds entries 1 and 2", nx.entry_numbers() == (1, 2))

    section("a single-workspace viewer refuses a two-entry file")
    message = raises(UnsupportedNXstressFile, lambda: read_single_workspace(path))
    print(f"  read_single_workspace -> UnsupportedNXstressFile: {message}")
    check("the viewer loader refuses the file", message is not None)
    print(f"  Left for the GUI steps: {path}")


def append(out) -> None:
    section("appending a workspace to an entry")
    path = out / "append.nxs"
    with nxstress_config(discriminator_fields=["direction"]):
        ws11 = synthetic_workspace([1, 2, 3], direction="11")
        with NXstress(path, "w") as nx:
            nx.write([ws11], [[synthetic_peaks(ws11)]])

        ws22 = synthetic_workspace([4, 5, 6], direction="22")
        with NXstress(path, "a") as nx:
            nx.write([ws22], [[synthetic_peaks(ws22)]])

        with NXstress(path, "r") as nx:
            workspaces, _ = nx.read()
        directions = [ws.direction for ws in workspaces]
        print(f"  after the append: directions {directions}, sub-runs {sub_runs(workspaces)}")
        check("entry 1 now reads back as two workspaces", sub_runs(workspaces) == [[1, 2, 3], [4, 5, 6]])

        section("a refused append changes nothing")
        before = sha256(path)
        ws33 = synthetic_workspace([5, 6], direction="33")
        with NXstress(path, "a") as nx:
            message = raises(RuntimeError, lambda: nx.write([ws33], [[synthetic_peaks(ws33)]]))
            print(f"  scan points 5, 6 again -> RuntimeError: {message}")
            check("the colliding append is refused", message is not None)

            ws44 = synthetic_workspace([7, 8], direction="44")
            message = raises(RuntimeError, lambda: nx.write([ws44], [[synthetic_peaks(ws44)]]))
            print(f"  a valid append on the same instance -> RuntimeError: {message}")
            check("the instance is unusable after the refusal", message is not None)
        check("the file is byte-for-byte unchanged", sha256(path) == before)


def main() -> None:
    out = output_dir()
    entries(out)
    append(out)


if __name__ == "__main__":
    run(main)
