"""Several workspaces in one entry, told apart by discriminator fields.

Shows that:

1. with `nxstress.discriminator_fields: [direction]`, two workspaces written to
   one entry read back as two, each with its own direction;
2. with no discriminator field, a multi-workspace write is refused: NXstress
   does not merge workspaces. A workspace merged in PyRS (as CombineRuns does)
   is written as one workspace -- under a discriminator it has no single value
   for too, recording no boundary, since one workspace has none to record;
3. a Case A append (a new direction) works; a Case B append (more scan points
   for a direction the entry already holds) is refused;
4. a read whose configuration disagrees with the fields the entry records is
   refused, and the message names the fields to configure;
5. two workspaces sharing a workspace key -- every discriminator value, and the
   run -- are refused at write time: they would read back as one workspace, and
   NXstress never merges workspaces.

Run from the repository root:

    pixi run python tests/scripts/cis_tests/NXstress/multi_workspace.py [output-dir]
"""

from _fixtures import check, nxstress_config, output_dir, run, section, sha256, synthetic_peaks, synthetic_workspace

import numpy as np

from pyrs.utilities.NXstress.NXstress import NXstress


def raises(exception: type[BaseException], action) -> str | None:
    """The first line of the message `action()` raised `exception` with, or None if it did not raise."""
    try:
        action()
    except exception as error:
        return str(error).splitlines()[0]
    return None


def read_back(path) -> list:
    with NXstress(path, "r") as nx:
        workspaces, _ = nx.read()
    return workspaces


def summary(path) -> list[tuple[str, list[int]]]:
    """(direction, sub-runs) per workspace read back."""
    return [(ws.direction, ws.get_sub_runs().raw_copy().tolist()) for ws in read_back(path)]


def two_directions() -> tuple[list, list]:
    """Directions 11 and 22 over scan points [1, 2, 3] and [4, 5, 6]."""
    ws11 = synthetic_workspace([1, 2, 3], direction="11")
    ws22 = synthetic_workspace([4, 5, 6], direction="22")
    return [ws11, ws22], [[synthetic_peaks(ws11)], [synthetic_peaks(ws22)]]


def main() -> None:
    out = output_dir()
    path = out / "two_directions.nxs"

    section("discriminator_fields: [direction]")
    with nxstress_config(discriminator_fields=["direction"]):
        with NXstress(path, "w") as nx:
            nx.write(*two_directions())
        result = summary(path)
        print(f"  read back (direction, sub-runs): {result}")
        check("two workspaces come back, one per direction", result == [("11", [1, 2, 3]), ("22", [4, 5, 6])])

    section("no discriminator field")
    refused = out / "refused.nxs"
    with nxstress_config(discriminator_fields=[]), NXstress(refused, "w") as nx:
        message = raises(ValueError, lambda: nx.write(*two_directions()))
    print(f"  two workspaces -> ValueError: {message}")
    check("a multi-workspace write is refused", message is not None)

    section("a workspace merged in PyRS")
    merged_ws = synthetic_workspace([1, 2, 3, 4, 5, 6])
    merged_ws.set_sample_log("direction", np.arange(1, 7), np.array(["11"] * 3 + ["22"] * 3))
    merged = out / "merged.nxs"
    with nxstress_config(discriminator_fields=[]):
        with NXstress(merged, "w") as nx:
            nx.write([merged_ws], [[synthetic_peaks(merged_ws)]])
        (back,) = read_back(merged)
    points = back.get_sub_runs().raw_copy().tolist()
    directions = back.get_sample_log_values("direction").tolist()
    print(f"  no discriminator -> read back sub-runs {points}, direction log {directions}")
    check("it round-trips as one workspace with its log intact", directions == ["11"] * 3 + ["22"] * 3)

    message = raises(ValueError, lambda: back.direction)
    print(f"  ws.direction -> ValueError: {message}")
    check("it has no single direction", message is not None)

    with nxstress_config(discriminator_fields=["direction"]):
        merged_direction = out / "merged_under_direction.nxs"
        with NXstress(merged_direction, "w") as nx:
            nx.write([merged_ws], [[synthetic_peaks(merged_ws)]])
        workspaces = read_back(merged_direction)
    print(f"  discriminator_fields: [direction] -> read back as {len(workspaces)} workspace(s)")
    check("under a discriminator it has no single value for, it is still one workspace", len(workspaces) == 1)

    with nxstress_config(discriminator_fields=["direction"]):
        section("Case A append: a new direction")
        ws33 = synthetic_workspace([7, 8], direction="33")
        with NXstress(path, "a") as nx:
            nx.write([ws33], [[synthetic_peaks(ws33)]])
        result = summary(path)
        print(f"  read back (direction, sub-runs): {result}")
        check("the entry now holds three workspaces", [d for d, _ in result] == ["11", "22", "33"])

        section("Case B append: more scan points for a direction already there")
        before = sha256(path)
        more11 = synthetic_workspace([9, 10], direction="11")
        with NXstress(path, "a") as nx:
            message = raises(NotImplementedError, lambda: nx.write([more11], [[synthetic_peaks(more11)]]))
        print(f"  -> NotImplementedError: {message}")
        check("Case B is refused", message is not None)
        check("the file is byte-for-byte unchanged", sha256(path) == before)

    section("reading under a different configuration")
    with nxstress_config(discriminator_fields=[]):
        message = raises(RuntimeError, lambda: summary(path))
    print(f"  discriminator_fields: [] -> RuntimeError: {message}")
    check("the read is refused", message is not None)
    check("the message names the fields to configure", "['direction']" in (message or ""))

    section("workspaces sharing a workspace key")
    with nxstress_config(discriminator_fields=["direction"]):
        later = synthetic_workspace([4, 5, 6], direction="11")
        earlier = synthetic_workspace([1, 2, 3], direction="11")
        with NXstress(out / "shared_key.nxs", "w") as nx:
            message = raises(
                ValueError,
                lambda: nx.write([later, earlier], [[synthetic_peaks(later)], [synthetic_peaks(earlier, "Fe 211")]]),
            )
    print(f"  direction 11 twice, same run -> ValueError: {message}")
    check("the write is refused: NXstress never merges workspaces", message is not None)
    check("the message names the shared key", message is not None and "same workspace key" in message)


if __name__ == "__main__":
    run(main)
