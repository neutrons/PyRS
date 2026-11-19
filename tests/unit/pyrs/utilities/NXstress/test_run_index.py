"""
The run index: every scan point of an entry is identified by `(run_number, scan_point)`.

Every HB2B run numbers its scan points from 1, so two runs in one entry collide on
scan point alone -- and the strain/stress workflow writes three directions, each
its own run, into one entry. Before the run index, NXstress refused any scan point
shared by two workspaces of an entry, whatever their discriminator values, so that
workflow could not be written at all. Measured on the real files: HB2B_1320,
1327, 1328, 1331 and 1332 each number their scan points 1..N.

What is pinned here:

- runs that collide on scan point round-trip, and two runs sharing every
  discriminator value come back as two workspaces;
- a `(run, scan point)` pair repeated across workspaces is refused -- including
  two workspaces with no `run_number` log, which share the configured default;
- a workspace merged from several runs is written alone, and only alone;
- an entry written before the run index still reads exactly as it did;
- a single workspace with no single value for a configured discriminator records
  no boundary, and reads back as one workspace under any configuration;
- a peak tag naming no lattice plane is kept, with hkl `(000)` meaning
  "unknown", and warned about both ways.

Unit tier: synthetic workspaces through `NXstress`'s own write and read, in
`tmp_path`.
"""

from collections.abc import Callable
import logging

import h5py
import numpy as np
import pytest
import yaml
from nexusformat.nexus import nxload

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress import _append, _discriminator, _runs
from pyrs.utilities.NXstress.NXstress import NXstress


def configure(config_override, yaml_text: str) -> None:
    config_override(yaml.safe_load(yaml_text))


def workspace(
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
    *,
    sub_runs: tuple,
    direction: str | None = None,
    run: int | list[int] | None = None,
    peak_tag: str = "Fe110",
) -> tuple[HidraWorkspace, list[PeakCollection]]:
    """One workspace, optionally carrying `direction` and `run_number` logs."""
    points = np.array(sub_runs)
    ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
    if direction is not None:
        ws.set_sample_log("direction", points, np.array([direction] * len(points)))
    if run is not None:
        runs = np.asarray(run) if isinstance(run, list) else np.full(len(points), run)
        ws.set_sample_log("run_number", points, runs)
    collection = minimal_PeakCollection(N_subrun=len(points), peak_tag=peak_tag, sub_runs=points)
    return ws, [collection]


def write(path, inputs) -> None:
    with NXstress(path, "w") as nx:
        nx.write([ws for ws, _ in inputs], [peaks for _, peaks in inputs])


def read(path):
    with NXstress(path, "r") as nx:
        return nx.read()


@pytest.fixture
def by_direction(config_override):
    configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")


class TestRunsCollidingOnScanPoint:
    def test_three_directions_from_three_runs_round_trip(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The strain/stress case: every run scans points 1..3."""
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3), direction=d, run=r)
            for d, r in (("11", 1327), ("22", 1331), ("33", 1332))
        ]
        path = tmp_path / "three.nxs"

        # Act
        write(path, inputs)
        workspaces, peakss = read(path)

        # Assert
        assert [ws.direction for ws in workspaces] == ["11", "22", "33"]
        assert [ws.get_sub_runs().raw_copy().tolist() for ws in workspaces] == [[1, 2, 3]] * 3
        assert [int(ws.get_sample_log_value("run_number")) for ws in workspaces] == [1327, 1331, 1332]
        assert [[int(pc.runnumber) for pc in peaks] for peaks in peakss] == [[1327], [1331], [1332]]

    def test_two_runs_in_one_direction_come_back_as_two_workspaces(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Every discriminator value shared: only the run tells them apart."""
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3), direction="11", run=r)
            for r in (1327, 1328)
        ]
        path = tmp_path / "two_runs.nxs"

        # Act
        write(path, inputs)
        workspaces, peakss = read(path)

        # Assert
        assert len(workspaces) == 2
        assert [ws.direction for ws in workspaces] == ["11", "11"]
        assert [int(peaks[0].runnumber) for peaks in peakss] == [1327, 1328]

    def test_the_run_index_is_written_beside_each_scan_point_axis(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction=d, run=r)
            for d, r in (("11", 1327), ("22", 1331))
        ]
        path = tmp_path / "index.nxs"

        # Act
        write(path, inputs)

        # Assert
        with h5py.File(path, "r") as f:
            sd, peaks = f["entry/SAMPLE_DESCRIPTION"], f["entry/peaks"]
            assert sd["run_number"][()].tolist() == [1327, 1327, 1331, 1331]
            assert sd["scan_point"][()].tolist() == [1, 2, 1, 2]
            assert peaks["run_number"][()].tolist() == [1327, 1327, 1331, 1331]
            assert bool(peaks.attrs[_runs.BOUNDARY_ATTRIBUTE]) is True


class TestOrderAndInterleaving:
    """The read gives workspaces back in written order, and each direction is one block on disk."""

    def test_workspaces_come_back_in_the_order_written(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction=d, run=r)
            for d, r in (("22", 1331), ("11", 1327), ("33", 1332))
        ]
        path = tmp_path / "order.nxs"

        # Act
        write(path, inputs)
        workspaces, peakss = read(path)

        # Assert -- by direction, then run: the most slowly varying parts of the key
        assert [(ws.direction, int(peaks[0].runnumber)) for ws, peaks in zip(workspaces, peakss)] == [
            ("11", 1327),
            ("22", 1331),
            ("33", 1332),
        ]

    def test_interleaved_runs_of_one_direction_stay_one_block(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Direction 11 from two runs, written either side of direction 22."""
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction=d, run=r)
            for d, r in (("11", 1327), ("22", 1331), ("11", 1328))
        ]
        path = tmp_path / "interleaved.nxs"

        # Act
        write(path, inputs)
        workspaces, peakss = read(path)

        # Assert
        with h5py.File(path, "r") as f:
            directions = [value.decode() for value in f["entry/peaks/direction"][()]]
        assert [d for n, d in enumerate(directions) if n == 0 or d != directions[n - 1]] == ["11", "22"]
        assert sorted((ws.direction, int(p[0].runnumber)) for ws, p in zip(workspaces, peakss)) == [
            ("11", 1327),
            ("11", 1328),
            ("22", 1331),
        ]


class TestRepeatedRunAndScanPointIsRefused:
    def test_the_same_run_and_scan_point_twice_is_refused(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction=d, run=1327)
            for d in ("11", "22")
        ]

        # Act / Assert
        with pytest.raises(ValueError, match=r"\(1327, 1\)"):
            write(tmp_path / "repeat.nxs", inputs)

    def test_two_workspaces_without_a_run_number_share_the_default(
        self, by_direction, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """No `run_number` log means the configured default -- the same run for both."""
        # Arrange
        configure(config_override, "nxstress:\n  default_run_number: -5\n")
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction=d)
            for d in ("11", "22")
        ]

        # Act / Assert
        with pytest.raises(ValueError, match=r"\(-5, 1\)"):
            write(tmp_path / "defaults.nxs", inputs)


class TestRunNumbersMustBeWhole:
    def test_a_fractional_run_number_is_refused(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Truncating 1327.5 to 1327 would be silent, and could make two runs collide."""
        # Arrange
        inputs = [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="11")]
        ws, _ = inputs[0]
        ws.set_sample_log("run_number", np.array([1, 2]), np.array([1327.5, 1327.5]))

        # Act / Assert
        with pytest.raises(ValueError, match=r"1327\.5"):
            write(tmp_path / "fractional.nxs", inputs)

    def test_a_fractional_run_number_refuses_an_append(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Reported as every append refusal is: `InvariantViolated`, the instance invalidated, nothing written."""
        # Arrange
        path = tmp_path / "fractional_append.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="11")])
        before = path.read_bytes()
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="22")
        ws.set_sample_log("run_number", np.array([1, 2]), np.array([1331.5, 1331.5]))

        # Act / Assert
        with NXstress(path, "a") as nx:
            with pytest.raises(_append.InvariantViolated, match=r"cannot append -- .*1331\.5"):
                nx.write([ws], [peaks])
            with pytest.raises(RuntimeError, match=r"no longer usable"):
                nx.write([ws], [peaks])
        assert path.read_bytes() == before


class TestAWorkspaceMergedFromSeveralRuns:
    """CombineRuns' output: one workspace, renumbered scan points, a varying `run_number` log."""

    def test_alone_it_round_trips_as_one_workspace(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        merged = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3, 4), run=[1327, 1327, 1328, 1328]
        )
        path = tmp_path / "merged.nxs"

        # Act
        write(path, [merged])
        (ws,), (peaks,) = read(path)

        # Assert
        assert ws.get_sub_runs().raw_copy().tolist() == [1, 2, 3, 4]
        assert ws.get_sample_log_values("run_number").tolist() == [1327, 1327, 1328, 1328]
        assert peaks[0].runnumber == _runs.default_run_number(), "a collection spanning two runs has no one run"
        with h5py.File(path, "r") as f:
            assert bool(f["entry/peaks"].attrs[_runs.BOUNDARY_ATTRIBUTE]) is False

    def test_beside_another_workspace_it_is_refused(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        merged = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="11", run=[1327, 1328]
        )
        other = workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="22", run=1331)

        # Act / Assert
        with pytest.raises(ValueError, match=r"hold several runs"):
            write(tmp_path / "mixed.nxs", [merged, other])


class TestEntriesWrittenBeforeTheRunIndex:
    """Older entries carry no run column and no boundary record, and read exactly as before."""

    @staticmethod
    def strip_run_index(path) -> None:
        with h5py.File(path, "r+") as f:
            del f["entry/peaks/run_number"]
            del f["entry/SAMPLE_DESCRIPTION/run_number"]
            del f["entry/peaks"].attrs[_runs.BOUNDARY_ATTRIBUTE]

    def test_a_single_workspace_entry_reads_unchanged(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        path = tmp_path / "old.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3), run=1017)])
        self.strip_run_index(path)

        # Act
        (ws,), (peaks,) = read(path)

        # Assert
        assert ws.get_sub_runs().raw_copy().tolist() == [1, 2, 3]
        assert peaks[0].runnumber == _runs.default_run_number()

    def test_a_multi_workspace_entry_still_splits_by_scan_point(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """An entry from before the run index: disjoint scan points, told apart by discriminator alone."""
        # Arrange
        path = tmp_path / "old_multi.nxs"
        write(
            path,
            [
                workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3), direction="11"),
                workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(4, 5, 6), direction="22"),
            ],
        )
        self.strip_run_index(path)

        # Act
        workspaces, _ = read(path)

        # Assert
        assert [ws.direction for ws in workspaces] == ["11", "22"]
        assert [ws.get_sub_runs().raw_copy().tolist() for ws in workspaces] == [[1, 2, 3], [4, 5, 6]]


class TestASingleWorkspaceWithNoSingleValue:
    """A configured discriminator a single workspace has no value for records no boundary.

    `discriminator_fields: ["direction"]` is the shipped default, and a PeakFitting,
    Texture or CombineRuns workspace has no direction. One workspace has no
    boundary to record, so the entry records an empty field list -- which every
    read takes as one workspace, whatever is configured at the time.
    """

    @pytest.mark.parametrize("read_with", [[], ["direction"], ["zone"]], ids=["none", "direction", "other"])
    def test_it_reads_back_as_one_workspace_under_any_configuration(
        self, by_direction, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path, read_with
    ):
        # Arrange
        path = tmp_path / "no_direction.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2, 3), run=1017)])

        # Act
        config_override({"nxstress": {"discriminator_fields": read_with}})
        workspaces, _ = read(path)

        # Assert
        assert len(workspaces) == 1
        with nxload(str(path), "r") as root:
            assert _discriminator.recorded_names(root["entry/peaks"]) == ()
            assert "direction" not in root["entry/peaks"]

    def test_several_workspaces_missing_value_refused(
        self, by_direction, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """There the discriminator is the only boundary marker."""
        # Arrange
        inputs = [
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), run=1327),
            workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), direction="22", run=1331),
        ]

        # Act / Assert
        with pytest.raises(ValueError, match=r"'direction'"):
            write(tmp_path / "missing.nxs", inputs)


class TestAPeakTagNamingNoLatticePlane:
    """Interim: hkl `(000)` means "unknown", and the whole tag is kept as the phase name.

    Real strain/stress files tag peaks `peak0`, which names no phase and no plane,
    and `peak000` reads as `(000)`, which is no family of planes. Both are written
    with `(0, 0, 0)` and read back as the tag they were. PyRS should always fill
    in a valid `<phase><hkl>` tag, so every occurrence is logged as a warning.
    """

    @pytest.mark.parametrize("tag", ["peak0", "peak000", "Si111"])
    def test_the_tag_round_trips_exactly(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path, tag):
        # Arrange
        path = tmp_path / "tag.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), peak_tag=tag)])

        # Act
        _, (peaks,) = read(path)

        # Assert
        assert [pc.peak_tag for pc in peaks] == [tag]

    @pytest.mark.parametrize("tag", ["peak0", "peak000"])
    def test_unknown_plane_written_as_000_and_warned(
        self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path, caplog, tag
    ):
        # Arrange
        path = tmp_path / "unknown.nxs"
        inputs = [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), peak_tag=tag)]

        # Act
        with caplog.at_level(logging.WARNING, logger="pyrs.utilities.NXstress._peaks"):
            write(path, inputs)
            written = [r.getMessage() for r in caplog.records]
            caplog.clear()
            read(path)
            on_read = [r.getMessage() for r in caplog.records]

        # Assert
        with h5py.File(path, "r") as f:
            peaks = f["entry/peaks"]
            assert set(zip(peaks["h"][()].tolist(), peaks["k"][()].tolist(), peaks["l"][()].tolist())) == {(0, 0, 0)}
            assert {p.decode() for p in peaks["phase_name"][()]} == {tag}
        assert any(tag in message and "names no lattice plane" in message for message in written)
        assert any(tag in message and "(0, 0, 0)" in message for message in on_read)

    def test_a_real_plane_is_not_warned_about(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path, caplog):
        # Arrange
        path = tmp_path / "real.nxs"

        # Act
        with caplog.at_level(logging.WARNING, logger="pyrs.utilities.NXstress._peaks"):
            write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), peak_tag="Si111")])
            read(path)

        # Assert
        assert not [r for r in caplog.records if "lattice plane" in r.getMessage() or "(0, 0, 0)" in r.getMessage()]


class TestOneScanPoint:
    """An entry holding a single scan point reads its axes back as arrays.

    nexusformat collapses a length-1 field to a bare scalar in `nxvalue`; the run
    index reads `nxdata`, which keeps the shape. Pinned for the single-workspace
    read, the multi-workspace split and the append pre-flight, which each measure
    the scan-point axis.
    """

    def test_single_workspace_round_trips(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        path = tmp_path / "one_point.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1,), run=1327)])

        # Act
        workspaces, peakss = read(path)

        # Assert
        assert workspaces[0].get_sub_runs().raw_copy().tolist() == [1]
        assert peakss[0][0].sub_runs.raw_copy().tolist() == [1]

    def test_append_to_a_one_point_entry(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        path = tmp_path / "one_point_append.nxs"
        write(
            path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1,), direction="11", run=1327)]
        )
        incoming = workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1,), direction="22", run=1331)

        # Act
        with NXstress(path, "a") as nx:
            nx.write([incoming[0]], [incoming[1]])
        workspaces, _ = read(path)

        # Assert
        assert [(ws.direction, ws.get_sub_runs().raw_copy().tolist()) for ws in workspaces] == [
            ("11", [1]),
            ("22", [1]),
        ]


class TestTheRunIsStoredOnce:
    """The run index's column is the one record of a scan point's run, as `vx` is of its position.

    It is not repeated in `SAMPLE_DESCRIPTION/logs`, and the `run_number` log is
    restored from it on read.
    """

    def test_no_copy_in_the_logs(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        path = tmp_path / "once.nxs"

        # Act
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), run=1327)])

        # Assert
        with h5py.File(path, "r") as f:
            assert f["entry/SAMPLE_DESCRIPTION/run_number"][()].tolist() == [1327, 1327]
            assert "run_number" not in f["entry/SAMPLE_DESCRIPTION/logs"]

    def test_the_log_is_restored_from_the_column(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        path = tmp_path / "restored.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), run=1327)])

        # Act
        (ws,), _ = read(path)

        # Assert
        assert ws.get_sample_log_values("run_number").tolist() == [1327, 1327]

    def test_no_log_reads_back_as_no_log(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        """The column holds the configured default for it, which means "no run number"."""
        # Arrange
        path = tmp_path / "no_log.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2))])

        # Act
        (ws,), _ = read(path)

        # Assert
        assert "run_number" not in ws.sample_log_names

    def test_a_file_before_the_run_index_keeps_its_logged_run(
        self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Phase 2 wrote the log into `logs` and had no column; it is read from there, as before."""
        # Arrange
        path = tmp_path / "phase_2.nxs"
        write(path, [workspace(minimal_HidraWorkspace, minimal_PeakCollection, sub_runs=(1, 2), run=1017)])
        TestEntriesWrittenBeforeTheRunIndex.strip_run_index(path)
        with h5py.File(path, "r+") as f:
            logged = f["entry/SAMPLE_DESCRIPTION/logs"].create_dataset("run_number", data=np.array([1017, 1017]))
            logged.attrs["local_name"] = "run_number"

        # Act
        (ws,), _ = read(path)

        # Assert
        assert ws.get_sample_log_values("run_number").tolist() == [1017, 1017]
