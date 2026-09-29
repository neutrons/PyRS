"""Which `peaks` columns are discriminators: configuration decides, the file is checked.

Every rule here is stated in `plans/PRs/phase-2.md`, "Discriminator columns on
read and append". In short:

- A write records its discriminator fields in the `peaks` group's
  `discriminator_fields` attribute; an empty list records no workspace boundary.
- A read takes its discriminator fields from configuration. An entry that records
  its fields must agree; one that records none is read as one workspace.
- Without the record, configured fields must have a column each (none at all
  means one workspace), and any other column is ignored.
- An append refuses an entry with any column it would not grow.

The entry-without-record cases stand for a file from another NXstress producer,
or one PyRS wrote before the attribute existed; they are made here by deleting
the attribute from a PyRS-written file.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._discriminator import FIELDS_ATTRIBUTE
from pyrs.utilities.NXstress.NXstress import NXstress

PEAKS = "entry/peaks"


def configure(config_override, yaml_text: str) -> None:
    config_override(yaml.safe_load(yaml_text))


def write_two_directions(
    path: Path,
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
) -> None:
    """Directions 11 and 22 over scan points [1, 2, 3] and [4, 5, 6], one `Fe 211` peak each.

    The shape in which reading with the discriminator omitted merges the two
    workspaces without any error.
    """
    workspaces, peakss = [], []
    for direction, points in (("11", np.array([1, 2, 3])), ("22", np.array([4, 5, 6]))):
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
        ws.set_sample_log("direction", points, np.array([direction] * len(points)))
        workspaces.append(ws)
        peakss.append([minimal_PeakCollection(N_subrun=3, peak_tag="Fe 211", sub_runs=points)])
    with NXstress(path, "w") as nx:
        nx.write(workspaces, peakss)


def recorded(path: Path) -> list:
    with h5py.File(path, "r") as f:
        return [to_str(v) for v in f[PEAKS].attrs[FIELDS_ATTRIBUTE]]


def to_str(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def drop_record(path: Path) -> None:
    with h5py.File(path, "a") as f:
        del f[PEAKS].attrs[FIELDS_ATTRIBUTE]


def add_foreign_column(path: Path, name: str = "intensity") -> None:
    """A row-aligned reflection field PyRS does not know, as another producer might write."""
    with h5py.File(path, "a") as f:
        n = f[PEAKS]["scan_point"].shape[0]
        f[PEAKS].create_dataset(name, data=np.ones(n), maxshape=(None,))


def read(path: Path) -> list:
    with NXstress(path, "r") as nx:
        workspaces, _ = nx.read()
    return [ws.get_sub_runs().raw_copy().tolist() for ws in workspaces]


@pytest.fixture
def two_directions(config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path) -> Path:
    configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
    path = tmp_path / "two.nxs"
    write_two_directions(path, minimal_HidraWorkspace, minimal_PeakCollection)
    return path


class TestTheRecord:
    def test_a_write_records_its_discriminator_fields(self, two_directions):
        assert recorded(two_directions) == ["direction"]

    def test_a_write_with_no_field_records_no_boundary(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        path = tmp_path / "single.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([ws], [[minimal_PeakCollection(N_subrun=3)]])

        # Assert
        assert recorded(path) == []


class TestReadWithTheRecord:
    def test_the_silent_merge_shape_is_refused(self, two_directions, config_override):
        """With `direction` omitted, these two workspaces would read back as one, without error."""
        configure(config_override, "nxstress:\n  discriminator_fields: []\n")
        with pytest.raises(RuntimeError, match=r"Set 'nxstress.discriminator_fields' to \['direction'\]"):
            read(two_directions)

    def test_an_empty_record_reads_as_one_workspace_under_any_fields(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange -- written with nothing configured
        path = tmp_path / "single.nxs"
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        with NXstress(path, "w") as nx:
            nx.write([ws], [[minimal_PeakCollection(N_subrun=3)]])
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")

        # Act / Assert
        assert read(path) == [[1, 2, 3]]

    def test_a_recorded_field_with_no_column_raises(self, two_directions):
        with h5py.File(two_directions, "a") as f:
            del f[PEAKS]["direction"]
        with pytest.raises(RuntimeError, match=r"has no column for \['direction'\]; the file is damaged"):
            read(two_directions)

    def test_a_foreign_column_is_ignored(self, two_directions):
        add_foreign_column(two_directions)
        assert read(two_directions) == [[1, 2, 3], [4, 5, 6]]


class TestReadWithoutTheRecord:
    def test_configured_fields_with_columns_split_the_entry(self, two_directions):
        drop_record(two_directions)
        assert read(two_directions) == [[1, 2, 3], [4, 5, 6]]

    def test_some_configured_fields_without_a_column_raise(self, two_directions, config_override):
        drop_record(two_directions)
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction', 'sample']\n")
        with pytest.raises(RuntimeError, match=r"not for \['sample'\]"):
            read(two_directions)

    def test_no_configured_field_with_a_column_reads_as_one_workspace(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange -- a single-workspace file from another producer, with a field PyRS does not know
        path = tmp_path / "foreign.nxs"
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        with NXstress(path, "w") as nx:
            nx.write([ws], [[minimal_PeakCollection(N_subrun=3)]])
        drop_record(path)
        add_foreign_column(path)

        # Act / Assert -- used to be refused: the extra column was taken for a discriminator
        assert read(path) == [[1, 2, 3]]

    def test_an_omitted_discriminator_is_not_detected(self, two_directions, config_override):
        """The accepted cost of configuration being the authority: with no record,
        nothing distinguishes an unconfigured discriminator from another producer's
        field, so the two workspaces read back as one. A file PyRS writes always
        carries the record, which refuses this (see `TestReadWithTheRecord`)."""
        drop_record(two_directions)
        configure(config_override, "nxstress:\n  discriminator_fields: []\n")
        assert read(two_directions) == [[1, 2, 3, 4, 5, 6]]


class TestAppend:
    def test_an_entry_with_a_foreign_column_is_refused(
        self, two_directions, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """The append would grow every column but this one, leaving the group misaligned."""
        # Arrange
        add_foreign_column(two_directions)
        points = np.array([7, 8])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
        ws.set_sample_log("direction", points, np.array(["33", "33"]))
        before = hashlib.sha256(two_directions.read_bytes()).hexdigest()

        # Act / Assert
        with NXstress(two_directions, "a") as nx:
            with pytest.raises(RuntimeError, match=r"column\(s\) \['intensity'\]"):
                nx.write([ws], [[minimal_PeakCollection(N_subrun=2, peak_tag="Fe 211", sub_runs=points)]])
        assert hashlib.sha256(two_directions.read_bytes()).hexdigest() == before
