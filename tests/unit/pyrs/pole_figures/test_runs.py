"""Reading reduced runs: run numbers, file checks, live and invalid sub-runs, and the logged gauge position."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyrs.pole_figures.core import runs
from pyrs.pole_figures.core.runs import RunData, read_run, run_number
from tests.util.pole_figures_helpers import replace_dataset, write_reduced_run


# ---------------------------------------------------------------- run numbers and file checks
@pytest.mark.parametrize(("name", "want"), [("x/HB2B_6494.h5", "6494"), ("other_file.h5", "other_file")])
def test_run_number(name: str, want: str) -> None:
    # Arrange
    path = name

    # Act
    run = run_number(path)

    # Assert
    assert run == want


def test_read_run_file_without_reduced_data_raises(tmp_path: Path) -> None:
    # Arrange
    f = tmp_path / "HB2B_1.h5"
    with h5py.File(f, "w") as h:
        h["something"] = [1, 2]

    # Act / Assert
    with pytest.raises(ValueError, match="not a reduced"):
        read_run(f)


def test_read_run_not_hdf5_raises(tmp_path: Path) -> None:
    # Arrange
    f = tmp_path / "bad.h5"
    f.write_text("not hdf5")

    # Act / Assert
    with pytest.raises(ValueError, match="cannot read"):
        read_run(f)


def test_read_run_nan_wavelength_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_4.h5")
    replace_dataset(path, "instrument/monochromator setting/wave length", np.array([np.nan]))

    # Act / Assert
    with pytest.raises(ValueError, match="not a valid HB-2B wavelength"):
        read_run(path)


def test_read_run_without_two_theta_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_5.h5")
    with h5py.File(path, "a") as h:
        del h["reduced diffraction data/2theta"]

    # Act / Assert
    with pytest.raises(ValueError, match="no reduced diffraction data/2theta"):
        read_run(path)


@pytest.mark.parametrize(
    ("name", "values"),
    [("raw data/logs/chi", np.zeros(5)), ("raw data/logs/sub-run duration", np.full(7, 30.0)),
     ("reduced diffraction data/main", np.ones((5, 200)))],
)  # fmt: skip
def test_read_run_lengths_differ_raises(tmp_path: Path, name: str, values: np.ndarray) -> None:
    """Six sub-runs in every other log: one log or the spectra with another length is refused, naming the file."""
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_7.h5")
    replace_dataset(path, name, values)

    # Act / Assert
    with pytest.raises(ValueError, match="HB2B_7.h5: the logs and spectra must have one value per sub-run"):
        read_run(path)


def test_read_run_spectra_width_differs_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_8.h5")
    replace_dataset(path, "reduced diffraction data/2theta", np.linspace(80.0, 100.0, 150)[None])

    # Act / Assert
    with pytest.raises(ValueError, match="HB2B_8.h5: the reduced spectra must be"):
        read_run(path)


def test_read_run_no_subruns_raises(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_9.h5")
    with h5py.File(path, "a") as h:
        for name in list(h["raw data/logs"]):
            values = h["raw data/logs"][name][()]
            del h["raw data/logs"][name]
            h["raw data/logs"][name] = values[:0]
    replace_dataset(path, "reduced diffraction data/main", np.ones((0, 200)))

    # Act / Assert
    with pytest.raises(ValueError, match="HB2B_9.h5 has no sub-runs"):
        read_run(path)


@pytest.mark.parametrize(
    ("stored", "want"),
    [(np.array([b"Pole figure (222)"] * 6), "Pole figure (222)"),
     (np.array([b"bad \xff byte"] * 6), "bad \ufffd byte"),
     (np.array(["text title"] * 6, dtype=h5py.string_dtype()), "text title")],
)  # fmt: skip
def test_read_run_title_is_decoded(tmp_path: Path, stored: np.ndarray, want: str) -> None:
    """Bytes are UTF-8 with bad bytes replaced. Variable-length strings are read as they are."""
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_10.h5")
    with h5py.File(path, "a") as h:
        h["raw data/logs/run_title"] = stored

    # Act
    title = read_run(path).title

    # Assert
    assert title == want


def test_hdf5_paths_match_pyrs_hidra_constants() -> None:
    """The file paths and log names are HidraConstants' strings. Skipped where PyRS is absent."""
    # Arrange
    constants = pytest.importorskip("pyrs.dataobjects.constants")
    hc = constants.HidraConstants

    # Act
    expected = (f"{hc.RAW_DATA}/{hc.SAMPLE_LOGS}", f"{hc.REDUCED_DATA}/{hc.REDUCED_MAIN}",
                f"{hc.REDUCED_DATA}/{hc.TWO_THETA}", f"{hc.INSTRUMENT}/{hc.MONO}/{hc.WAVELENGTH}",
                hc.SUB_RUN_DURATION, hc.SUB_RUNS, tuple(hc.SAMPLE_COORDINATE_NAMES))  # fmt: skip

    # Assert
    assert (runs.LOGS_PATH, runs.REDUCED_MAIN_PATH, runs.REDUCED_2THETA_PATH, runs.WAVELENGTH_PATH,
            runs.DURATION_LOG, runs.SUB_RUNS_LOG, tuple(runs.POSITION_KEYS)) == expected  # fmt: skip


def test_read_run_notes_omega_motion(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_6.h5")
    replace_dataset(path, "raw data/logs/omega", np.array([45.0, 45.0, 45.0, 45.0, 45.0, 45.5]))

    # Act
    notes = read_run(path).notes

    # Assert
    assert any("omega moved by 0.500" in n for n in notes)


# ---------------------------------------------------------------- live and invalid sub-runs
@pytest.fixture
def run_with_odd_durations(tmp_path: Path) -> RunData:
    """A long real count (1200 s), a NaN duration and the placeholder of a last, still counting sub-run."""
    path = write_reduced_run(tmp_path / "HB2B_3.h5", n=6)
    replace_dataset(path, "raw data/logs/sub-run duration", np.array([30.0, 1200.0, 30.0, np.nan, 30.0, 99999.0]))
    return read_run(path)


def test_read_run_skips_last_live_subrun(run_with_odd_durations: RunData) -> None:
    # Arrange
    run = run_with_odd_durations

    # Act
    skipped, live = run.live_skipped, run.live

    # Assert
    assert (skipped, live) == (1, True)


def test_read_run_keeps_long_count(run_with_odd_durations: RunData) -> None:
    # Arrange
    run = run_with_odd_durations

    # Act
    count_s = run.count_s

    # Assert
    assert 1200.0 in count_s


def test_read_run_skips_invalid_subrun(run_with_odd_durations: RunData) -> None:
    # Arrange
    run = run_with_odd_durations

    # Act
    text = run.describe()

    # Assert
    assert run.invalid_skipped == 1
    assert "non-finite" in text


# ---------------------------------------------------------------- logged gauge position
def test_logged_position_is_read(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_1.h5", position=(29.4, 19.75, 1.65))

    # Act
    run = read_run(path)

    # Assert
    assert run.logged_position is not None
    assert np.allclose(run.logged_position, [29.4, 19.75, 1.65])


def test_steady_position_has_no_spread(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_1.h5", position=(29.4, 19.75, 1.65))

    # Act
    run = read_run(path)

    # Assert
    assert run.position_spread == 0.0


def test_nan_position_means_not_logged(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_2.h5", position=None)

    # Act
    run = read_run(path)

    # Assert
    assert run.logged_position is None
    assert run.position_spread == 0.0


def test_position_spread_of_moving_run(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_3.h5", position=(10.0, 5.0, 1.0), moving=0.5)

    # Act
    run = read_run(path)

    # Assert
    assert run.logged_position is not None
    assert abs(run.position_spread - 0.5) < 1e-9


@pytest.fixture
def run_with_skipped_tail(tmp_path: Path) -> RunData:
    return read_run(write_reduced_run(tmp_path / "HB2B_4.h5", position=(10.0, 5.0, 1.0), short_tail=True))


def test_skipped_tail_subrun_is_left_out(run_with_skipped_tail: RunData) -> None:
    # Arrange
    run = run_with_skipped_tail

    # Act
    skipped = run.short_skipped

    # Assert
    assert skipped == 1


def test_skipped_tail_subrun_does_not_count_as_motion(run_with_skipped_tail: RunData) -> None:
    # Arrange
    run = run_with_skipped_tail

    # Act
    spread, position = run.position_spread, run.logged_position

    # Assert
    assert spread == 0.0
    assert position is not None and np.allclose(position, [10.0, 5.0, 1.0])


def test_missing_vz_log_says_why(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_5.h5", position=(10.0, 5.0, 1.0), keys=("vx", "vy"))

    # Act
    run = read_run(path)

    # Assert
    assert run.logged_position is None
    assert run.position_note == "vz not logged"


@pytest.fixture
def run_with_nan_subruns(tmp_path: Path) -> RunData:
    return read_run(write_reduced_run(tmp_path / "HB2B_6.h5", position=(10.0, 5.0, 1.0), nan_subruns=2))


def test_nan_subruns_keep_the_position(run_with_nan_subruns: RunData) -> None:
    # Arrange
    run = run_with_nan_subruns

    # Act
    position = run.logged_position

    # Assert
    assert position is not None and np.allclose(position, [10.0, 5.0, 1.0])


def test_nan_subruns_say_why(run_with_nan_subruns: RunData) -> None:
    # Arrange
    run = run_with_nan_subruns

    # Act
    note = run.position_note

    # Assert
    assert note == "2 of 6 sub-runs have no full vx, vy, vz"


@pytest.mark.parametrize("position", [(10.0, 5.0, 1.0), None], ids=["fully_logged", "not_logged"])
def test_position_note_empty_when_complete(tmp_path: Path, position: tuple[float, float, float] | None) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_7.h5", position=position)

    # Act
    note = read_run(path).position_note

    # Assert
    assert note == ""


def test_position_log_of_the_wrong_length_says_why(tmp_path: Path) -> None:
    # Arrange
    path = write_reduced_run(tmp_path / "HB2B_9.h5", position=(10.0, 5.0, 1.0))
    replace_dataset(path, "raw data/logs/vy", np.zeros(4))

    # Act
    run = read_run(path)

    # Assert
    assert run.logged_position is None
    assert "vy log has 4 values for 6 sub-runs" in run.position_note
