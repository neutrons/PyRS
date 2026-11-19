"""
The strain/stress viewer's model over project files: its saved state (`Model.to_json` /
`Model.from_json`), and loading a direction from a file it cannot use.

A file that fails to load is reported and leaves the model empty: a state
restores whole or not at all. A restore that loads but cannot be completed -- a
peak one direction lacks, a file with no fitted peaks -- is reported by the
window and leaves stress uncalculated, as the window itself does before it has
every input. Neither raises out of the Load action. States naming NXstress files are covered beside the rest of the
NXstress round trip, in `test_nxstress_viewer_roundtrip.py`.

Integration tier: real project files from `tests/data`. Imports only the model,
so the module loads without a display.
"""

import json
import os

import pytest

from pyrs.interface.strainstressviewer.model import Model

pytestmark = pytest.mark.integration


def write_state(path, test_data_dir: str, **overrides) -> str:
    """A saved state: in-plane stress from HB2B_1320 in 11 and 22, with `overrides` applied."""
    state = {
        "stress_case": "in-plane-stress",
        "filenames_11": [os.path.join(test_data_dir, "HB2B_1320.h5")],
        "filenames_22": [os.path.join(test_data_dir, "HB2B_1320.h5")],
        "filenames_33": [],
        "peak_tag": "peak0",
        "youngs_modulus": 200,
        "poisson_ratio": 0.3,
        "d0": {"d0": 1.0123, "d0_error": 0.000123},
    }
    state.update(overrides)
    with open(path, "w") as f:
        json.dump(state, f)
    return str(path)


@pytest.fixture
def model_and_failures() -> tuple[Model, list]:
    """A fresh model, and the `failureMsg` reports it emits."""
    model = Model()
    failures: list = []
    model.failureMsg.connect(lambda *args: failures.append(args))
    return model, failures


def test_a_missing_file_is_reported(model_and_failures, test_data_dir, tmp_path):
    """The whole model is left empty, not only that direction."""
    # Arrange
    model, failures = model_and_failures
    state = write_state(tmp_path / "state.json", test_data_dir, filenames_22=[str(tmp_path / "missing.h5")])

    # Act
    model.from_json(state)

    # Assert
    assert len(failures) == 1 and "missing.h5" in failures[0][0]
    assert model.e11 == model.e22 == model.e33 == []
    assert model.stress is None
    assert model.modelUpdated == (None, None, None, None)


def test_a_peak_one_direction_lacks_is_not_a_crash(model_and_failures, test_data_dir, tmp_path):
    """HB2B_1328 holds peak1 and HB2B_1320 does not: the window names the gap."""
    # Arrange
    model, failures = model_and_failures
    state = write_state(
        tmp_path / "state.json",
        test_data_dir,
        filenames_11=[os.path.join(test_data_dir, "HB2B_1328.h5")],
        peak_tag="peak1",
    )

    # Act
    model.from_json(state)

    # Assert
    assert failures == []
    assert model.stress is None
    assert model.validate_selection("22") == "Peak peak1 is not in e22"


def test_diagonal_without_e33_holds_stress_back(model_and_failures, test_data_dir, tmp_path):
    # Arrange
    model, failures = model_and_failures
    state = write_state(tmp_path / "state.json", test_data_dir, stress_case="diagonal")

    # Act
    model.from_json(state)

    # Assert
    assert failures == []
    assert model.stress is None
    assert model.validate_selection("33") == "e33 file hasn't been loaded"


def test_a_file_with_no_peaks_is_never_mispaired(model_and_failures, test_data_dir, tmp_path):
    """HB2B_1423 has no fitted peaks. Its workspace must not take HB2B_1320's peaks."""
    # Arrange
    model, failures = model_and_failures
    files = [os.path.join(test_data_dir, name) for name in ("HB2B_1423.h5", "HB2B_1320.h5")]
    state = write_state(tmp_path / "state.json", test_data_dir, filenames_11=files)

    # Act
    model.from_json(state)

    # Assert
    assert failures == []
    assert len(model.e11) == len(model.e11_peaks) == 2
    assert model.e11_peaks[0] == {}
    assert model.stress is None
    assert model.validate_selection("11") == "Peak peak0 is not in e11"


def test_a_file_with_no_peaks_loads_as_e22(test_data_dir):
    """A file without fitted peaks is reported as lacking the peak, never an `IndexError` from its strain."""
    # Arrange
    model = Model()
    model.e11 = os.path.join(test_data_dir, "HB2B_1320.h5")

    # Act
    model.e22 = os.path.join(test_data_dir, "HB2B_1423.h5")

    # Assert
    assert len(model.e22) == 1
    assert model.validate_selection("22") == "Peak None is not in e22"


def test_a_restore_over_a_calculated_stress_rebuilds_it_with_its_d0(model_and_failures, test_data_dir, tmp_path):
    """Every restore path clears the stress first, so the saved d0 is applied to a new one."""
    # Arrange
    model, failures = model_and_failures
    model.from_json(write_state(tmp_path / "first.json", test_data_dir))
    stress = model.stress
    second = write_state(tmp_path / "second.json", test_data_dir, d0={"d0": 1.05, "d0_error": 0.0002})

    # Act
    model.from_json(second)

    # Assert
    assert failures == []
    assert stress is not None and model.stress is not stress
    assert model.d0.values.tolist() == pytest.approx([1.05] * len(model.d0.values))
