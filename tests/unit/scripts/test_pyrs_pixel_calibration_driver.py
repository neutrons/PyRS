"""Unit tests for the ring-calibration driver (scripts/pyrs_pixel_calibration.py), on a synthetic run

The NeXus conversion is replaced by a synthetic workspace, so everything after it (ring extraction,
refinement, error estimate, the calibration file and the summary) runs for real without data files.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyrs.calibration.powder_standards import HB2B_STANDARDS
from pyrs.utilities.calibration_file_io import read_calibration_json_file
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace

REPO = Path(__file__).resolve().parents[3]
TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])


@pytest.fixture(scope="module")
def driver():
    # loaded by path: importing it as scripts.pyrs_pixel_calibration makes mypy see the file twice
    spec = importlib.util.spec_from_file_location(
        "pyrs_pixel_calibration", REPO / "scripts" / "pyrs_pixel_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic_run():
    return synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS)


def _stub_conversion(monkeypatch, workspace):
    """Make NeXusConvertingApp return ``workspace``; returns the (nexus, mask) arguments it saw"""
    seen = []

    class _Converter:
        def __init__(self, nexus_file, mask_file=None):
            seen.append((nexus_file, mask_file))

        def convert(self):
            return workspace

    monkeypatch.setattr("pyrs.core.nexus_conversion.NeXusConvertingApp", _Converter)
    return seen


@pytest.fixture
def nexus_file(tmp_path):
    path = tmp_path / "HB2B_0001.nxs.h5"
    path.write_bytes(b"")  # only its existence is checked; the conversion is stubbed
    return path


def test_main_writes_calibration(driver, monkeypatch, synthetic_run, nexus_file, tmp_path, capsys):
    # Arrange
    seen = _stub_conversion(monkeypatch, synthetic_run)
    output, rings = tmp_path / "cal.json", tmp_path / "rings.npz"

    # Act
    status = driver.main(
        [str(nexus_file), "-o", str(output), "--no-mask", "--wavelength", "1.54", "--ring-points", str(rings)]
    )

    # Assert - a complete file reduction reads, 2theta_0 held, the true wavelength recovered
    assert status == 0
    assert seen == [(str(nexus_file), None)]
    raw = json.loads(output.read_text())
    assert raw["error_TTH_0"] == -1.0
    assert all(raw[f"error_{name}"] > 0 for name in ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z"))
    wavelength, wavelength_error = read_calibration_json_file(str(output))[2:4]
    assert abs(wavelength - TRUE_PARAMS[7]) < max(4 * wavelength_error, 1e-5)
    assert rings.is_file()
    out = capsys.readouterr().out
    assert "TTH_0" in out and "held" in out and f"written to {output}" in out


def test_main_sub_runs_and_no_polish(driver, monkeypatch, synthetic_run, nexus_file, tmp_path):
    # Arrange
    _stub_conversion(monkeypatch, synthetic_run)
    output, rings = tmp_path / "cal.json", tmp_path / "rings.npz"
    args = [str(nexus_file), "-o", str(output), "--no-mask", "--wavelength", "1.54", "--ring-points", str(rings)]

    # Act
    status = driver.main(args + ["--sub-runs", "1", "2", "3", "4", "--no-polish", "--passes", "2"])

    # Assert
    from pyrs.calibration.pixel_pos_calibration import RingPoints

    assert status == 0
    assert set(RingPoints.load(rings).sub_run.tolist()) <= {1, 2, 3, 4}


def test_main_no_ring_points_returns_1(driver, monkeypatch, nexus_file, tmp_path, capsys):
    # Arrange - rings barely above a high background: extraction finds nothing
    faint = synthetic_calibration_workspace(
        default_hb2b_geometry(), TRUE_PARAMS, [76.0], [HB2B_STANDARDS[0]], peak_height=0.02, background=500.0
    )
    _stub_conversion(monkeypatch, faint)
    output = tmp_path / "cal.json"

    # Act
    status = driver.main([str(nexus_file), "-o", str(output), "--no-mask", "--wavelength", "1.54"])

    # Assert
    assert status == 1
    assert "no ring points" in capsys.readouterr().err
    assert not output.exists()


def test_main_missing_nexus_returns_1(driver, tmp_path, capsys):
    # Act
    status = driver.main([str(tmp_path / "missing.nxs.h5"), "-o", str(tmp_path / "cal.json"), "--no-mask"])

    # Assert
    assert status == 1
    assert "does not exist" in capsys.readouterr().err


def test_mask_and_no_mask_together_raises(driver):
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["run.nxs.h5", "-o", "cal.json", "--mask", "m.xml", "--no-mask"])


def test_parser_defaults(driver):
    # Act
    args = driver.build_parser().parse_args(["run.nxs.h5", "-o", "cal.json"])

    # Assert
    assert args.mask == driver.DEFAULT_MASK and not args.no_mask
    assert (args.strips, args.passes, args.no_polish, args.wavelength) == (5, 3, False, None)


@pytest.mark.parametrize(
    "error, name, text",
    [
        (-1.0, "TTH_0", "held (2theta_0"),
        (-1.0, "Rot_x", "held"),
        (-2.0, "Shift_x", "undetermined"),
        (1e-4, "Shift_x", "+/- 0.1"),
    ],
)
def test_summary_labels_error_codes(driver, monkeypatch, error, name, text):
    # Arrange - a stand-in carrying just what _summary reads
    from types import SimpleNamespace

    from pyrs.calibration.pixel_pos_calibration import PARAMETER_NAMES
    from tests.util.ring_calibration_helpers import synthetic_ring_points

    errors = np.full(8, 1e-4)
    errors[PARAMETER_NAMES.index(name)] = error
    calibration = SimpleNamespace(
        ring_points=synthetic_ring_points(default_hb2b_geometry(), TRUE_PARAMS),
        stages=[SimpleNamespace(rmse=0.001)],
        calibration_array=TRUE_PARAMS,
        calibration_error_array=errors,
        calibration_status=1,
    )

    # Act
    summary = driver._summary(calibration)
    line = next(line for line in summary.splitlines() if line.split()[0] == name)

    # Assert
    assert text in line and "-2000" not in line
    assert "WARNING" not in summary


def test_summary_flags_unconverged_result(driver):
    # Arrange - status 0: the last stage stopped at max_nfev
    from types import SimpleNamespace

    from tests.util.ring_calibration_helpers import synthetic_ring_points

    calibration = SimpleNamespace(
        ring_points=synthetic_ring_points(default_hb2b_geometry(), TRUE_PARAMS),
        stages=[SimpleNamespace(rmse=0.001)],
        calibration_array=TRUE_PARAMS,
        calibration_error_array=np.full(8, 1e-4),
        calibration_status=0,
    )

    # Act / Assert
    assert "WARNING: the last stage stopped at max_nfev before converging" in driver._summary(calibration)
