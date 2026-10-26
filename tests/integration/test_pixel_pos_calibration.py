"""Integration test of the ring-based calibration driver on a real calibration run"""

import importlib.util
import json
from pathlib import Path

import pytest

from pyrs.utilities.calibration_file_io import check_calibration_status, read_calibration_json_file

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
NEXUS_FILE = REPO / "tests" / "data" / "calibration_tests" / "HB2B_3510.nxs.h5"
MASK_FILE = REPO / "tests" / "data" / "HB2B_Mask_12-18-19.xml"


@pytest.fixture(scope="module")
def driver():
    # loaded by path: importing it as scripts.pyrs_pixel_calibration makes mypy see the file twice
    spec = importlib.util.spec_from_file_location(
        "pyrs_pixel_calibration", REPO / "scripts" / "pyrs_pixel_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_calibrates_hb2b_3510(tmp_path, driver, capsys):
    # Arrange
    output, rings = tmp_path / "HB2B_3510_cal.json", tmp_path / "rings.npz"

    # Act
    status = driver.main(
        [str(NEXUS_FILE), "--output", str(output), "--mask", str(MASK_FILE), "--ring-points", str(rings)]
    )

    # Assert - a file reduction's own reader and status check accept
    assert status == 0
    shift, shift_error, wavelength, wavelength_error, fit_status = read_calibration_json_file(str(output))
    check_calibration_status(fit_status)
    raw = json.loads(output.read_text())
    assert raw["error_TTH_0"] == -1.0  # held for the error estimate (degenerate with Shift_x/Rot_y)
    assert all(
        raw[f"error_{name}"] > 0 for name in ("Shift_x", "Shift_y", "Shift_z", "Rot_x", "Rot_y", "Rot_z", "Lambda")
    )
    # Si422 run: the refined wavelength, with the 2020 lattice constants (docs/ground_truths.md)
    assert wavelength == pytest.approx(1.5306, abs=1e-3)
    assert wavelength_error < 1e-3
    assert raw["Shift_z"] == pytest.approx(-0.075, abs=0.01)
    assert rings.is_file()
    assert "12 rings" in capsys.readouterr().out


def test_driver_missing_mask_fails_cleanly(tmp_path, driver, capsys):
    # Act
    status = driver.main([str(NEXUS_FILE), "--output", str(tmp_path / "x.json"), "--mask", str(tmp_path / "none.xml")])

    # Assert
    assert status == 1
    assert "--no-mask" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()
