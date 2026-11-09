"""Texture fitting saves the fit of every η-wedge into the project file (#1080)

Every wedge is fitted with the same peak labels, so each wedge's peaks are saved as
``<label>_<wedge>`` (e.g. ``peak0_eta_-5.0``).
"""

import hashlib
import shutil

import pytest

from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration

TEXTURE_FILE = "tests/data/HB2B_1599_texture.h5"
FIT_WINDOW = (62.864, 66.9115)  # the window tests/ui/test_texture_fitting.py fits


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _peak_tags(path):
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    tags = project.read_peak_tags()
    project.close()
    return set(tags)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A model with a copy of the texture file loaded, and its fits keyed by η-wedge."""
    path = tmp_path_factory.mktemp("texture") / "HB2B_1599_texture.h5"
    shutil.copy(TEXTURE_FILE, path)
    model = TextureFittingModel(peak_fit_core=None)
    model.load_hidra_project_file(str(path))
    wedges = [mask for mask in model.ws.reduction_masks if "_var" not in mask]
    fits = {
        wedge: model.fit_diff_peaks(
            [FIT_WINDOW[0]], [FIT_WINDOW[1]], ["peak0"], "PseudoVoigt", "Linear", out_of_plane_angle=wedge
        )
        for wedge in wedges
    }
    return model, path, fits


def test_save_writes_every_wedge_under_its_own_tag(fitted):
    """Save updates the loaded file with one peak entry per wedge."""
    # Arrange
    model, path, fits = fitted
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(str(path), fits)
    model.save_fit_result(str(path), fits)  # saving again overwrites the same entries

    # Assert
    assert failures == []
    assert len(fits) > 1
    assert {f"peak0_{wedge}" for wedge in fits} <= _peak_tags(path)


def test_save_as_writes_a_copy_and_keeps_the_loaded_file(fitted, tmp_path):
    """Save As writes the wedges' peaks to a copy; the loaded file is not modified."""
    # Arrange
    model, path, fits = fitted
    digest = _digest(path)
    copy_path = tmp_path / "saved_as.h5"

    # Act
    model.save_fit_result(str(copy_path), fits)

    # Assert
    assert {f"peak0_{wedge}" for wedge in fits} <= _peak_tags(copy_path)
    assert _digest(path) == digest


def test_save_without_loaded_file_reports_failure(tmp_path):
    """Saving before a project file is loaded reports an error instead of raising."""
    # Arrange
    model = TextureFittingModel(peak_fit_core=None)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(str(tmp_path / "out.h5"), {"": object()})

    # Assert
    assert len(failures) == 1
    assert "No project file is loaded" in failures[0][1]


@pytest.mark.parametrize("wedge, expected", [("eta_-5.0", "peak0_eta_-5.0"), ("", "peak0")])
def test_saved_peak_tag(wedge, expected):
    """A wedge's peaks get the wedge name appended; a fit without wedges keeps the label."""
    assert TextureFittingModel.saved_peak_tag("peak0", wedge) == expected
