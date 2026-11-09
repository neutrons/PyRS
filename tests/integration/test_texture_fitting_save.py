"""Texture fitting saves the fit of every η-wedge into the project file (#1080)

Every wedge is fitted with the same peak labels, so each wedge's peaks are saved as
``<label>_<wedge>`` (e.g. ``peak0_eta_-5.0``); a fit without wedges keeps the plain label.
All files written here are copies in a temporary directory, never ``tests/data``.
"""

import hashlib
import shutil

import numpy as np
import pytest

from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration

TEXTURE_FILE = "tests/data/HB2B_1599_texture.h5"
TEXTURE_WINDOW = (62.864, 66.9115)  # the window tests/ui/test_texture_fitting.py fits
UNSPLIT_FILE = "tests/data/HB2B_1327.h5"
UNSPLIT_WINDOW = (88.6, 91.6)


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _peak_tags(path):
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        return set(project.read_peak_tags())


def _saved_centres(path, tag):
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        values, _ = project.read_peak_parameters(tag).get_effective_params()
    return values["Center"]


def _fit_centres(fit_result):
    values, _ = fit_result.peakcollections[0].get_effective_params()
    return values["Center"]


def _fit_every_wedge(source, tmp_path, window):
    """A model with a copy of ``source`` loaded, and peak0 fitted for each of its wedges."""
    path = tmp_path / source.split("/")[-1]
    shutil.copy(source, path)
    model = TextureFittingModel(peak_fit_core=None)
    model.load_hidra_project_file(str(path))
    # a file not split into wedges is fitted once, keyed "" (its masks are [None] in older files,
    # or the main mask and its "_var" in newer ones)
    wedges = [mask for mask in model.ws.reduction_masks if mask and "_var" not in mask] or [""]
    fits = {
        wedge: model.fit_diff_peaks(
            [window[0]], [window[1]], ["peak0"], "PseudoVoigt", "Linear", out_of_plane_angle=wedge or None
        )
        for wedge in wedges
    }
    return model, path, fits


def test_save_writes_every_wedge_under_its_own_tag(tmp_path):
    """Save updates the loaded file with each wedge's own fit; saving again overwrites them."""
    # Arrange
    model, path, fits = _fit_every_wedge(TEXTURE_FILE, tmp_path, TEXTURE_WINDOW)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result(str(path), fits)
    model.save_fit_result(str(path), fits)

    # Assert
    assert failures == []
    assert len(fits) > 1
    assert _peak_tags(path) == {f"peak0_{wedge}" for wedge in fits}
    for wedge, fit_result in fits.items():
        np.testing.assert_allclose(_saved_centres(path, f"peak0_{wedge}"), _fit_centres(fit_result))


def test_save_as_writes_a_copy_and_keeps_the_loaded_file(tmp_path):
    """Save As writes the wedges' peaks to a copy; the loaded file is not modified."""
    # Arrange
    model, path, fits = _fit_every_wedge(TEXTURE_FILE, tmp_path, TEXTURE_WINDOW)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))
    digest = _digest(path)
    copy_path = tmp_path / "saved_as.h5"

    # Act
    model.save_fit_result(str(copy_path), fits)

    # Assert
    assert failures == []
    assert _peak_tags(copy_path) == {f"peak0_{wedge}" for wedge in fits}
    assert _digest(path) == digest


def test_save_of_unsplit_file_keeps_plain_label(tmp_path):
    """A file not split into wedges saves its fit under the plain peak label."""
    # Arrange
    model, path, fits = _fit_every_wedge(UNSPLIT_FILE, tmp_path, UNSPLIT_WINDOW)
    copy_path = tmp_path / "unsplit_saved.h5"

    # Act
    model.save_fit_result(str(copy_path), fits)

    # Assert
    assert list(fits) == [""]
    assert "peak0" in _peak_tags(copy_path)
    np.testing.assert_allclose(_saved_centres(copy_path, "peak0"), _fit_centres(fits[""]))
