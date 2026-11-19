"""
The strain/stress viewer's NXstress actions: enabled by configuration and content, never hidden.

**Save as NXstress...** is enabled only when `nxstress.enable` is set and every
loaded workspace can be written -- it must carry instrument geometry, which
`NXstress.write` requires and which project files with an incomplete
`geometry setup` do not provide. When it cannot be used it is greyed out, not
removed: a hidden action gives no hint that the feature exists. **Load NXstress...**
follows `nxstress.enable` alone, which switches the whole NXstress feature set.

GUI tier: these drive a real `QMainWindow` and its `QAction`s.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.strainstressviewer.controller import Controller
from pyrs.interface.strainstressviewer import strain_stress_view
from pyrs.interface.strainstressviewer.model import Model
from pyrs.interface.strainstressviewer.strain_stress_view import StrainStressViewer
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.projectfile.file_mode import HidraProjectFileMode
from pyrs.projectfile.file_object import HidraProjectFile
from tests.util.project_file_helpers import write_minimal_h5_project  # noqa: F401

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture
def window(qtbot):
    model = Model()
    viewer = StrainStressViewer(model, Controller(model))
    qtbot.addWidget(viewer)
    return viewer


@pytest.fixture
def fitted_project(
    write_minimal_h5_project,  # noqa: F811
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
) -> Callable[..., Path]:
    """A factory for a fitted project file, with or without instrument geometry."""

    def make(*, with_instrument: bool) -> Path:
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=with_instrument, with_masks=True)
        points = ws.get_sub_runs().raw_copy()
        ws.set_sample_log("run_number", points, np.full(len(points), 1327))
        # A line scan along vx: the window plots the strain field on load, and its
        # interpolation fails for points on a diagonal through 3-D space (DD-32).
        ws.set_sample_log("vy", points, np.zeros(len(points)))
        ws.set_sample_log("vz", points, np.zeros(len(points)))
        path = write_minimal_h5_project(
            ws, filename=f"fitted_{with_instrument}.h5", with_instrument=with_instrument, with_masks=True
        )
        peaks = minimal_PeakCollection(N_subrun=len(points), sub_runs=points, peak_tag="Fe110")
        project = HidraProjectFile(str(path), mode=HidraProjectFileMode.READWRITE)
        project.write_peak_parameters(peaks)
        project.save()
        project.close()
        return path

    return make


def test_save_is_disabled_but_visible_when_nxstress_is_off(window, config_override, fitted_project):
    # Arrange
    config_override({"nxstress": {"enable": False}})
    window.model.e11 = [str(fitted_project(with_instrument=True))]

    # Act
    window.update_nxstress_action()

    # Assert
    assert not window.saveNXstressAction.isEnabled()
    assert window.saveNXstressAction.isVisible()


def test_save_is_disabled_with_nothing_loaded(window):
    assert not window.saveNXstressAction.isEnabled()
    assert window.saveNXstressAction.isVisible()


def test_save_is_enabled_once_a_writable_direction_is_loaded(window, fitted_project):
    # Arrange
    window.model.e11 = [str(fitted_project(with_instrument=True))]

    # Act
    window.update_nxstress_action()

    # Assert
    assert window.saveNXstressAction.isEnabled()


def test_save_stays_disabled_for_a_workspace_without_geometry(window, fitted_project):
    """The strain/stress project files in `tests/data` are in this state."""
    # Arrange
    window.model.e11 = [str(fitted_project(with_instrument=False))]

    # Act
    window.update_nxstress_action()

    # Assert
    assert not window.saveNXstressAction.isEnabled()
    assert window.saveNXstressAction.isVisible()


def test_load_follows_nxstress_enable(window, config_override):
    # Arrange
    assert window.loadNXstressAction.isEnabled()
    config_override({"nxstress": {"enable": False}})

    # Act
    window.update_nxstress_action()

    # Assert
    assert not window.loadNXstressAction.isEnabled()
    assert window.loadNXstressAction.isVisible()


def test_file_fields_name_an_nxstress_file_once(window, test_data_dir, tmp_path, monkeypatch):
    """After Load NXstress, and after reloading a state saved from it, each field shows the `.nxs` once.

    e11 holds two runs, so a saved state lists the `.nxs` twice for it. In-plane
    strain needs no e33: these three are the real runs with geometry that share
    one set of sample logs, which NXstress requires of an entry.
    """
    # Arrange
    model = window.model
    files = {"11": ["HB2B_1331", "HB2B_1332"], "22": ["HB2B_1328"]}
    for direction, stems in files.items():
        setattr(model, f"e{direction}", [str(Path(test_data_dir) / f"{stem}_with_instrument.h5") for stem in stems])
    model.selectedPeak = "peak0"
    model.calculate_stress("in-plane-strain", 200.0, 0.3, None)
    path = tmp_path / "stress.nxs"
    model.save_as_nxstress(str(path))
    monkeypatch.setattr(strain_stress_view.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    state = tmp_path / "state.json"

    def fields() -> list[str]:
        return [getattr(window.fileLoading, f"file_load_e{d}").lineEdit.text() for d in ("11", "22")]

    # Act / Assert -- Load NXstress
    window.load_nxstress()
    assert fields() == ["stress.nxs"] * 2

    # Act / Assert -- a state saved now, loaded back
    model.calculate_stress("in-plane-strain", 200.0, 0.3, None)
    model.to_json(str(state))
    for direction in files:
        window.fileLoading.set_text_values(direction, "")
    window.controller.load(str(state))
    assert fields() == ["stress.nxs"] * 2
