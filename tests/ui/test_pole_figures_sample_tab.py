"""Sample tab: load an STL, gauge center, mount conventions, the sample-to-stage matrix, preview redraw."""

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.interface.pole_figures.models.sample import SampleModel
from pyrs.interface.pole_figures.mount_matrix import IDENTITY9
from pyrs.interface.pole_figures.presenters.sample import SamplePresenter
from pyrs.interface.pole_figures.tabs.sample_tab import SampleTab
from tests.util.pole_figures_helpers import STL_DIR

pytestmark = [pytest.mark.gui, pytest.mark.integration]

SLAB_STL = STL_DIR / "slab.stl"
CYLINDER_STL = STL_DIR / "cylinder_D50mm_H50mm.stl"


@pytest.fixture
def presenter(qtbot: QtBot) -> SamplePresenter:
    tab = SampleTab()
    qtbot.addWidget(tab)
    return SamplePresenter(tab, model=SampleModel())


def test_load_stl_slab_centers_gauge_and_draws(presenter: SamplePresenter) -> None:
    # Arrange
    path = SLAB_STL

    # Act
    loaded = presenter.load_stl(path)

    # Assert
    assert loaded
    np.testing.assert_allclose(presenter.model.gauge_center(), [17.7, 27.3, 1.65])
    assert "closed" in presenter.view.info_label.text()
    assert len(presenter.view.canvas.figure.axes) == 3
    assert presenter.gauge_inside()


def test_load_stl_bad_file_shows_error(
    presenter: SamplePresenter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    shown: list[str] = []
    monkeypatch.setattr(presenter.view, "show_error", lambda title, text: shown.append(title))
    bad = tmp_path / "bad.stl"
    bad.write_text("solid nothing\nendsolid")

    # Act
    loaded = presenter.load_stl(bad)

    # Assert
    assert not loaded and shown and presenter.mesh is None


def test_form_edits_update_model(presenter: SamplePresenter, qtbot: QtBot) -> None:
    # Arrange
    presenter.load_stl(SLAB_STL)
    w = presenter.view.gauge_form.widgets
    m = presenter.view.mount_form.widgets
    p = presenter.view.preview_form.widgets["preview_chi"]
    assert isinstance(w["gauge_vx"], QDoubleSpinBox)
    assert isinstance(m["incident_from"], QComboBox)
    assert isinstance(p, QDoubleSpinBox)

    # Act
    w["gauge_vx"].setValue(29.4)
    m["incident_from"].setCurrentText("-vy")
    p.setValue(40)

    # Assert
    assert presenter.model.gauge_vx == pytest.approx(29.4)
    assert presenter.model.conventions() == MountConventions(incident_from="-vy", sample_to_instrument=IDENTITY9)
    assert np.array_equal(presenter.model.conventions().M0, np.eye(3))  # identity = PF's mount
    assert presenter.model.preview_chi == 40
    qtbot.waitUntil(lambda: "chi 40" in presenter.view.canvas.figure.axes[1].get_title(loc="left"), timeout=3000)


def test_init_spin_box_limits_from_model(qtbot: QtBot) -> None:
    # Arrange
    tab = SampleTab()
    qtbot.addWidget(tab)

    # Act
    spin = tab.preview_form.widgets["preview_tth"]

    # Assert
    assert isinstance(spin, QDoubleSpinBox)
    assert spin.minimum() == pytest.approx(0.01) and spin.maximum() == pytest.approx(179.99)
    assert SampleModel.model_fields["preview_tth"].title == "2theta (deg)"


def test_rotate_chi_then_phi_axis(presenter: SamplePresenter) -> None:
    """Rotate buttons turn the sample about the stage axes. The turns add up."""
    # Arrange: the cylinder's long axis is vz
    v = presenter.view
    presenter.load_stl(CYLINDER_STL)
    assert v.matrix_values() == IDENTITY9 and "+vx up along the phi axis" in v.matrix_note.text()

    # Act: +90 about the chi axis takes +vz to the phi axis
    v.rotate_angles[1].setValue(90)
    v.rotate_buttons[1].click()

    # Assert
    m0 = presenter.model.conventions().M0
    assert np.allclose(m0, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert "+vz up along the phi axis" in v.matrix_note.text() and "-vx along Q" in v.matrix_note.text()
    assert np.allclose(v.matrix_values(), m0.ravel())

    # Act: then 90 about the phi axis: vz stays up
    v.rotate_angles[0].setValue(90)
    v.rotate_buttons[0].click()

    # Assert
    assert presenter.model.conventions().axes_at_zero().up == "+vz"


def test_set_mount_matrix_invalid_edit_refused_then_reset(presenter: SamplePresenter) -> None:
    # Arrange
    v = presenter.view
    presenter.load_stl(CYLINDER_STL)
    v.rotate_angles[1].setValue(90)
    v.rotate_buttons[1].click()
    before = presenter.model.mount_matrix

    # Act: a typed value that breaks the rotation
    v.matrix_spins[0][0].setValue(0.5)

    # Assert: refused, the last valid matrix stays in use
    assert presenter.model.mount_matrix == before and "not a rotation" in v.matrix_note.text()

    # Act
    v.reset_mount_button.click()

    # Assert
    assert presenter.model.mount_matrix == IDENTITY9 and v.matrix_values() == IDENTITY9


def test_matrix_typed_valid_made_orthonormal(presenter: SamplePresenter) -> None:
    # Arrange: a full valid matrix typed with 4 decimals
    v = presenter.view
    presenter.load_stl(CYLINDER_STL)
    vals = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)

    # Act
    for i in range(3):
        for j in range(3):
            v.matrix_spins[i][j].setValue(vals[3 * i + j])

    # Assert
    assert np.allclose(presenter.model.conventions().M0, np.reshape(vals, (3, 3)))
