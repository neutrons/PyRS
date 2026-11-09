"""Unit tests for `TextureFittingModel` saving (#1080): no file is read or written."""

import pytest

from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel


@pytest.mark.parametrize("wedge, expected", [("eta_-5.0", "peak0_eta_-5.0"), ("", "peak0")])
def test_saved_peak_tag(wedge, expected):
    """A wedge's peaks get the wedge name appended; a fit without wedges keeps the label."""
    # Act
    tag = TextureFittingModel.saved_peak_tag("peak0", wedge)

    # Assert
    assert tag == expected


def test_save_without_loaded_file_reports_failure():
    """Saving before a project file is loaded reports an error instead of raising."""
    # Arrange
    model = TextureFittingModel(peak_fit_core=None)
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    # Act
    model.save_fit_result("never_written.h5", {"": object()})

    # Assert
    assert len(failures) == 1
    assert "No project file is loaded" in failures[0][1]
