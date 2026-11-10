"""Re-writing a peak whose profile, background or sub-runs changed replaces its datasets

h5py's ``dataset[...] = array`` silently accepts a different compound dtype, so re-saving a peak
fitted with another profile (PseudoVoigt -> Gaussian) used to keep the old fields' values beside
the new ones; a different number of sub-runs raised ``TypeError`` instead.
"""

import numpy as np
import pytest

from pyrs.core.peak_profile_utility import BackgroundFunction, PeakShape
from pyrs.peaks import PeakCollection  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

pytestmark = pytest.mark.integration


def _peaks(shape, background, sub_runs, offset):
    """A peak collection whose parameter values are ``offset + index``."""
    names = shape.native_parameters + background.native_parameters
    data_type = [(name, np.float32) for name in names]
    values = np.zeros(len(sub_runs), dtype=data_type)
    errors = np.zeros(len(sub_runs), dtype=data_type)
    for index, name in enumerate(names):
        values[name] = offset + index
        errors[name] = 0.1
    peaks = PeakCollection("peak0", shape, background)
    peaks.set_peak_fitting_values(np.array(sub_runs), values, errors, np.full(len(sub_runs), 0.5))
    return peaks


def _rewrite(tmp_path, first, second):
    path = str(tmp_path / "peaks.h5")
    with HidraProjectFile(path, HidraProjectFileMode.OVERWRITE) as project:
        project.write_wavelength(1.54)
        project.write_peak_parameters(first)
        project.write_peak_parameters(second)
        project.save(False)
    with HidraProjectFile(path, HidraProjectFileMode.READONLY) as project:
        return project.read_peak_parameters("peak0")


@pytest.mark.parametrize(
    "first_shape, first_background, second_shape, second_background",
    [
        (PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, PeakShape.GAUSSIAN, BackgroundFunction.LINEAR),
        (PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, PeakShape.PSEUDOVOIGT, BackgroundFunction.QUADRATIC),
    ],
    ids=["new-profile", "new-background"],
)
def test_rewrite_with_new_model_keeps_only_new_values(
    tmp_path, first_shape, first_background, second_shape, second_background
):
    """The re-written peak reads back exactly as the second fit, with no leftover fields."""
    # Arrange
    first = _peaks(first_shape, first_background, [1, 2, 3], offset=100.0)
    second = _peaks(second_shape, second_background, [1, 2, 3], offset=1.0)

    # Act
    loaded = _rewrite(tmp_path, first, second)

    # Assert
    assert str(loaded.peak_profile) == str(second_shape)
    assert str(loaded.background_type) == str(second_background)
    values, _ = loaded.get_native_params()
    expected, _ = second.get_native_params()
    assert values.dtype.names == expected.dtype.names
    for name in expected.dtype.names:
        np.testing.assert_allclose(values[name], expected[name])


def test_rewrite_with_more_sub_runs(tmp_path):
    """A peak re-written over a different number of sub-runs replaces the old arrays."""
    # Arrange
    first = _peaks(PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, [1, 2, 3], offset=100.0)
    second = _peaks(PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, [1, 2, 3, 4, 5], offset=1.0)

    # Act
    loaded = _rewrite(tmp_path, first, second)

    # Assert
    assert list(loaded.sub_runs) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(loaded.get_native_params()[0]["Intensity"], second.get_native_params()[0]["Intensity"])


def test_rewrite_with_same_model_updates_values(tmp_path):
    """The common case, same model and sub-runs, still overwrites the values."""
    # Arrange
    first = _peaks(PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, [1, 2, 3], offset=100.0)
    second = _peaks(PeakShape.PSEUDOVOIGT, BackgroundFunction.LINEAR, [1, 2, 3], offset=1.0)

    # Act
    loaded = _rewrite(tmp_path, first, second)

    # Assert
    np.testing.assert_allclose(loaded.get_native_params()[0]["Intensity"], second.get_native_params()[0]["Intensity"])
