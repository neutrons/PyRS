"""Tests for tests/util/peak_collection_helpers.py."""

import pytest

from tests.util.peak_collection_helpers import createPeakCollection  # noqa: F401


def test_createPeakCollection_inverted_error_fraction_bounds_raises_value_error(createPeakCollection) -> None:  # noqa: F811
    """Test that createPeakCollection rejects error_fraction_min > error_fraction_max."""
    with pytest.raises(ValueError, match="invalid error_fraction bounds"):
        createPeakCollection(
            peak_tag="t",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.0,
            projectfilename="/does/not/exist.h5",
            runnumber=1,
            N_subrun=1,
            error_fraction_min=0.05,
            error_fraction_max=0.005,  # inverted
        )


def test_createPeakCollection_zero_error_fraction_min_raises_value_error(createPeakCollection) -> None:  # noqa: F811
    """Test that createPeakCollection rejects a non-positive error_fraction_min."""
    with pytest.raises(ValueError, match="invalid error_fraction bounds"):
        createPeakCollection(
            peak_tag="t",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.0,
            projectfilename="/does/not/exist.h5",
            runnumber=1,
            N_subrun=1,
            error_fraction_min=0.0,
            error_fraction_max=0.05,
        )
