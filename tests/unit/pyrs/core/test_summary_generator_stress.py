"""
Unit tests for pyrs.core.summary_generator_stress — split out of
tests/integration/test_write_stress_csv.py.

These two tests exercise SummaryGeneratorStress's input-validation error paths and
need no real project file. See tests/integration/test_write_stress_csv.py for the
CSV-writing tests that do (gold-file comparisons against real HB2B project files).
"""

import pytest

from pyrs.peaks import PeakCollectionLite  # type: ignore
from pyrs.dataobjects.sample_logs import PointList
from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from pyrs.core.summary_generator_stress import SummaryGeneratorStress


def strain_instantiator(
    name: str,
    values: list[float],
    errors: list[float],
    x: list[float],
    y: list[float],
    z: list[float],
) -> StrainField:
    """Build a minimal `StrainField` directly from strain values, with no backing project file.

    Pairs a `PeakCollectionLite` (strain values/errors only, no full peak-fit profile)
    with a `PointList` built from the given coordinates. Used by this module's
    `SummaryGeneratorStress` input-validation tests, which need a `StrainField` whose
    `filenames` is empty (i.e. not loaded from a real project file).

    Args:
        name: Tag identifying the strain. Passed through as `StrainField`'s `filename`
            argument and `PeakCollectionLite`'s `peak_tag` -- despite the parameter
            name, no file is read; supplying `peak_collection`/`point_list` directly
            means this string is used only as a label.
        values: Strain value at each sample point.
        errors: Strain uncertainty at each sample point, same length as `values`.
        x: X coordinate of each sample point.
        y: Y coordinate of each sample point.
        z: Z coordinate of each sample point.

    Returns:
        A `StrainField` built from the given values, with no backing project file.
    """
    return StrainField(
        name,
        peak_collection=PeakCollectionLite(name, strain=values, strain_error=errors),
        point_list=PointList([x, y, z]),
    )


def test_summary_generator_stress_init_strain_without_filenames_raises_runtime_error() -> None:
    """Test that `SummaryGeneratorStress` rejects strains with no `filenames`.

    Strains built directly from values via `strain_instantiator` (not loaded from a
    real project file) have empty `filenames`; the "11"/"22" directions require a
    non-empty `filenames`, so construction must raise `RuntimeError` naming the
    affected direction.
    """
    # strain that doesn't come from a project file
    X = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
    Y = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    Z = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]

    strain11 = strain_instantiator(
        "strain",
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
        X,
        Y,
        Z,
    )
    strain22 = strain_instantiator(
        "strain",
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
        X,
        Y,
        Z,
    )
    strain33 = strain_instantiator(
        "strain",
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
        [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
        X,
        Y,
        Z,
    )

    stress = StressField(strain11, strain22, strain33, 200, 0.3)
    with pytest.raises(RuntimeError) as exception_info:
        SummaryGeneratorStress("dummy.csv", stress)
    assert "StrainField filenames in direction " in str(exception_info.value)


def test_summary_generator_stress_init_none_stress_raises_runtime_error() -> None:
    """Test that `SummaryGeneratorStress` rejects `stress_input=None`.

    `None` is neither a `StressField` nor a `StressFacade`, so construction must raise
    `RuntimeError`.
    """
    with pytest.raises(RuntimeError) as exception_info:
        SummaryGeneratorStress("dummy.csv", None)
    assert "stress input must be of type StressField or StressFacade" in str(exception_info.value)
