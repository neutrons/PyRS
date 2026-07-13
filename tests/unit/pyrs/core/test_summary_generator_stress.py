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


def strain_instantiator(name, values, errors, x, y, z):
    return StrainField(
        name,
        peak_collection=PeakCollectionLite(name, strain=values, strain_error=errors),
        point_list=PointList([x, y, z]),
    )


def test_write_csv_empty_strain_filenames():
    with pytest.raises(RuntimeError) as exception_info:
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
        SummaryGeneratorStress("dummy.csv", stress)
        assert "StrainField filenames in direction " in str(exception_info.value)


def test_write_csv_none_stress():
    with pytest.raises(RuntimeError) as exception_info:
        SummaryGeneratorStress("dummy.csv", None)
        assert "Error: stress input must be of type StressField" in str(exception_info.value)
