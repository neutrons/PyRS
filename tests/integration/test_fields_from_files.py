"""
Integration tests for pyrs.dataobjects.fields — split out of test_fields.py.

These tests all depend on real HB2B project files via `test_data_dir` (directly, or
via the `strain_field_samples` fixture below). See tests/unit/pyrs/dataobjects/test_fields.py
for the in-memory unit tests of the same classes.
"""

# Standard and third party libraries
import copy
import numpy as np
import os
import pytest

# PyRs libraries
from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS, get_parameter_dtype
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import DEFAULT_POINT_RESOLUTION
from pyrs.dataobjects.fields import (
    StrainField,
    StrainFieldSingle,
    StressField,
)
from pyrs.peaks import PeakCollection  # type: ignore
from tests.conftest import assert_allclose_with_sorting

to_megapascal = StressField.to_megapascal

DIRECTIONS = ("11", "22", "33")  # directions for the StrainField


@pytest.fixture(scope="module")
def strain_field_samples(test_data_dir):
    r"""
    A number of StrainField objects from mock and real data
    """
    sample_fields = {}
    #####
    # The first sample has 2 points in each direction
    #####
    subruns = np.arange(1, 9, dtype=int)

    # create the test peak collection - d-refernce is 1 to make checks easier
    # uncertainties are all zero
    peaks_array = np.zeros(subruns.size, dtype=get_parameter_dtype("gaussian", "Linear"))
    peaks_array["PeakCentre"][:] = 180.0  # position of two-theta in degrees
    peaks_array["Height"][:] = 1.0  # position of two-theta in degrees
    peaks_error = np.zeros(subruns.size, dtype=get_parameter_dtype("gaussian", "Linear"))
    peak_collection = PeakCollection(
        "dummy", "gaussian", "linear", wavelength=2.0, d_reference=1.0, d_reference_error=0.0
    )
    peak_collection.set_peak_fitting_values(
        subruns, peaks_array, parameter_errors=peaks_error, fit_costs=np.zeros(subruns.size, dtype=float)
    )

    # create the test workspace - only sample logs are needed
    workspace = HidraWorkspace()
    workspace.set_sub_runs(subruns)
    # arbitray points in space
    workspace.set_sample_log("vx", subruns, np.arange(1, 9, dtype=int))
    workspace.set_sample_log("vy", subruns, np.arange(11, 19, dtype=int))
    workspace.set_sample_log("vz", subruns, np.arange(21, 29, dtype=int))

    # call the function
    strain = StrainFieldSingle(hidraworkspace=workspace, peak_collection=peak_collection)

    # test the result
    assert strain
    assert not strain.filenames
    assert len(strain) == subruns.size
    assert strain.peak_collections == [peak_collection]
    np.testing.assert_almost_equal(strain.values, 0.0)
    np.testing.assert_equal(strain.errors, np.zeros(subruns.size, dtype=float))
    sample_fields["strain with two points per direction"] = strain

    #####
    # Create StrainField samples from two files and different peak tags
    #####
    # TODO: substitute/fix HB2B_1628.h5 with other data, because reported vx, vy, and vz are all 'nan'
    # filename_tags_pairs = [('HB2B_1320.h5', ('', 'peak0')), ('HB2B_1628.h5', ('peak0', 'peak1', 'peak2'))]
    filename_tags_pairs = [("HB2B_1320.h5", ("", "peak0"))]
    for filename, tags in filename_tags_pairs:
        file_path = os.path.join(test_data_dir, filename)
        prefix = filename.split(".")[0] + "_"
        for tag in tags:
            sample_fields[prefix + tag] = StrainFieldSingle(filename=file_path, peak_tag=tag)
            assert sample_fields[prefix + tag].filenames == [filename]

    return sample_fields


class TestStrainFieldSingle:
    @pytest.mark.integration
    def test_get_peak_params(self, strain_field_samples):
            strain = strain_field_samples["strain with two points per direction"]  # mock object

            # test that getting non-existant parameter works
            with pytest.raises(ValueError) as exception_info:
                strain.get_effective_peak_parameter("impossible")
            assert "impossible" in str(exception_info.value)

            num_values = len(strain)
            for name in EFFECTIVE_PEAK_PARAMETERS:
                scalar_field = strain.get_effective_peak_parameter(name)
                assert scalar_field, f"Failed to get {name}"
                assert len(scalar_field) == num_values, f"{name} does not have correct length"


class Test_StrainField:
    class StrainFieldMock(StrainField):
            r"""Mocks a StrainField object overloading the initialization"""

            def __init__(self, *strains):
                for s in strains:
                    assert isinstance(s, StrainFieldSingle)
                self._strains = strains

    @pytest.mark.integration
    def test_eq(self, strain_field_samples):
            strains_single_scan = copy.deepcopy(list(strain_field_samples.values()))
            # single-scan strains
            assert strains_single_scan[0] == strains_single_scan[0]
            assert (strains_single_scan[0] == strains_single_scan[1]) is False

            # multi-scan scans
            strain_multi = self.StrainFieldMock(*strains_single_scan)
            assert strain_multi == strain_multi
            assert (strain_multi == strains_single_scan[0]) is False
            strain_multi_2 = self.StrainFieldMock(*strains_single_scan[:-1])  # all except the last one
            assert (strain_multi == strain_multi_2) is False


class TestStrainField:
    @pytest.mark.integration
    def test_peak_collection(self, strain_field_samples):
            strain = strain_field_samples["strain with two points per direction"]
            assert isinstance(strain.peak_collections, list)
            assert len(strain.peak_collections) == 1
            assert isinstance(strain.peak_collections[0], PeakCollection)

    @pytest.mark.integration
    def test_peak_collections(self, strain_field_samples):
            strain = strain_field_samples["strain with two points per direction"]
            assert len(strain.peak_collections) == 1
            assert isinstance(strain.peak_collections[0], PeakCollection)

    @pytest.mark.integration
    def test_coordinates(self, strain_field_samples):
            strain = strain_field_samples["strain with two points per direction"]
            coordinates = np.array(
                [
                    [1.0, 11.0, 21.0],
                    [2.0, 12.0, 22.0],
                    [3.0, 13.0, 23.0],
                    [4.0, 14.0, 24.0],
                    [5.0, 15.0, 25.0],
                    [6.0, 16.0, 26.0],
                    [7.0, 17.0, 27.0],
                    [8.0, 18.0, 28.0],
                ]
            )
            assert np.allclose(strain.coordinates, coordinates)

    @pytest.mark.integration
    def test_fuse_with(self, strain_field_samples):
            strain1 = strain_field_samples["HB2B_1320_peak0"]
            strain2 = strain_field_samples["strain with two points per direction"]
            strain = strain1.fuse_with(strain2)
            assert strain.peak_collections == [strain1.peak_collections[0], strain2.peak_collections[0]]
            assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))
            assert strain.field  # should return something

            # fusing a scan with itself creates a new copy of the strain
            assert strain.peak_collections[0] == strain1.peak_collections[0]

    @pytest.mark.integration
    def test_add(self, strain_field_samples):
            strain1 = strain_field_samples["HB2B_1320_peak0"]
            strain2 = strain_field_samples["strain with two points per direction"]
            strain = strain1 + strain2
            assert strain.peak_collections == [strain1.peak_collections[0], strain2.peak_collections[0]]
            assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))

    @pytest.mark.integration
    def test_create_strain_field_from_file_no_peaks(self, test_data_dir):
            # this project file doesn't have peaks in it
            file_path = os.path.join(test_data_dir, "HB2B_1060_first3_subruns.h5")
            try:
                _ = StrainField(file_path)  # noqa F841
                assert False, "Should not be able to read " + file_path
            except IOError:
                pass

    @pytest.mark.integration
    def test_from_file(self, test_data_dir):
            file_path = os.path.join(test_data_dir, "HB2B_1320.h5")
            strain = StrainField(filename=file_path, peak_tag="peak0")

            assert strain
            assert strain.field
            assert strain.get_effective_peak_parameter("Center")

    @pytest.mark.integration
    def test_fuse_strains(self, strain_field_samples):
            # TODO HB2B_1320_peak0 and HB2B_1320_ are the same scan. We need two different scans
            strain1 = strain_field_samples["HB2B_1320_peak0"]
            strain2 = strain_field_samples["HB2B_1320_"]
            strain3 = strain_field_samples["strain with two points per direction"]
            # Use fuse_strains().
            strain_fused = StrainField.fuse_strains(
                strain1, strain2, strain3, resolution=DEFAULT_POINT_RESOLUTION, criterion="min_error"
            )
            # the sum should give the same, since we passed default resolution and criterion options
            strain_sum = strain1 + strain2 + strain3
            for strain in (strain_fused, strain_sum):
                assert len(strain) == 312 + 8  # strain1 and strain2 give strain1 because they contain the same data
                assert strain.peak_collections == [s.peak_collections[0] for s in (strain1, strain2, strain3)]
                values = np.concatenate((strain1.values, strain3.values))  # no strain2 because it's the same as strain1
                assert_allclose_with_sorting(strain.values, values)

    @pytest.mark.integration
    def test_stack_strains(self, strain_field_samples, allclose_with_sorting):
            strain1 = strain_field_samples["HB2B_1320_peak0"]
            strain2 = strain_field_samples["HB2B_1320_"]

            # Stack two strains having the same evaluation points.
            strain1_stacked, strain2_stacked = strain1 * strain2  # default resolution and stacking mode
            for strain in (strain1_stacked, strain2_stacked):
                assert len(strain) == len(strain1)
                assert bool(np.all(np.isfinite(strain.values))) is True  # all points are common to strain1 and strain2

            # Stack two strains having completely different evaluation points.
            strain3 = strain_field_samples["strain with two points per direction"]
            strain2_stacked, strain3_stacked = strain2 * strain3  # default resolution and stacking mode
            # The common list of points is the sum of the points from each strain
            for strain in (strain2_stacked, strain3_stacked):
                assert len(strain) == len(strain2) + len(strain3)

            # verify the filenames got copied over
            for strain_stacked, strain in ((strain2_stacked, strain2), (strain3_stacked, strain3)):
                assert strain_stacked.filenames == strain.filenames

            # There's no common point that is common to both strain2 and strain3
            # Each stacked strain only have finite measurements on points coming from the un-stacked strain
            for strain_stacked, strain in ((strain2_stacked, strain2), (strain3_stacked, strain3)):
                finite_measurements_count = len(np.where(np.isfinite(strain_stacked.values))[0])
                assert finite_measurements_count == len(strain)

            # The points evaluated as 'nan' must come from the other scan
            for strain_stacked, strain_other in ((strain2_stacked, strain3), (strain3_stacked, strain2)):
                nan_measurements_count = len(np.where(np.isnan(strain_stacked.values))[0])
                assert nan_measurements_count == len(strain_other)

    @pytest.mark.integration
    def test_fuse_and_stack_strains(self, strain_field_samples, allclose_with_sorting):
            # TODO HB2B_1320_peak0 and HB2B_1320_ are the same scan. We need two different scans
            strain1 = strain_field_samples["HB2B_1320_peak0"]
            strain2 = strain_field_samples["HB2B_1320_"]
            strain3 = strain_field_samples["strain with two points per direction"]
            strain1_stacked, strain23_stacked = strain1 * (strain2 + strain3)  # default resolution and stacking mode
            # Check number of points with finite strains measuments
            for strain_stacked in (strain1_stacked, strain23_stacked):
                assert len(strain_stacked) == len(strain2) + len(strain3)
            for strain_stacked, finite_count, nan_count in zip((strain1_stacked, strain23_stacked), (312, 320), (8, 0)):
                finite_measurements_count = len(np.where(np.isfinite(strain_stacked.values))[0])
                assert finite_measurements_count == finite_count
                nan_measurements_count = len(np.where(np.isnan(strain_stacked.values))[0])
                assert nan_measurements_count == nan_count
            # Check peak collections carry-over
            assert strain1_stacked.peak_collections[0] == strain1.peak_collections[0]
            assert strain23_stacked.peak_collections == [strain2.peak_collections[0], strain3.peak_collections[0]]

    @pytest.mark.integration
    def test_to_md_histo_workspace(self, strain_field_samples):
            strain = strain_field_samples["HB2B_1320_peak0"]
            histo = strain.to_md_histo_workspace(method="linear", resolution=DEFAULT_POINT_RESOLUTION)
            assert histo.id() == "MDHistoWorkspace"
            minimum_values = (-31.76, -7.20, -15.00)  # bin boundary with the smallest coordinate along X, Y, and Z
            maximum_values = (31.76, 7.20, 15.00)  # bin boundary with the largest coordinate along X, Y, and Z
            bin_counts = (18, 6, 3)  # number of bins along  X, Y, and Z
            for i, (min_value, max_value, bin_count) in enumerate(zip(minimum_values, maximum_values, bin_counts)):
                dimension = histo.getDimension(i)
                assert dimension.getUnits() == "mm"
                assert dimension.getMinimum() == pytest.approx(min_value, abs=0.01)
                assert dimension.getMaximum() == pytest.approx(max_value, abs=0.01)
                assert dimension.getNBins() == bin_count


@pytest.mark.integration
def test_stress_field_from_files(test_data_dir):
    HB2B_1320_PROJECT = os.path.join(test_data_dir, "HB2B_1320.h5")
    YOUNG = 200.0
    POISSON = 0.3

    # create 3 strain objects
    sample11 = StrainField(HB2B_1320_PROJECT)
    sample22 = StrainField(HB2B_1320_PROJECT)
    sample33 = StrainField(HB2B_1320_PROJECT)
    # create the stress field (with very uninteresting values
    stress = StressField(sample11, sample22, sample33, YOUNG, POISSON)

    # confirm the strains are unchanged
    for direction in DIRECTIONS:
        stress.select(direction)
        np.testing.assert_allclose(
            stress.strain.values,
            sample11.peak_collections[0].get_strain(units="microstrain")[0],
            atol=1,
            err_msg=f"strain direction {direction}",
        )

    # calculate the values for stress
    strains = sample11.peak_collections[0].get_strain(units="microstrain")[0]
    stress_exp = strains + POISSON * (strains + strains + strains) / (1.0 - 2.0 * POISSON)
    stress_exp *= YOUNG / (1.0 + POISSON)

    # since all of the contributing strains are identical, everything else should match
    for direction in DIRECTIONS:
        stress.select(direction)
        np.testing.assert_equal(stress.point_list.coordinates, sample11.point_list.coordinates)
        np.testing.assert_allclose(
            stress.values, to_megapascal(stress_exp), atol=1, err_msg=f"stress direction {direction}"
        )

    stress11 = stress.stress11.values
    print(stress11)
    assert np.all(np.logical_not(np.isnan(stress11)))  # confirm something was set

    # redo the calculation - this should change nothing
    stress.update_stress_calculation()
    np.testing.assert_equal(stress.stress11.values, stress11)

    # set the d-reference and see that the values are changed
    stress.set_d_reference((42.0, 0.0))
    assert np.all(stress.stress11.values != stress11)
