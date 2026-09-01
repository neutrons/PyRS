"""
Integration tests for pyrs.dataobjects.fields — split out of test_fields.py.

These tests all depend on real HB2B project files via `test_data_dir` (directly, or
via the `strain_field_samples` fixture below). See tests/unit/pyrs/dataobjects/test_fields.py
for the in-memory unit tests of the same classes.
"""

# Standard and third party libraries
from collections.abc import Callable
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
def strain_field_samples(test_data_dir: str) -> dict[str, StrainFieldSingle]:
    r"""Build a set of named `StrainFieldSingle` samples, from mock and real data.

    Args:
        test_data_dir: Path to the `tests/data` directory (session fixture).

    Returns:
        A dict mapping a descriptive sample name to a `StrainFieldSingle` instance.
        Keys used elsewhere in this module:
            - "strain with two points per direction": a synthetic sample built from
              an in-memory `HidraWorkspace` (no file I/O), 8 sub-runs.
            - "HB2B_1320_peak0": read from `HB2B_1320.h5` with peak tag "peak0".
            - "HB2B_1320_": read from `HB2B_1320.h5` with the default (empty) peak tag.
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

    strain = StrainFieldSingle(hidraworkspace=workspace, peak_collection=peak_collection)

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
    """Tests for `StrainFieldSingle` backed by real and synthetic HB2B data."""

    @pytest.mark.integration
    def test_get_effective_peak_parameter_invalid_name_raises_value_error(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `get_effective_peak_parameter` raises `ValueError` for an unknown parameter name."""
        strain = strain_field_samples["strain with two points per direction"]  # mock object

        with pytest.raises(ValueError) as exception_info:
            strain.get_effective_peak_parameter("impossible")
        assert "impossible" in str(exception_info.value)

    @pytest.mark.integration
    def test_get_effective_peak_parameter_supported_name_returns_scalar_field(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `get_effective_peak_parameter` returns a per-point field for every known parameter name."""
        strain = strain_field_samples["strain with two points per direction"]  # mock object

        num_values = len(strain)
        for name in EFFECTIVE_PEAK_PARAMETERS:
            scalar_field = strain.get_effective_peak_parameter(name)
            assert scalar_field, f"Failed to get {name}"
            assert len(scalar_field) == num_values, f"{name} does not have correct length"


class Test_StrainField:
    """Tests for `StrainField` equality, using a minimal mock subclass."""

    class StrainFieldMock(StrainField):
        r"""Mocks a StrainField object overloading the initialization"""

        def __init__(self, *strains: StrainFieldSingle) -> None:
            """Store `strains` directly, bypassing `StrainField`'s normal file/fuse-based init."""
            for s in strains:
                assert isinstance(s, StrainFieldSingle)
            self._strains = list(strains)

    @pytest.mark.integration
    def test_eq_matching_and_differing_strains_returns_expected_bool(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test `==` for both single-scan `StrainFieldSingle` and multi-scan `StrainField` objects."""
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
    """Tests for `StrainField` fuse/stack/export behavior, backed by real HB2B project files."""

    @pytest.mark.integration
    def test_peak_collections_property_returns_single_element_list(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `peak_collections` returns a one-element list of `PeakCollection`."""
        strain = strain_field_samples["strain with two points per direction"]
        assert isinstance(strain.peak_collections, list)
        assert len(strain.peak_collections) == 1
        assert isinstance(strain.peak_collections[0], PeakCollection)

    @pytest.mark.integration
    def test_coordinates_property_matches_sample_log_positions(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `coordinates` matches the (vx, vy, vz) sample logs used to build the strain."""
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
    def test_fuse_with_two_strains_returns_merged_strain(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `fuse_with` concatenates peak collections and coordinates from both strains."""
        strain1 = strain_field_samples["HB2B_1320_peak0"]
        strain2 = strain_field_samples["strain with two points per direction"]
        strain = strain1.fuse_with(strain2)
        assert strain.peak_collections == [strain1.peak_collections[0], strain2.peak_collections[0]]
        assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))
        assert strain.field  # should return something

        # fusing a scan with itself creates a new copy of the strain
        assert strain.peak_collections[0] == strain1.peak_collections[0]

    @pytest.mark.integration
    def test_fuse_with_invalid_criterion_raises_value_error(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `fuse_with` rejects an unrecognized `criterion` value."""
        strain1 = strain_field_samples["HB2B_1320_peak0"]
        strain2 = strain_field_samples["strain with two points per direction"]
        with pytest.raises(ValueError, match="Unallowed value of criterion"):
            strain1.fuse_with(strain2, criterion="bogus")

    @pytest.mark.integration
    def test_add_operator_two_strains_returns_merged_strain(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `+` is equivalent to `fuse_with` for two single-scan strains."""
        strain1 = strain_field_samples["HB2B_1320_peak0"]
        strain2 = strain_field_samples["strain with two points per direction"]
        strain = strain1 + strain2
        assert strain.peak_collections == [strain1.peak_collections[0], strain2.peak_collections[0]]
        assert np.allclose(strain.coordinates, np.concatenate((strain1.coordinates, strain2.coordinates)))

    @pytest.mark.integration
    def test_strain_field_init_file_without_peaks_raises_io_error(self, test_data_dir: str) -> None:
        """Test that loading a project file with no fitted peaks raises `IOError`."""
        # this project file doesn't have peaks in it
        file_path = os.path.join(test_data_dir, "HB2B_1060_first3_subruns.h5")
        with pytest.raises(IOError):
            StrainField(file_path)

    @pytest.mark.integration
    def test_strain_field_init_from_file_returns_populated_field(self, test_data_dir: str) -> None:
        """Test that a `StrainField` can be built directly from a project file and peak tag."""
        file_path = os.path.join(test_data_dir, "HB2B_1320.h5")
        strain = StrainField(filename=file_path, peak_tag="peak0")

        assert strain
        assert strain.field
        assert strain.get_effective_peak_parameter("Center")

    @pytest.mark.integration
    def test_fuse_strains_classmethod_matches_sequential_addition(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `StrainField.fuse_strains` matches summing the same strains with `+`."""
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
    def test_stack_operator_overlapping_and_disjoint_strains_returns_expected_fields(
        self, strain_field_samples: dict[str, StrainFieldSingle], allclose_with_sorting: Callable[..., bool]
    ) -> None:
        """Test `*` (stacking) for strains with overlapping, and with disjoint, evaluation points."""
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
        for strain_stacked, original_strain in ((strain2_stacked, strain2), (strain3_stacked, strain3)):
            assert strain_stacked.filenames == original_strain.filenames

        # There's no common point that is common to both strain2 and strain3
        # Each stacked strain only have finite measurements on points coming from the un-stacked strain
        for strain_stacked, original_strain in ((strain2_stacked, strain2), (strain3_stacked, strain3)):
            finite_measurements_count = len(np.where(np.isfinite(strain_stacked.values))[0])
            assert finite_measurements_count == len(original_strain)

        # The points evaluated as 'nan' must come from the other scan
        for strain_stacked, strain_other in ((strain2_stacked, strain3), (strain3_stacked, strain2)):
            nan_measurements_count = len(np.where(np.isnan(strain_stacked.values))[0])
            assert nan_measurements_count == len(strain_other)

    @pytest.mark.integration
    def test_stack_strains_unimplemented_mode_raises_not_implemented_error(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `StrainField.stack_strains` rejects the recognized-but-not-yet-implemented
        `stack_mode="intersection"` value.

        Requires two strains with genuinely different point lists: `stack_strains` has
        a "trivial case" early return when all input point lists are already equal,
        which would otherwise skip the mode check entirely before it's ever reached.
        `*` (`__mul__`) always hardcodes `mode="union"`, so this path can only be
        reached by calling the classmethod directly. A wholly unrecognized `stack_mode`
        string (e.g. `"bogus"`) instead raises `ValueError` at an earlier validation
        step -- a different, narrower error than this one.
        """
        strain1 = strain_field_samples["HB2B_1320_peak0"]
        strain3 = strain_field_samples["strain with two points per direction"]
        with pytest.raises(NotImplementedError, match="is not currently supported"):
            StrainField.stack_strains(strain1, strain3, stack_mode="intersection")

    @pytest.mark.integration
    def test_fuse_then_stack_strains_matches_expected_finite_and_nan_counts(
        self, strain_field_samples: dict[str, StrainFieldSingle], allclose_with_sorting: Callable[..., bool]
    ) -> None:
        """Test that stacking a strain against a fused pair matches fusing then stacking, point-for-point."""
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
    def test_to_md_histo_workspace_returns_expected_bin_geometry(
        self, strain_field_samples: dict[str, StrainFieldSingle]
    ) -> None:
        """Test that `to_md_histo_workspace` produces an MDHistoWorkspace with the expected bin geometry."""
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
def test_stress_field_from_identical_strains_computes_expected_values(test_data_dir: str) -> None:
    """Test `StressField` computed from three identical strains loaded from a real project file."""
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
        assert stress.strain is not None  # guaranteed non-None immediately after select()
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

    assert stress.stress11 is not None  # populated by StressField.__init__
    stress11 = stress.stress11.values
    print(stress11)
    assert np.all(np.logical_not(np.isnan(stress11)))  # confirm something was set

    # redo the calculation - this should change nothing
    stress.update_stress_calculation()
    assert stress.stress11 is not None
    np.testing.assert_equal(stress.stress11.values, stress11)

    # set the d-reference and see that the values are changed
    stress.set_d_reference((42.0, 0.0))
    assert stress.stress11 is not None
    assert np.all(stress.stress11.values != stress11)


@pytest.mark.integration
def test_stress_field_select_invalid_direction_raises_value_error(test_data_dir: str) -> None:
    """Test that `StressField.select` rejects an unrecognized direction string."""
    HB2B_1320_PROJECT = os.path.join(test_data_dir, "HB2B_1320.h5")
    YOUNG = 200.0
    POISSON = 0.3

    sample11 = StrainField(HB2B_1320_PROJECT)
    sample22 = StrainField(HB2B_1320_PROJECT)
    sample33 = StrainField(HB2B_1320_PROJECT)
    stress = StressField(sample11, sample22, sample33, YOUNG, POISSON)

    with pytest.raises(ValueError, match="Cannot determine direction type"):
        stress.select("bogus")
