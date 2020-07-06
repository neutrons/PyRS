# Standard and third party libraries
from collections import namedtuple
import numpy as np
import pytest
# PyRs libraries
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.fields import (aggregate_scalar_field_samples, fuse_scalar_field_samples, ScalarFieldSample,
                                     StrainField, stack_scalar_field_samples)
from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.peaks import PeakCollection  # type: ignore
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore

SampleMock = namedtuple('SampleMock', 'name values errors x y z')


class TestScalarFieldSample:

    sample1 = SampleMock('lattice',
                         [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090],  # values
                         [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                         [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000],  # x
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                         )

    # The last three points of sample1 overlaps with the first three points of sample1
    sample2 = SampleMock('lattice',
                         [1.071, 1.081, 1.091, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],  # values
                         [0.008, 0.008, 0.008, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # errors
                         [7.009, 8.001, 9.005, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00],  # x
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                         )

    sample3 = SampleMock('strain',
                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # values
                         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # errors
                         [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],  # x
                         [1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5],  # y
                         [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5],  # z
                         )

    sample4 = SampleMock('strain',
                         [float('nan'), 0.1, 0.2, float('nan'), float('nan'), 0.5, 0.6, float('nan')],  # values
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # errors
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # x
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # y
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # z
                         )

    # Points 2 and 3 overlap
    sample5 = SampleMock('lattice',
                         [float('nan'), 1.0, 1.0, 2.0, 3.0, float('nan'), 0.1, 1.1, 2.1, 3.1, 4.1],
                         [0.0, 0.10, 0.11, 0.2, 0.3, 0.4, 0.1, 0.1, 0.2, 0.3, 0.4],
                         [0.0, 1.000, 1.001, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],  # x
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # y
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # z
                         )

    def test_init(self):
        assert ScalarFieldSample(*TestScalarFieldSample.sample1)
        assert ScalarFieldSample(*TestScalarFieldSample.sample2)
        assert ScalarFieldSample(*TestScalarFieldSample.sample3)
        sample_bad = list(TestScalarFieldSample.sample1)
        sample_bad[1] = [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080]  # one less value
        with pytest.raises(AssertionError):
            ScalarFieldSample(*sample_bad)

    def test_len(self):
        assert len(ScalarFieldSample(*TestScalarFieldSample.sample1)) == 10

    def test_values(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.values, TestScalarFieldSample.sample1.values)

    def test_errors(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.errors, TestScalarFieldSample.sample1.errors)

    def test_point_list(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.point_list.vx, TestScalarFieldSample.sample1.x)

    def test_coordinates(self):
        sample = list(TestScalarFieldSample.sample1)
        field = ScalarFieldSample(*sample)
        np.testing.assert_equal(field.coordinates, np.array(sample[3:]).transpose())

    def test_x(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.x, TestScalarFieldSample.sample1.x)

    def test_y(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.y, TestScalarFieldSample.sample1.y)

    def test_z(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        np.testing.assert_equal(field.z, TestScalarFieldSample.sample1.z)

    def test_isfinite(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample4).isfinite
        for attribute in ('values', 'errors', 'x', 'y', 'z'):
            assert getattr(field, attribute) == pytest.approx([0.1, 0.2, 0.5, 0.6])

    def test_interpolated_sample(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample5)
        interpolated = field.interpolated_sample(method='nearest', keep_nan=True,
                                                 resolution=0.01, criterion='min_error')
        np.testing.assert_equal(np.where(np.isnan(interpolated.values))[0], [0, 4])
        np.testing.assert_equal(interpolated.coordinates[4], [4.0, 0.0, 0.0])
        assert interpolated.values[9] == pytest.approx(2.0)
        np.testing.assert_equal(interpolated.coordinates[9], [4.0, 1.0, 0.0])
        # TODO interpolation using method='linear'

    def test_extract(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample1)
        target_indexes = range(0, 10, 2)
        selection = field.extract(target_indexes)
        assert selection.name == 'lattice'
        np.testing.assert_equal(selection.values, [1.000, 1.020, 1.040, 1.060, 1.080])
        np.testing.assert_equal(selection.errors, [0.000, 0.002, 0.004, 0.006, 0.008])
        np.testing.assert_equal(selection.x, [0.000, 2.000, 4.000, 6.000, 8.000])

    def test_aggregate(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample2 = ScalarFieldSample(*TestScalarFieldSample.sample2)
        sample = sample1.aggregate(sample2)
        # index 9 of aggregate sample corresponds to the last point of sample1
        # index 10 of aggregate sample corresponds to the first point of sample2
        np.testing.assert_equal(sample.values[9: 11], [1.090, 1.071])
        np.testing.assert_equal(sample.errors[9: 11], [0.009, 0.008])
        np.testing.assert_equal(sample.x[9: 11], [9.000, 7.009])

    def test_intersection(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.intersection(ScalarFieldSample(*TestScalarFieldSample.sample2))
        assert len(sample) == 6  # three points from sample1 and three points from sample2
        assert sample.name == 'lattice'
        np.testing.assert_equal(sample.values, [1.070, 1.080, 1.090, 1.071, 1.081, 1.091])
        np.testing.assert_equal(sample.errors, [0.007, 0.008, 0.009, 0.008, 0.008, 0.008])
        np.testing.assert_equal(sample.x, [7.000, 8.000, 9.000, 7.009, 8.001, 9.005])

    def test_coalesce(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.aggregate(ScalarFieldSample(*TestScalarFieldSample.sample2))
        sample = sample.coalesce(criterion='min_error')
        assert len(sample) == 17  # discard the last point from sample1 and the first two points from sample2
        assert sample.name == 'lattice'
        # index 6 of aggregate sample corresponds to index 6 of sample1
        # index 11 of aggregate sample corresponds to index 3 of sample2
        np.testing.assert_equal(sample.values[6: 11], [1.060, 1.070, 1.080, 1.091, 1.10])
        np.testing.assert_equal(sample.errors[6: 11], [0.006, 0.007, 0.008, 0.008, 0.0])
        np.testing.assert_equal(sample.x[6: 11], [6.000, 7.000, 8.000, 9.005, 10.00])

    def test_fuse(self):
        sample1 = ScalarFieldSample(*TestScalarFieldSample.sample1)
        sample = sample1.fuse(ScalarFieldSample(*TestScalarFieldSample.sample2), criterion='min_error')
        assert len(sample) == 17  # discard the last point from sample1 and the first two points from sample2
        assert sample.name == 'lattice'
        # index 6 of aggregate sample corresponds to index 6 of sample1
        # index 11 of aggregate sample corresponds to index 3 of sample2
        np.testing.assert_equal(sample.values[6: 11], [1.060, 1.070, 1.080, 1.091, 1.10])
        np.testing.assert_equal(sample.errors[6: 11], [0.006, 0.007, 0.008, 0.008, 0.0])
        np.testing.assert_equal(sample.x[6: 11], [6.000, 7.000, 8.000, 9.005, 10.00])

    def test_export(self):
        # Create a scalar field
        xyz = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
        xyz = np.vstack(np.meshgrid(*xyz)).reshape(3, -1)  # shape = (3, 1000)
        signal, errors = np.arange(0, 1000, 1, dtype=float), np.zeros(1000, dtype=float)
        sample = ScalarFieldSample('strain', signal, errors, *xyz)

        # Test export to MDHistoWorkspace
        workspace = sample.export(form='MDHistoWorkspace', name='strain1', units='mm')
        assert workspace.name() == 'strain1'

        # Test export to CSV file
        with pytest.raises(NotImplementedError):
            sample.export(form='CSV', file='/tmp/csv.txt')

    def test_to_md(self):
        field = ScalarFieldSample(*TestScalarFieldSample.sample3)
        assert field
        histo = field.to_md_histo_workspace('sample3', interpolate=False)
        assert histo
        assert histo.id() == 'MDHistoWorkspace'

        for i, (min_value, max_value) in enumerate(((0.0, 0.5), (1.0, 1.5), (2.0, 2.5))):
            dimension = histo.getDimension(i)
            assert dimension.getUnits() == 'meter'
            # adding half a bin each direction since values from mdhisto are boundaries and constructor uses centers
            assert dimension.getMinimum() == min_value - .25
            assert dimension.getMaximum() == max_value + .25
            assert dimension.getNBins() == 2

        np.testing.assert_equal(histo.getSignalArray().ravel(), self.sample3.values, err_msg='Signal')
        np.testing.assert_equal(histo.getErrorSquaredArray().ravel(), np.square(self.sample3.errors), err_msg='Errors')

        # clean up
        histo.delete()


def test_create_strain_field():
    # 2 points in each direction
    subruns = np.arange(1, 9, dtype=int)

    # create the test peak collection - d-refernce is 1 to make checks easier
    # uncertainties are all zero
    peaks_array = np.zeros(subruns.size, dtype=get_parameter_dtype('gaussian', 'Linear'))
    peaks_array['PeakCentre'][:] = 180.  # position of two-theta in degrees
    peaks_error = np.zeros(subruns.size, dtype=get_parameter_dtype('gaussian', 'Linear'))
    peak_collection = PeakCollection('dummy', 'gaussian', 'linear', wavelength=2.,
                                     d_reference=1., d_reference_error=0.)
    peak_collection.set_peak_fitting_values(subruns, peaks_array, parameter_errors=peaks_error,
                                            fit_costs=np.zeros(subruns.size, dtype=float))

    # create the test workspace - only sample logs are needed
    workspace = HidraWorkspace()
    workspace.set_sub_runs(subruns)
    # arbitray points in space
    workspace.set_sample_log('vx', subruns, np.arange(1, 9, dtype=int))
    workspace.set_sample_log('vy', subruns, np.arange(11, 19, dtype=int))
    workspace.set_sample_log('vz', subruns, np.arange(21, 29, dtype=int))

    # call the function
    strain = StrainField(hidraworkspace=workspace, peak_collection=peak_collection)

    # test the result
    assert strain
    assert len(strain) == subruns.size
    np.testing.assert_almost_equal(strain.values, 0.)
    np.testing.assert_equal(strain.errors, np.zeros(subruns.size, dtype=float))


def test_create_strain_field_from_file_no_peaks():
    # this project file doesn't have peaks in it
    filename = 'tests/data/HB2B_1060_first3_subruns.h5'
    try:
        _ = StrainField(filename)
        assert False, 'Should not be able to read ' + filename
    except IOError:
        pass  # this is what should happen


# 1320 has one peak defined
# 1628 has three peaks defined
@pytest.mark.parametrize('filename,peaknames', [('tests/data/HB2B_1320.h5', ('', 'peak0')),
                                                ('tests/data/HB2B_1628.h5', ('peak0', 'peak1', 'peak2'))])
def test_create_strain_field_from_file(filename, peaknames):
    # directly from a filename
    for tag in peaknames:
        assert StrainField(filename=filename, peak_tag=tag)

    # from a project file
    projectfile = HidraProjectFile(filename, HidraProjectFileMode.READONLY)
    for tag in peaknames:
        assert StrainField(projectfile=projectfile, peak_tag=tag)


@pytest.fixture(scope='module')
def field_sample_collection():
    return {
        'sample1': SampleMock('strain',
                              [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090],  # values
                              [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],  # errors
                              [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
        # distance resolution is assumed to be 0.01
        # The first four points of sample2 still overlaps with the first four points of sample1
        # the last four points of sample1 are not in sample2, and viceversa
        'sample2': SampleMock('strain',
                              [1.071, 1.081, 1.091, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],  # values
                              [0.008, 0.008, 0.008, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # errors
                              [0.009, 1.009, 2.009, 3.009, 4.000, 5.000, 6.011, 7.011, 8.011, 9.011],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
        # the last two points of sample3 are redundant, as they are within resolution distance
        'sample3': SampleMock('strain',
                              [1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060, 1.070, 1.080, 1.090, 1.091],  # values
                              [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.008],  # errors
                              [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 8.991, 9.000],  # x
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # y
                              [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  # z
                              ),
    }


def test_aggregate_scalar_field_samples(field_sample_collection):
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    aggregated_sample = aggregate_scalar_field_samples(sample1, sample2, sample3)
    assert len(aggregated_sample) == len(sample1) + len(sample2) + len(sample3)
    # check some key field values
    assert aggregated_sample.values[len(sample1)] == pytest.approx(sample2.values[0])
    assert aggregated_sample.values[len(sample1) + len(sample2)] == pytest.approx(sample3.values[0])
    assert aggregated_sample.values[-1] == pytest.approx(sample3.values[-1])
    # check some key point coordinate values
    assert aggregated_sample.x[len(sample1)] == pytest.approx(sample2.x[0])
    assert aggregated_sample.x[len(sample1) + len(sample2)] == pytest.approx(sample3.x[0])
    assert aggregated_sample.x[-1] == pytest.approx(sample3.x[-1])


def test_fuse_scalar_field_samples(field_sample_collection):
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    fused_sample = fuse_scalar_field_samples(sample1, sample2, sample3, resolution=0.01, criterion='min_error')
    assert len(fused_sample) == 14
    assert sorted(fused_sample.values) == pytest.approx([1., 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.08,
                                                         1.091, 1.1, 1.13, 1.14, 1.15, 1.16])
    assert sorted(fused_sample.errors) == pytest.approx([0., 0., 0.001, 0.002, 0.004, 0.005, 0.006,
                                                         0.007, 0.008, 0.008, 0.03, 0.04, 0.05, 0.06])
    assert sorted(fused_sample.x) == pytest.approx([0.0, 1.0, 2.0, 3.009, 4.0, 5.0, 6.0, 6.011,
                                                    7.0, 7.011, 8.0, 8.011, 9.0, 9.011])


def test_stack_scalar_field_samples(field_sample_collection,
                                    approx_with_sorting, assert_almost_equal_with_sorting, allclose_with_sorting):
    r"""Stack three scalar fields"""
    # test stacking with the 'common' mode
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    sample1, sample2, sample3 = stack_scalar_field_samples(sample1, sample2, sample3, stack_mode='common')

    for sample in (sample1, sample2, sample3):
        assert len(sample) == 6
        assert_almost_equal_with_sorting(sample.x, [5.0, 4.0, 3.003, 2.003, 1.003, 0.003])

    # Assert evaluations for sample1
    sample1_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0]
    assert np.allclose(sample1.values, sample1_values)
    # Assert evaluations for sample2
    sample2_values = [1.12, 1.11, 1.1, 1.091, 1.081, 1.071]
    assert np.allclose(sample2.values, sample2_values)
    # Assert evaluations for sample3
    sample3_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0]
    assert np.allclose(sample3.values, sample3_values)

    # test stacking with the 'complete' mode
    sample1 = ScalarFieldSample(*field_sample_collection['sample1'])
    sample2 = ScalarFieldSample(*field_sample_collection['sample2'])
    sample3 = ScalarFieldSample(*field_sample_collection['sample3'])
    sample1, sample2, sample3 = stack_scalar_field_samples(sample1, sample2, sample3, stack_mode='complete')

    for sample in (sample1, sample2, sample3):
        assert len(sample) == 14
        approx_with_sorting(sample.x,
                            [5.0, 4.0, 3.003, 2.003, 1.003, 0.003, 9.0, 8.0, 6.0, 7.0, 9.011, 8.011, 6.011, 7.011])

    # Assert evaluations for sample1
    sample1_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0,
                      1.09, 1.08, 1.06, 1.07,
                      float('nan'), float('nan'), float('nan'), float('nan')]
    assert allclose_with_sorting(sample1.values, sample1_values, equal_nan=True)

    # Assert evaluations for sample2
    sample2_values = [1.12, 1.11, 1.1, 1.091, 1.081, 1.071,
                      float('nan'), float('nan'), float('nan'), float('nan'),
                      1.16, 1.15, 1.13, 1.14]
    assert allclose_with_sorting(sample2.values, sample2_values, equal_nan=True)

    # Assert evaluations for sample3
    sample3_values = [1.05, 1.04, 1.03, 1.02, 1.01, 1.0,
                      1.091, 1.08, 1.06, 1.07,
                      float('nan'), float('nan'), float('nan'), float('nan')]
    assert allclose_with_sorting(sample3.values, sample3_values, equal_nan=True)


if __name__ == '__main__':
    pytest.main()
