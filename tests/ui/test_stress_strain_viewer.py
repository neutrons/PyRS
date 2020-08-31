# This is a test of the model component of the strain/stress viewer
from pyrs.interface.strainstressviewer.model import Model
import numpy as np
import os
import pytest


def test_model(tmpdir, test_data_dir):
    model = Model()

    assert model.selectedPeak is None
    assert model.e11 == []
    assert model.e22 == []
    assert model.e33 == []
    assert model._stress is None

    assert model.validate_selection('11') == "e11 file hasn't been loaded"
    assert model.validate_selection('22') == "e22 file hasn't been loaded"
    assert model.validate_selection('33') == "e33 file hasn't been loaded"

    # load file without fitted peaks
    model.e11 = os.path.join(test_data_dir, 'HB2B_1423.h5')
    assert model.validate_selection('11') == "e11 contains no peaks, fit peaks first"

    # load file with fitted peaks
    model.e11 = os.path.join(test_data_dir, 'HB2B_1320.h5')
    assert len(model.e11) == 1
    assert model.e11[0].name == '11'
    assert model.peakTags == ['peak0']
    assert len(model.e11_peaks) == 1
    assert 'peak0' in model.e11_peaks[0]

    # select non-existing peak
    model.selectedPeak = 'peak_label'
    assert model.selectedPeak == 'peak_label'
    assert model.validate_selection('11') == "Peak peak_label is not in e11"

    # select existing peak
    model.selectedPeak = 'peak0'
    assert model.selectedPeak == 'peak0'

    assert model.validate_selection('11') is None
    assert model.validate_selection('22') == "e22 file hasn't been loaded"
    assert model.validate_selection('33') == "e33 file hasn't been loaded"

    assert model.d0 is None

    for plot_param in ("dspacing-center",
                       "d-reference",
                       "Center",
                       "Height",
                       "FWHM",
                       "Mixing",
                       "Intensity",
                       "strain"):
        e11_md = model.get_field('11', plot_param, 'stress_case').to_md_histo_workspace()
        assert e11_md.name() == f'{plot_param}'
        assert e11_md.getNumDims() == 3
        assert [e11_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]
        assert model.get_field('22', plot_param, 'stress_case') is None
        assert model.get_field('33', plot_param, 'stress_case') is None

    # Need to load ε22 so it should fail
    with pytest.raises(TypeError) as exception_info:
        model.calculate_stress('in-plane-stress', 200, 0.3, (1.08, 0))
    assert 'None is not a _StrainField object' in str(exception_info.value)

    model.e22 = os.path.join(test_data_dir, 'HB2B_1320.h5')
    assert len(model.e22) == 1
    assert model.e22[0].name == '22'

    model.calculate_stress('in-plane-stress', 200, 0.3, (1.08, 0))

    assert model.d0.values[0] == 1.08

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'In-plane stress').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Stress should be all zero for in-plane stress case
    assert np.count_nonzero(model.get_field('33', 'stress', 'In-plane stress')
                            .to_md_histo_workspace().getSignalArray()) == 0
    # Strain should be all non-zero for in-plane stress case
    assert np.count_nonzero(model.get_field('33', 'strain', 'In-plane stress')
                            .to_md_histo_workspace().getSignalArray()) == 18*6*3

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1320_1320_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv1.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318

    # Check writing with bad filename, should fail but should emit failure message
    model.write_stress_to_csv("/bin/false")

    model.calculate_stress('in-plane-strain', 200, 0.3, (1.08, 0))

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'In-plane strain').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Stress should be all non-zero for in-plane strain case
    assert np.count_nonzero(model.get_field('33', 'stress', 'In-plane strain')
                            .to_md_histo_workspace().getSignalArray()) == 18*6*3
    # Strain shouldn't exist for for in-plane strain case
    assert model.get_field('33', 'strain', 'In-plane strain') is None

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1320_1320_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv2.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318

    model.e33 = os.path.join(test_data_dir, 'HB2B_1320.h5')
    assert len(model.e33) == 1
    assert model.e33[0].name == '33'

    model.calculate_stress('diagonal', 200, 0.3, (1, 0))

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'diagonal').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Should be all non-zero for diagonal stress case
    assert np.count_nonzero(model.get_field('33', 'stress', 'diagonal')
                            .to_md_histo_workspace().getSignalArray()) == 18*6*3

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1320_1320_1320_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv3.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318

    # Check rerunning calculate_stress while change and not change
    # parameter, it should only recalculate when changed
    current_stress = model._stress
    current_stress11_values = current_stress.stress11.values
    current_strain11 = model._stress.strain11
    current_strain11_values = model._stress.strain11.values
    d0_grid = model.d0

    # Should not re-calculate stress
    model.calculate_stress('diagonal', 200, 0.3, (1, 0))
    assert model._stress is current_stress
    assert model._stress.strain11 is current_strain11
    np.testing.assert_equal(current_stress11_values, model._stress.stress11.values)
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)

    # Should re-calculate stress with different young modulus, stress*2
    model.calculate_stress('diagonal', 400, 0.3, (1, 0))
    assert model._stress is current_stress
    assert model._stress.strain11 is current_strain11
    np.testing.assert_allclose(current_stress11_values*2, model._stress.stress11.values)
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)

    # Should re-calculate stress with different poissons ratio
    model.calculate_stress('diagonal', 200, 0.4, (1, 0))
    assert model._stress is current_stress
    assert model._stress.strain11 is current_strain11
    assert not np.all(current_stress11_values == model._stress.stress11.values)
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)

    # Should re-calculate strain and stress with new d, but should be the same StressField
    model.calculate_stress('diagonal', 200, 0.3, (1.08, 0))
    assert model._stress is current_stress
    assert model._stress.strain11 is current_strain11
    assert not np.all(current_stress11_values == model._stress.stress11.values)
    assert not np.all(current_strain11_values == model._stress.strain11.values)

    # Should re-calculate strain and stress with new d grid
    model.calculate_stress('diagonal', 200, 0.3,
                           (list(d0_grid.values),
                            list(d0_grid.errors),
                            list(d0_grid.x),
                            list(d0_grid.y),
                            list(d0_grid.z)))
    assert model._stress is current_stress
    assert model._stress.strain11 is current_strain11
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)

    # Should build new StressField
    model.calculate_stress('in-plane-stress', 200, 0.3, (1, 0))
    assert model._stress is not current_stress
    assert model._stress.strain11 is current_strain11
    np.testing.assert_equal(current_strain11_values, model._stress.strain11.values)

    # Check message when 22 is loaded without 11
    model = Model()
    model.e22 = os.path.join(test_data_dir, 'HB2B_1320.h5')
    assert model.e22 is not None
    assert model.validate_selection('22') == "e11 is not loaded, the peak tags from this file will be used"

    # try loading a file that isn't a HidraProjectFile
    model.e22 = os.path.join(test_data_dir, 'HB2B_938.nxs.h5')
    assert model.e22 == []

    # Check set_workspace, this is what is called by the controller
    model.e11 == []
    model.set_workspaces('11', os.path.join(test_data_dir, 'HB2B_1320.h5'))
    model.e11 is not None


def test_model_multiple_files(tmpdir, test_data_dir):
    model = Model()

    assert model.selectedPeak is None
    assert model.e11 == []
    assert model.e22 == []
    assert model.e33 == []
    assert model._stress is None

    # load 2 files with fitted peaks
    model.e11 = [os.path.join(test_data_dir, name) for name in ('HB2B_1327.h5', 'HB2B_1328.h5')]
    assert len(model.e11) == 2
    assert model.e11[0].name == '11'
    assert model.e11[1].name == '11'
    assert model.peakTags == ['peak0', 'peak1']
    assert len(model.e11_peaks) == 2
    assert 'peak0' in model.e11_peaks[0]
    assert 'peak0' in model.e11_peaks[1]

    # select non-existing peak
    model.selectedPeak = 'peak_label'
    assert model.selectedPeak == 'peak_label'
    assert model.validate_selection('11') == "Peak peak_label is not in e11"

    # select existing peak
    model.selectedPeak = 'peak0'
    assert model.selectedPeak == 'peak0'

    assert model.validate_selection('11') is None
    assert model.validate_selection('22') == "e22 file hasn't been loaded"
    assert model.validate_selection('33') == "e33 file hasn't been loaded"

    assert model.d0 is None

    for plot_param in ("dspacing-center",
                       "d-reference",
                       "Center",
                       "Height",
                       "FWHM",
                       "Mixing",
                       "Intensity",
                       "strain"):
        e11_md = model.get_field('11', plot_param, 'stress_case').to_md_histo_workspace()
        assert e11_md.name() == f'{plot_param}'
        assert e11_md.getNumDims() == 3
        assert [e11_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]
        assert model.get_field('22', plot_param, 'stress_case') is None
        assert model.get_field('33', plot_param, 'stress_case') is None

    # Need to load ε22 so it should fail
    with pytest.raises(TypeError) as exception_info:
        model.calculate_stress('in-plane-stress', 200, 0.3, (1.05, 0))
    assert 'None is not a _StrainField object' in str(exception_info.value)

    model.e22 = [os.path.join(test_data_dir, name) for name in ('HB2B_1327.h5', 'HB2B_1328.h5')]
    assert len(model.e22) == 2
    assert model.e22[0].name == '22'
    assert model.e22[1].name == '22'

    model.calculate_stress('in-plane-stress', 200, 0.3, (1.05, 0))

    assert model.d0.values[0] == 1.05

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'In-plane stress').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Stress should be all zero for in-plane stress case
    # assert np.count_nonzero(model.get_field('33', 'stress', 'In-plane stress')
    #                        .to_md_histo_workspace().getSignalArray()) == 0
    # Strain should be all non-zero for in-plane stress case
    assert np.count_nonzero(model.get_field('33', 'strain', 'In-plane stress')
                            .to_md_histo_workspace().getSignalArray()) == 18*6*3

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1327_1328_1327_1328_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv1.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318

    model.calculate_stress('in-plane-strain', 200, 0.3, (1.05, 0))

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'In-plane strain').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Stress should be all non-zero for in-plane strain case
    # assert np.count_nonzero(model.get_field('33', 'stress', 'In-plane strain')
    #                        .to_md_histo_workspace().getSignalArray()) == 18*49*3
    # Strain shouldn't exist for for in-plane strain case
    assert model.get_field('33', 'strain', 'In-plane strain') is None

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1327_1328_1327_1328_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv2.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318

    model.e33 = os.path.join(test_data_dir, 'HB2B_1327.h5')
    assert len(model.e33) == 1
    assert model.e33[0].name == '33'

    model.calculate_stress('diagonal', 200, 0.3, (1.05, 0))

    for direction in ('11', '22', '33'):
        stress_md = model.get_field(direction, 'stress', 'diagonal').to_md_histo_workspace()
        assert stress_md.name() == 'stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Should be all non-zero for diagonal stress case
    assert np.count_nonzero(model.get_field('33', 'stress', 'diagonal')
                            .to_md_histo_workspace().getSignalArray()) == 18*6*3

    # Check default csv filename
    assert model.get_default_csv_filename() == "HB2B_1327_1328_1327_1328_1327_stress_grid_peak0.csv"

    # Check writing csv file
    filename = tmpdir.join("test_model_csv3.csv")
    model.write_stress_to_csv(str(filename))
    # check number of lines written
    assert len(open(filename).readlines()) == 318
