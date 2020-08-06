# This is a test of the model component of the strain/stress viewer
from pyrs.interface.strainstressviewer.model import Model
import numpy as np
import pytest


def test_model(tmpdir):
    model = Model()

    assert model.selectedPeak is None
    assert model.e11 is None
    assert model.e22 is None
    assert model.e33 is None
    assert model._stress is None

    assert model.validate_selection('11') == "e11 file hasn't been loaded"
    assert model.validate_selection('22') == "e22 file hasn't been loaded"
    assert model.validate_selection('33') == "e33 file hasn't been loaded"

    # load file without fitted peaks
    model.e11 = 'tests/data/HB2B_1423.h5'
    assert model.validate_selection('11') == "e11 contains no peaks, fit peaks first"

    # load file with fitted peaks
    model.e11 = 'tests/data/HB2B_1320.h5'
    assert model.e11.name == '11'
    assert model.peakTags == ['peak0']
    assert 'peak0' in model.e11_peaks

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

    assert model.d0 == 1

    model.d0 = 1.05
    assert model.d0 == 1.05

    for plot_param in ("dspacing_center",
                       "d_reference",
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
    with pytest.raises(KeyError):
        model.calculate_stress('in-plane-stress', 200, 0.3)

    model.e22 = 'tests/data/HB2B_1320.h5'
    assert model.e22.name == '22'

    model.calculate_stress('in-plane-stress', 200, 0.3)

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

    model.calculate_stress('in-plane-strain', 200, 0.3)

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

    model.e33 = 'tests/data/HB2B_1320.h5'
    assert model.e33.name == '33'

    model.calculate_stress('diagonal', 200, 0.3)

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

    # Check message when 22 is loaded without 11
    model = Model()
    model.e22 = 'tests/data/HB2B_1320.h5'
    assert model.e22 is not None
    assert model.validate_selection('22') == "e11 is not loaded, the peak tags from this file will be used"

    # try loading a file that isn't a HidraProjectFile
    model.e22 = 'tests/data/HB2B_938.nxs.h5'
    assert model.e22 is None

    # Check set_workspace, this is what is called by the controller
    model.e11 is None
    model.set_workspace('11', 'tests/data/HB2B_1320.h5')
    model.e11 is not None
