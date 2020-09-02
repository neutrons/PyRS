from pyrs.interface.strainstressviewer.strain_stress_view import StrainStressViewer
from pyrs.interface.strainstressviewer.model import Model
from pyrs.interface.strainstressviewer.controller import Controller
from qtpy import QtCore, QtWidgets
import functools
import numpy as np
import os
import pytest
import json
from tests.conftest import ON_TRAVIS  # set to True when running on build servers

wait = 100


# This is a test of the model component of the strain/stress viewer
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

    model.calculate_stress('in-plane-stress', 200, 0.3, None)

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv1_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318

    # Check saving to JSON file
    filename = tmpdir.join("test_model1.json")
    model.to_json(filename)
    # check file output
    with open(filename) as f:
        data = json.load(f)
    assert data.pop("stress_case") == "in-plane-stress"
    filenames_11 = data.pop("filenames_11")
    assert len(filenames_11) == 1
    assert filenames_11[0].endswith("tests/data/HB2B_1320.h5")
    filenames_22 = data.pop("filenames_22")
    assert len(filenames_22) == 1
    assert filenames_22[0].endswith("tests/data/HB2B_1320.h5")
    filenames_33 = data.pop("filenames_33")
    assert len(filenames_33) == 0
    assert data.pop("peak_tag") == "peak0"
    assert data.pop("youngs_modulus") == 200
    assert data.pop("poisson_ratio") == 0.3
    assert data.pop("d0") is None
    assert len(data) == 0

    # Check writing with bad filename, should fail but should emit failure message
    model.write_stress_to_csv("/bin/false", False)

    model.calculate_stress('in-plane-strain', 200, 0.3, (1.08, 0))

    assert model.d0.values[0] == 1.08

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv2_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318

    # Check saving to JSON file
    filename = tmpdir.join("test_model2.json")
    model.to_json(filename)
    # check file output
    with open(filename) as f:
        data = json.load(f)
    assert data.pop("stress_case") == "in-plane-strain"
    filenames_11 = data.pop("filenames_11")
    assert len(filenames_11) == 1
    assert filenames_11[0].endswith("tests/data/HB2B_1320.h5")
    filenames_22 = data.pop("filenames_22")
    assert len(filenames_22) == 1
    assert filenames_22[0].endswith("tests/data/HB2B_1320.h5")
    filenames_33 = data.pop("filenames_33")
    assert len(filenames_33) == 0
    assert data.pop("peak_tag") == "peak0"
    assert data.pop("youngs_modulus") == 200
    assert data.pop("poisson_ratio") == 0.3
    d0 = data.pop("d0")
    assert d0["d0"] == 1.08
    assert d0["d0_error"] == 0
    assert len(data) == 0

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv3_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318

    # Check saving to JSON file
    filename = tmpdir.join("test_model3.json")
    model.to_json(filename)
    # check file output
    with open(filename) as f:
        data = json.load(f)
    assert data.pop("stress_case") == "diagonal"
    filenames_11 = data.pop("filenames_11")
    assert len(filenames_11) == 1
    assert filenames_11[0].endswith("tests/data/HB2B_1320.h5")
    filenames_22 = data.pop("filenames_22")
    assert len(filenames_22) == 1
    assert filenames_22[0].endswith("tests/data/HB2B_1320.h5")
    filenames_33 = data.pop("filenames_33")
    assert len(filenames_33) == 1
    assert filenames_33[0].endswith("tests/data/HB2B_1320.h5")
    assert data.pop("peak_tag") == "peak0"
    assert data.pop("youngs_modulus") == 200
    assert data.pop("poisson_ratio") == 0.3
    d0 = data.pop("d0")
    assert d0["d0"] == 1
    assert d0["d0_error"] == 0
    assert len(data) == 0

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

    # Check saving to JSON file
    filename = tmpdir.join("test_model4.json")
    model.to_json(filename)
    # check file output
    with open(filename) as f:
        data = json.load(f)
    assert data.pop("stress_case") == "diagonal"
    filenames_11 = data.pop("filenames_11")
    assert len(filenames_11) == 1
    assert filenames_11[0].endswith("tests/data/HB2B_1320.h5")
    filenames_22 = data.pop("filenames_22")
    assert len(filenames_22) == 1
    assert filenames_22[0].endswith("tests/data/HB2B_1320.h5")
    filenames_33 = data.pop("filenames_33")
    assert len(filenames_33) == 1
    assert filenames_33[0].endswith("tests/data/HB2B_1320.h5")
    assert data.pop("peak_tag") == "peak0"
    assert data.pop("youngs_modulus") == 200
    assert data.pop("poisson_ratio") == 0.3
    d0 = data.pop("d0")
    np.testing.assert_equal(d0["vx"], d0_grid.x)
    np.testing.assert_equal(d0["vy"], d0_grid.y)
    np.testing.assert_equal(d0["vz"], d0_grid.z)
    np.testing.assert_equal(d0["d0"], d0_grid.values)
    np.testing.assert_equal(d0["d0_error"], d0_grid.errors)
    assert len(data) == 0

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv1_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318

    # Check saving to JSON file
    filename = tmpdir.join("test_model1.json")
    model.to_json(filename)
    # check file output
    with open(filename) as f:
        data = json.load(f)
    assert data.pop("stress_case") == "in-plane-stress"
    filenames_11 = data.pop("filenames_11")
    assert len(filenames_11) == 2
    assert filenames_11[0].endswith("tests/data/HB2B_1327.h5")
    assert filenames_11[1].endswith("tests/data/HB2B_1328.h5")
    filenames_22 = data.pop("filenames_22")
    assert len(filenames_22) == 2
    assert filenames_22[0].endswith("tests/data/HB2B_1327.h5")
    assert filenames_22[1].endswith("tests/data/HB2B_1328.h5")
    filenames_33 = data.pop("filenames_33")
    assert len(filenames_33) == 0
    assert data.pop("peak_tag") == "peak0"
    assert data.pop("youngs_modulus") == 200
    assert data.pop("poisson_ratio") == 0.3
    d0 = data.pop("d0")
    assert d0["d0"] == 1.05
    assert d0["d0_error"] == 0
    assert len(data) == 0

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv2_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318

    model.e33 = os.path.join(test_data_dir, 'HB2B_1327.h5')
    assert len(model.e33) == 1
    assert model.e33[0].name == '33'
    model._e33_strain.set_d_reference((1.05, 0))

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
    model.write_stress_to_csv(str(filename), False)
    # check number of lines written
    assert len(open(filename).readlines()) == 318
    filename = tmpdir.join("test_model_csv3_full.csv")
    model.write_stress_to_csv(str(filename), True)
    assert len(open(filename).readlines()) == 318


# changes to SliceViewer from Mantid in the version 5.1 is needed for the stress/strain viewer to run
@pytest.mark.skipif(ON_TRAVIS, reason='Need mantid version >= 5.1')
def test_stress_strain_viewer(qtbot):

    model = Model()
    ctrl = Controller(model)
    window = StrainStressViewer(model, ctrl)

    qtbot.addWidget(window)
    window.show()
    qtbot.wait(wait)

    assert window.isVisible()

    # This is to handle modal dialogs
    def handle_dialog(filename):
        # get a reference to the dialog and handle it here
        dialog = window.findChild(QtWidgets.QFileDialog)
        # get a File Name field
        lineEdit = dialog.findChild(QtWidgets.QLineEdit)
        # Type in file to load and press enter
        qtbot.keyClicks(lineEdit, filename)
        qtbot.wait(wait)
        qtbot.keyClick(lineEdit, QtCore.Qt.Key_Enter)

    # Browse e11 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "tests/data/HB2B_1320.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_e11.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)

    # check that the sliceviewer widget is created
    assert window.viz_tab.strainSliceViewer is not None

    # go to stress and check that it is removed
    qtbot.keyClick(window.plot_select.plot_param, QtCore.Qt.Key_Down)
    # check that the sliceviewer widget is removed
    assert window.viz_tab.strainSliceViewer is None

    qtbot.wait(wait)

    # Browse e22 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "tests/data/HB2B_1320.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_e22.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)

    # set constants
    qtbot.keyClicks(window.mechanicalConstants.youngModulus, "200")
    qtbot.keyClicks(window.mechanicalConstants.poissonsRatio, "0.3")
    qtbot.keyClick(window.mechanicalConstants.poissonsRatio, QtCore.Qt.Key_Enter)
    qtbot.wait(wait)

    # check that the sliceviewer widget is created
    assert window.viz_tab.strainSliceViewer is not None

    # Change dimension to 22
    qtbot.keyClick(window.plot_select.measure_dir, QtCore.Qt.Key_Down)
    qtbot.wait(wait)
    assert window.viz_tab.strainSliceViewer is not None

    # Change dimension to 33
    qtbot.keyClick(window.plot_select.measure_dir, QtCore.Qt.Key_Down)
    qtbot.wait(wait)
    assert window.viz_tab.strainSliceViewer is not None

    # Change to in-plane strain
    qtbot.keyClick(window.stressCase.combo, QtCore.Qt.Key_Down)
    qtbot.wait(wait)
    assert window.viz_tab.strainSliceViewer is not None

    # Change to 3D
    qtbot.mouseClick(window.stressCase.switch.button_3d, QtCore.Qt.LeftButton)
    qtbot.wait(wait)
    # check that the sliceviewer widget is removed
    assert window.viz_tab.strainSliceViewer is None

    # Browse e33 Data File ...
    # wait until dialog is loaded then handle it, this is required
    # because the dialog is modal
    QtCore.QTimer.singleShot(500, functools.partial(handle_dialog, "tests/data/HB2B_1320.h5"))
    qtbot.mouseClick(window.fileLoading.file_load_e33.browse_button, QtCore.Qt.LeftButton)

    qtbot.wait(wait)

    # check that the sliceviewer widget is created
    assert window.viz_tab.strainSliceViewer is not None

    qtbot.wait(wait)

    # Change plot selection, check other paramters
    for _ in range(9):
        qtbot.keyClick(window.plot_select.plot_param, QtCore.Qt.Key_Up)
        qtbot.wait(wait)
        # check that the sliceviewer widget is created
        assert window.viz_tab.strainSliceViewer is not None
