from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile  # type: ignore
from pyrs.core.instrument_geometry import HidraSetup  # type: ignore
import os
import pytest


def test_read_write_merged_project_file():
    '''
    Testing that merged project files are propperly read into pyrs

    Returns
    -------
    None.

    '''
    # import data file: detector ID and file name
    test_data = ['tests/data/HB2B_1327.h5',
                 'tests/data/HB2B_1328.h5',
                 'tests/data/HB2B_1331.h5',
                 'tests/data/HB2B_1332.h5']

    test_ws = HidraWorkspace('test_powder_pattern')
    test_project = HidraProjectFile(test_data[0])
    test_ws.load_hidra_project(test_project, load_raw_counts=False, load_reduced_diffraction=True)
    test_project.close()

    for test_file in test_data[1:]:
        test_project = HidraProjectFile(test_file)
        test_ws.append_hidra_project(test_project)
        test_project.close()

    export_project = HidraProjectFile('./HB2B_1327_1328_1331_1332.h5', 'w')
    test_ws.save_experimental_data(export_project, sub_runs=test_ws._sample_logs.subruns, ignore_raw_counts=True)
    test_ws.save_reduced_diffraction_data(export_project)
    export_project.save()

    exported_ws = HidraWorkspace('exported data from various runs')
    exported_projectfile = HidraProjectFile('./HB2B_1327_1328_1331_1332.h5')
    exported_ws.load_hidra_project(exported_projectfile, load_raw_counts=False, load_reduced_diffraction=True)
    exported_projectfile.close()

    assert test_ws.get_sub_runs() == exported_ws.get_sub_runs()


def test_rw_raw(test_data_dir):
    """Test read a project to workspace and write in the scope of raw data

    Returns
    -------

    """
    raw_project_name = os.path.join(test_data_dir, 'HZB_Raw_Project.h5')

    # Read to workspace
    source_project = HidraProjectFile(raw_project_name, 'r')

    # To the workspace
    source_workspace = HidraWorkspace('Source HZB')
    source_workspace.load_hidra_project(source_project, load_raw_counts=True,
                                        load_reduced_diffraction=False)

    # Export
    target_project = HidraProjectFile('HZB_HiDra_Test.h5', 'w')
    # Experiment data
    source_workspace.save_experimental_data(target_project, sub_runs=range(1, 41))

    # Instrument
    detector_setup = source_workspace.get_instrument_setup()
    instrument_setup = HidraSetup(detector_setup=detector_setup)
    target_project.write_instrument_geometry(instrument_setup)

    # Save
    target_project.save(True)

    return


if __name__ == '__main__':
    pytest.main([__file__])
