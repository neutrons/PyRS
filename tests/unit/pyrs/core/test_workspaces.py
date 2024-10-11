import numpy as np

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile  # type: ignore


class TestHidraWorkspace:

    def test_set_sample_log(self):
        workspace = HidraWorkspace()
        subruns, vx = np.array([1, 2, 3], dtype=int), np.array([0.0, 0.1, 0.2])
        workspace.set_sample_log('vx', subruns, vx)
        assert workspace.get_sample_log_units('vx') == ''
        workspace.set_sample_log('vx', subruns, vx, 'mm')
        assert workspace.get_sample_log_units('vx') == 'mm'

    def test_append_projectfiels(self):

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

        assert test_ws.get_sub_runs().size == 362
