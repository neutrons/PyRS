import numpy as np

from pyrs.core.workspaces import HidraWorkspace


class TestHidraWorkspace:

    def test_set_sample_log(self):
        workspace = HidraWorkspace()
        subruns, vx = np.array([1, 2, 3], dtype=int), np.array([0.0, 0.1, 0.2])
        workspace.set_sample_log('vx', subruns, vx)
        assert workspace.get_sample_log_units('vx') == ''
        workspace.set_sample_log('vx', subruns, vx, 'mm')
        assert workspace.get_sample_log_units('vx') == 'mm'
