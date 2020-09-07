import numpy as np
import pytest
import os

from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.dataobjects.constants import HidraConstants


@pytest.fixture(scope='module')
def converter_HB2B_938(test_data_dir):
    r"""File with only one subrun"""
    return NeXusConvertingApp(os.path.join(test_data_dir, 'HB2B_938.nxs.h5'))


class TestNeXusConvertingApp:

    def test_split_sample_logs(self, converter_HB2B_938):
        converter_HB2B_938.split_sample_logs(subruns=np.zeros(1, dtype=int))
        sample_logs = converter_HB2B_938._hidra_workspace._sample_logs
        assert sample_logs.units('vx') == 'mm'
        assert sample_logs.units(HidraConstants.SUB_RUN_DURATION) == 'second'
