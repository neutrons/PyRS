from unittest.mock import MagicMock

from mantid.kernel import Int32TimeSeriesProperty as IntLog
import numpy as np
import pytest
import os

from pyrs.core.nexus_conversion import Splitter, NeXusConvertingApp
from pyrs.dataobjects.constants import HidraConstants


@pytest.fixture(scope="module")
def converter_HB2B_938(test_data_dir):
    r"""File with only one subrun"""
    return NeXusConvertingApp(os.path.join(test_data_dir, "HB2B_938.nxs.h5"))


class TestNeXusConvertingApp:
    pytestmark = pytest.mark.integration

    def test_split_sample_logs(self, converter_HB2B_938):
        converter_HB2B_938.split_sample_logs(subruns=np.zeros(1, dtype=int))
        sample_logs = converter_HB2B_938._hidra_workspace._sample_logs
        assert sample_logs.units("vx") == "mm"
        assert sample_logs.units(HidraConstants.SUB_RUN_DURATION) == "second"


class TestSplitter:
    def test_empty_constructor(self):
        # pass in empty dictionary
        try:
            _ = Splitter(dict())
            assert False, "Should not have created a Splitter"
        except RuntimeError:
            pass

        try:
            logs = {"scan_index": IntLog("scan_index")}
            _ = Splitter(logs)
            assert False, "Should not have created a Splitter"
        except RuntimeError:
            pass

    def test_nominal(self):
        # single scan index
        scan_index = IntLog("scan_index")
        scan_index.addValue("1999-03-31T12:00Z", 0)
        scan_index.addValue("1999-03-31T12:01Z", 1)
        scan_index.addValue("1999-03-31T12:02Z", 0)

        splitter = Splitter({"scan_index": scan_index})

        assert splitter.size == 1
        np.testing.assert_equal(splitter.durations, [60])
        np.testing.assert_equal(splitter.subruns, [1])

        # add another scan index and create a new splitter
        scan_index.addValue("1999-03-31T12:03Z", 2)
        scan_index.addValue("1999-03-31T12:04Z", 0)

        splitter = Splitter({"scan_index": scan_index})

        assert splitter.size == 2
        np.testing.assert_equal(splitter.durations, [60, 60])
        np.testing.assert_equal(splitter.subruns, [1, 2])

    def test_no_end(self):
        # when end isn't defined one day into the future is used
        ONE_DAY = 24 * 60 * 60

        # single scan index
        scan_index = IntLog("scan_index")
        scan_index.addValue("1999-03-31T12:00Z", 0)
        scan_index.addValue("1999-03-31T12:01Z", 1)

        splitter = Splitter({"scan_index": scan_index})

        assert splitter.size == 1
        np.testing.assert_equal(splitter.durations, [ONE_DAY])
        np.testing.assert_equal(splitter.subruns, [1])

        # add another scan index and create a new splitter
        scan_index.addValue("1999-03-31T12:02Z", 0)
        scan_index.addValue("1999-03-31T12:03Z", 2)

        splitter = Splitter({"scan_index": scan_index})

        assert splitter.size == 2
        np.testing.assert_equal(splitter.durations, [60, ONE_DAY])
        np.testing.assert_equal(splitter.subruns, [1, 2])


def test_mask_workspace_named_from_file_name(tmp_path, monkeypatch):
    """A dot in the mask's directory does not truncate the mask workspace name."""
    # Arrange
    mask_dir = tmp_path / "masks.v1.2"
    mask_dir.mkdir()
    mask_file = mask_dir / "HB2B_Mask_12-18-19.xml"
    mask_file.write_text("<mask/>")
    load_mask_calls = []

    def fake_load_mask(**kwargs):
        load_mask_calls.append(kwargs)
        mask_ws = MagicMock()
        mask_ws.extractY.return_value = np.zeros((4, 1))
        return mask_ws

    monkeypatch.setattr("pyrs.core.nexus_conversion.LoadMask", fake_load_mask)
    monkeypatch.setattr("pyrs.core.nexus_conversion.DeleteWorkspace", lambda **kwargs: None)
    converter = NeXusConvertingApp.__new__(NeXusConvertingApp)
    converter._event_wksp = MagicMock()

    # Act
    converter._NeXusConvertingApp__load_mask(str(mask_file))

    # Assert
    assert load_mask_calls[0]["OutputWorkspace"] == "HB2B_Mask_12-18-19"
