import os
import pytest
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp


def _nexus_to_subscans(nexusfile, mask_file_name):
    converter = NeXusConvertingApp(nexusfile, mask_file_name)
    hydra_ws = converter.convert(use_mantid=False)
    return hydra_ws

def create_powder_patterns(hidra_workspace, calibration, mask, subruns, project_file_name):
    reducer = ReductionApp(False)
    # load HidraWorkspace
    reducer.load_hidra_workspace(hidra_workspace)

    reducer.reduce_data(instrument_file=None,
                        calibration_file=calibration,
                        mask=mask,
                        sub_runs=subruns)

    reducer.save_diffraction_data(project_file_name)

@pytest.mark.skipif(not os.path.isdir('/HFIR/HB2B/'), reason="Test requires access to the HFIR mount.")
def test_hidra_workflow(tmpdir):
    nexus = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1060.nxs.h5'
    mask = '/HFIR/HB2B/shared/CALIBRATION/HB2B_MASK_Latest.xml'
    calibration = '/HFIR/HB2B/shared/CALIBRATION/HB2B_Latest.json'
    hidra_ws = _nexus_to_subscans(nexus, mask_file_name=mask)
    project = os.path.basename(nexus).split('.')[0] + '.h5'
    project = os.path.join(str(tmpdir), project)
    try:
        create_powder_patterns(hidra_ws, calibration, None, list(), project)
    finally:
        os.remove(project)