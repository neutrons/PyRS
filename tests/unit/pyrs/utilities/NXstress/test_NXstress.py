import numpy as np
from nexusformat.nexus import (
    NXdata,
    NXentry,
    NXinstrument,
    NXparameters,
    NXprocess,
    NXreflections,
    NXsample
)
from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
import pyrs.utilities.NXstress._definitions
from pyrs.utilities.NXstress._fit import _Fit
from pyrs.utilities.NXstress._input_data import _InputData
from pyrs.utilities.NXstress._instrument import _Instrument
from pyrs.utilities.NXstress._peaks import _Peaks
from pyrs.utilities.NXstress._sample import _Sample
from pyrs.utilities.NXstress.NXstress import NXstress

import pytest

class TestNXstress:
    # instrument, input data, reduced data, no mask
    PROJECT_FILE_A = "HB2B_1017.h5"
   
    # instrument, mask, reduced data, but no input data
    PROJECT_FILE_B = "HB2B_1628.h5"

    # instrument, mask (from '1628'), input data, reduced data
    PROJECT_FILE_C = "HB2B_1017_w_mask.h5"
 
    
    @pytest.fixture(autouse=True)
    def setUp(self, load_HidraWorkspace, createPeakCollection):
        """
        self.sampleLogs = self.ws._sample_logs
        self.subruns = self.ws._sample_logs.subruns.raw_copy()
        
        N_subrun = len(self.subruns)
        self.peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )
        self.peak1 = createPeakCollection(
            peak_tag="111 Si",
            peak_profile="PseudoVoigt",
            background_type="Linear",
            wavelength=10.1,
            projectfilename="/does/not/exist2.h5",
            runnumber=12346,
            N_subrun=N_subrun   
        )
        """
        # Unfortunately, no available project file actually includes
        # all of the necessary data to initialize the NXstress file.
        # The tests below, use different files to verify different sections.
        yield
        
        # teardown follows ...
        pass
    
    def test_NXstress_context_manager(
        self,
        tmp_path: Path,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_C,
            name='test_workspace',
            # raw-counts load => instrument load
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
        sampleLogs = ws._sample_logs
        subruns = sampleLogs.subruns.raw_copy()
        
        N_subrun = len(subruns)
        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )        
        
        file_path = tmp_path / 'test_NXstress_context_manager.nxs'
        assert not file_path.exists()
        
        with NXstress(file_path, 'w') as nx:
            nx.write(ws, peak0)
            assert nx._nx._root is not None
        assert file_path.exists()
    
    def test_NXentry_fields(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )

        required_fields = (
            'definition',
            'start_time',
            'end_time',
            'processing_type'            
        )
        entry = NXstress._init_group(ws)
        assert isinstance(entry, NXentry)
        for key in required_fields:
            assert key in entry
    
    def test_NXentry_multiple(
        self,
        tmp_path: Path,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_C,
            name='test_workspace',
            # raw-counts load => instrument load
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
        sampleLogs = ws._sample_logs
        subruns = sampleLogs.subruns.raw_copy()
        
        N_subrun = len(subruns)
        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )        
        
        file_path = tmp_path / 'test_NXentry_multiple.nxs'
        with NXstress(file_path, 'w') as nx:
            nx.write(ws, peak0)
            nx.write(ws, peak0)
            assert len(nx._nx.root.NXentry) == 2
            assert 'entry' in nx._nx.root
            assert 'entry_2' in nx._nx.root
        
    def test__Instrument(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A,
            name='test_workspace',
            # raw-counts load => instrument load
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
        
        inst = _Instrument.init_group(ws)
        assert isinstance(inst, NXinstrument)
                
    def test__Sample(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )

        sample = _Sample.init_group(ws._sample_logs)
        assert isinstance(sample, NXsample)
                
    def test__Fit(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )
        sampleLogs = ws._sample_logs
        subruns = sampleLogs.subruns.raw_copy()
        
        N_subrun = len(subruns)
        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )        
        
        fit = _Fit.init_group('_DEFAULT_', ws, peak0, sampleLogs)
        assert isinstance(fit, NXprocess)
                
    def test__Fit_multiple(self):
        pass
                
    def test__Peaks(
        self,
        tmp_path: Path,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )
        sampleLogs = ws._sample_logs
        subruns = sampleLogs.subruns.raw_copy()
        
        N_subrun = len(subruns)
        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )
                
        peaks = _Peaks.init_group(peak0, sampleLogs)
        assert isinstance(peaks, NXreflections)
    
    def test__InputData(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A,
            name='test_workspace',
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
            
        raw_data = _InputData.init_group(ws)
        assert isinstance(raw_data, NXdata)
        
    def test__InputData_omitted(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            # PROJECT_B doesn't include any raw-counts data.
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )
            
        raw_data = _InputData.init_group(ws)
        assert raw_data is not None
        assert isinstance(raw_data, NXdata)
