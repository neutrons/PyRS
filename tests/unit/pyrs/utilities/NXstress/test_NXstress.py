import numpy as np
from nexusformat.nexus import (
    NXbeam,
    NXcollection,
    NXdata,
    NXdetector,
    NXentry,
    NXinstrument,
    NXmonochromator,
    NXnote,
    NXparameters,
    NXprocess,
    NXreflections,
    NXsample,
    NXsource
)
from pathlib import Path
import shutil

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._definitions import (
    FIELD_DTYPE, GROUP_NAME
)
from pyrs.utilities.NXstress._fit import (
    _Diffractogram, _Fit, _PeakParameters, _BackgroundParameters
)
from pyrs.utilities.NXstress._input_data import _InputData
from pyrs.utilities.NXstress._instrument import (
    _Instrument, _Masks
)
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
            assert nx._root is not None
        assert file_path.exists()
    
    def test_NXentry_fields(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        # Verify that all required datasets, and attributes are present
        #   on the `NXentry` 
        
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_C,
            name='test_workspace',
            # raw-counts load => instrument load
            load_raw_counts=True,
            load_reduced_diffraction=True
        )

        required_datasets = (
            'definition',
            'start_time',
            'end_time',
            'processing_type'            
        )
        
        entry = NXstress._init(ws)
        assert isinstance(entry, NXentry)        
        for key in required_datasets:
            assert key in entry
    
    def test_NXentry_subgroups(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        # Verify that all required subgroups are present
        #   on the `NXentry` 
        
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
        
        required_groups = (
            (GROUP_NAME.SAMPLE_DESCRIPTION, NXsample),
            (GROUP_NAME.FIT, NXprocess),
            (GROUP_NAME.PEAKS, NXreflections) 
        )
        
        entry = NXstress.init_group(ws, peak0)
        assert isinstance(entry, NXentry)
        for key, NXclass_ in required_groups:
            assert key in entry
            assert isinstance(entry[key], NXclass_)
    
    def test_NXentry_input_data(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        # Verify that an optional `input_data` `NXdata` group will be created on the `NXentry`
        #   when detector-counts data is attached to the source workspace.
        
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
        
        entry = NXstress.init_group(ws, peak0)
        assert isinstance(entry, NXentry)
        key, NXclass_ = GROUP_NAME.INPUT_DATA, NXdata
        assert key in entry
        assert isinstance(entry[key], NXclass_)

    
    def test_NXentry_input_data_optional(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        # When no input data is attached to the source workspace:
        #   verify that an empty (i.e. no scan-points) `input_data` `NXdata` group is created on the `NXentry`.
        
        # Notes:
        # -- A successful instrument load is required; this is keyed to detector-counts data load.
        #    So we need to fudge the workspace after the load in order to _remove_ the attached input data.
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_C,
            name='test_workspace',
            # raw-counts load => instrument load
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
        # remove the input data:
        ws._raw_counts = dict()
        
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

        entry = NXstress.init_group(ws, peak0)
        assert isinstance(entry, NXentry)
        key, NXclass_ = GROUP_NAME.INPUT_DATA, NXdata
        assert key in entry
        assert isinstance(entry[key], NXclass_)
        assert len(entry[key]['scan_point']) == 0

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
        peak1 = createPeakCollection(
            peak_tag="Si 111",
            peak_profile="PseudoVoigt",
            background_type="Quadratic",
            wavelength=20.6,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )
                
        file_path = tmp_path / 'test_NXstress_multiple_NXentry.nxs'
        with NXstress(file_path, 'w') as nx:
            nx.write(ws, peak0)
            nx.write(ws, peak1)
            assert len(nx._root.NXentry) == 2
            assert 'entry' in nx._root
            assert 'entry_2' in nx._root
        assert file_path.exists()
        
        # *** DEBUG *** : for validation:
        shutil.copy2(file_path, Path('/mnt/data0/workspaces/ORNL-work/PyRS/tmp/validation'))
        
    def test__Instrument_fields_and_subgroups(
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
        
        required_fields = ('name',)
        required_subgroups = (
            (GROUP_NAME.SOURCE, NXsource),
            (GROUP_NAME.BEAM, NXbeam),
            (GROUP_NAME.MONOCHROMATOR, NXmonochromator),
            (GROUP_NAME.DETECTOR, NXdetector),
            # Optional field for NXstress:
            (GROUP_NAME.MASKS, NXcollection)
        )
        
        inst = _Instrument.init_group(ws)
        assert isinstance(inst, NXinstrument)
        for key in required_fields:
            assert key in inst
        for key, NXclass_ in required_subgroups:
            assert key in inst
            assert isinstance(inst[key], NXclass_)
        
    def test__Masks_fields_and_subgroups(
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
        
        required_fields = ('names',)
        required_subgroups = (
            ('detector', NXcollection),
            ('solid_angle', NXcollection)
        )
        
        masks = _Masks.init_group(ws, detectorMasks=True)
        assert isinstance(masks, NXcollection)
        for key in required_fields:
            assert key in masks
        for key, NXclass_ in required_subgroups:
            assert key in masks
            assert isinstance(masks[key], NXclass_)
                 
    def test__Sample_fields_and_subgroups(
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
            'name',
            # not required by `NXstress`, but _possibly_ required by PyRS:
            'vx', 'vy', 'vz'
        )
        required_subgroups = ()
        
        sample = _Sample.init_group(ws._sample_logs)
        assert isinstance(sample, NXsample)
        for key in required_fields:
            assert key in sample
        # Placeholder: no required subgroups yet:
        for key, NXclass_ in required_subgroups:
            assert key in sample
            assert isinstance(sample[key], NXclass_)
                
    def test__Fit_fields_and_subgroups(
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
        
        required_fields = (
            'raw_data_file',
            'date',
            'program'
        )
        required_subgroups = (
            (GROUP_NAME.DESCRIPTION, NXnote),
            (GROUP_NAME.PEAK_PARAMETERS, NXparameters),
            (GROUP_NAME.BACKGROUND_PARAMETERS, NXparameters),
            (GROUP_NAME.DIFFRACTOGRAM, NXdata)
        )
        
        fit = _Fit.init_group('_DEFAULT_', ws, peak0, sampleLogs)
        assert isinstance(fit, NXprocess)
        for key in required_fields:
            assert key in fit
        for key, NXclass_ in required_subgroups:
            assert key in fit
            assert isinstance(fit[key], NXclass_)
                
    def test__Fit_multiple(self):
        pass
                
    def test__PeakParameters_fields_and_subgroups(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        
        # Load a workspace in order to get a realistic <scan point> axis.
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
            peak_profile="PseudoVoigt",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )        
        peak0_params_value, peak0_params_error = peak0.get_effective_params()
        
        required_fields = (
            'title',
            'center',
            # Not all of these fields are required by `NXstress`, but
            #   as these are available in PyRS, they will be used here.
            'center_errors',
            'height',
            'height_errors',
            'fwhm',
            'fwhm_errors',
            'form_factor',
            'form_factor_errors'
        )
        
        peak_parameters = _PeakParameters.init_group(peak0)
        assert isinstance(peak_parameters, NXparameters)
        for key in required_fields:
            assert key in peak_parameters
        
        # Verify mixing to form factor conversion.
        assert pytest.approx(peak_parameters['form_factor'], 1.0e-6) ==\
            (1.0 - peak0_params_value['Mixing']).astype(FIELD_DTYPE.FLOAT_DATA.value)
                
    def test__BackgroundParameters_fields_and_subgroups(
        self,
        load_HidraWorkspace: HidraWorkspace,
        createPeakCollection: PeakCollection
        ):
        
        # Load a workspace in order to get a realistic <scan point> axis.
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
            peak_profile="PseudoVoigt",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun    
        )        
        peak0_params_value, peak0_params_error = peak0.get_effective_params()
        
        required_fields = (
            'title',
            # Not all of these fields are required by `NXstress`, but
            #   as PyRS uses polynomial background functions, they will be used here.
            'A',
            'A_errors'
        )
        
        background_parameters = _BackgroundParameters.init_group(peak0)
        assert isinstance(background_parameters, NXparameters)
        for key in required_fields:
            assert key in background_parameters
                
    def test__Diffractogram_fields_and_subgroups(
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
        
        required_attributes = (
            'signal',
            'auxiliary_signals',
            'axes'
        )
        required_fields = (
            'scan_point',
            'two_theta',
            GROUP_NAME.DGRAM_DIFFRACTOGRAM,
            GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS,
            GROUP_NAME.DGRAM_FIT,
            GROUP_NAME.DGRAM_FIT_ERRORS
        )
        required_subgroups = ()
        
        dgram = _Diffractogram.init_group(ws, '_DEFAULT_', peak0)
        assert isinstance(dgram, NXdata)
        for key in required_attributes:
            assert key in dgram.attrs
        for key in required_fields:
            assert key in dgram
        # Placeholder: there are currently no required subgroups:
        for key, NXclass_ in required_subgroups:
            assert key in dgram
            assert isinstance(dgram[key], NXclass_)
                
    def test__Peaks_fields_and_subgroups(
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
        
        required_fields = (
            'h', 'k', 'l', 'phase_name', 'qx', 'qy', 'qz',
            'center', 'center_errors', 'center_type',
            'sx', 'sy', 'sz'
        )
        required_subgroups = ()
             
        peaks = _Peaks.init_group(peak0, sampleLogs)
        assert isinstance(peaks, NXreflections)
        for key in required_fields:
            assert key in peaks
        # Placeholder: no required subgroups yet:
        for key, NXclass_ in required_subgroups:
            assert key in peaks
            assert isinstance(peaks[key], NXclass_)

    def test__parse_peak_tag(self):
        phase, (h, k, l) = _Peaks._parse_peak_tag('Al 452411') 
        assert phase == 'Al'
        assert (45, 24, 11) == (h, k, l)
        
        phase, (h, k, l) = _Peaks._parse_peak_tag('111 Al2O3') 
        assert phase == 'Al2O3'
        assert (1, 1, 1) == (h, k, l)
        
        phase, (h, k, l) = _Peaks._parse_peak_tag('010203 Silicon') 
        assert phase == 'Silicon'
        assert (1, 2, 3) == (h, k, l)
        
        # Behavior check: takes longest digits substring.
        #   This would not be a "real" peak tag!
        phase, (h, k, l) = _Peaks._parse_peak_tag('321 010203 Silicon') 
        assert phase == '321  Silicon'
        assert (1, 2, 3) == (h, k, l)
        
        with pytest.raises(RuntimeError, match=r".*Unable to parse peak tag.*"):
            # substrings must have length divisible by 3
            _phase, (_h, _k, _l) = _Peaks._parse_peak_tag('0102 Silicon') 
        
        with pytest.raises(RuntimeError, match=r".*Unable to parse peak tag.*"):
            # must contain an (h, k, l) substring
            _phase, (_h, _k, _l) = _Peaks._parse_peak_tag('Silicon') 
        
        with pytest.raises(RuntimeError, match=r".*Unable to parse <phase name> from peak tag.*"):
            # must contain a phase substring
            _phase, (_h, _k, _l) = _Peaks._parse_peak_tag('102030') 
        
    def test__InputData_fields_and_subgroups(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A,
            name='test_workspace',
            load_raw_counts=True,
            load_reduced_diffraction=True
        )
        
        required_attributes = (
            'axes', 'signal'
        )
        required_fields = (
            'scan_point', 'detector_counts'
        )
        required_subgroups = ()  
        
        data = _InputData.init_group(ws)
        assert isinstance(data, NXdata)
        for key in required_attributes:
            assert key in data.attrs
        for key in required_fields:
            assert key in data
        # Placeholder: no required subgroups yet:
        for key, NXclass_ in required_subgroups:
            assert key in data
            assert isinstance(data[key], NXclass_)
        
    def test__InputData_omitted(
        self,
        load_HidraWorkspace: HidraWorkspace,
        ):
        # When input-data is not attached to the source workspace,
        #   the structure of the input-data group should still be filled in.
        ws = load_HidraWorkspace(
            # PROJECT_B doesn't include any raw-counts data.
            file_name=self.PROJECT_FILE_B,
            name='test_workspace',
            load_raw_counts=False,
            load_reduced_diffraction=True
        )
        
        required_attributes = (
            'axes', 'signal'
        )
        required_fields = (
            'scan_point', 'detector_counts'
        )
        required_subgroups = ()  
                    
        data = _InputData.init_group(ws)
        assert isinstance(data, NXdata)
        for key in required_attributes:
            assert key in data.attrs
        for key in required_fields:
            assert key in data
        # Placeholder: no required subgroups yet:
        for key, NXclass_ in required_subgroups:
            assert key in data
            assert isinstance(data[key], NXclass_)
