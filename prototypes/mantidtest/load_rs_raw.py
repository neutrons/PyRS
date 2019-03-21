import h5py
import numpy

def load_raw_measurement_data(file_name):
        """
        Load raw data measured
        :param file_name:
        :return:
        """
        # checkdatatypes.check_file_name(file_name, check_exist=True)

        # access sub tree
        scan_h5 = h5py.File(file_name)
        if 'raw' not in scan_h5.keys() or 'instrument' not in scan_h5.keys():
            # TODO - TONIGHT 1 - better error message
            raise RuntimeError(scan_h5.keys())

        # get diffraction data/counts
        diff_data_group = scan_h5['raw']

        # loop through the Logs
        counts = diff_data_group['counts'].value
        
        # instrument
        instrument_group = scan_h5['instrument']
        two_theta = instrument_group['2theta'].value
        
        print (counts)
        print (type(counts))
        
        print (two_theta)
        print (type(two_theta))
        
        """
        [0 0 0 ..., 0 0 0]
        <type 'numpy.ndarray'>
        35.0
        <type 'numpy.float64'>
        """
        
        return counts, two_theta
        
        
        
file_name = 'LaB6_10kev_35deg-00004_Rotated_TIF.h5'


count_vec, two_theta = load_raw_measurement_data(file_name)

vec_x = numpy.zeros(count_vec.shape)

raw = CreateWorkspace(DataX=vec_x, DataY=count_vec, DataE=numpy.sqrt(count_vec), NSpec=1)
raw = Transpose(raw)

raw_data_ws_name = 'raw'
AddSampleLog(Workspace=raw_data_ws_name, LogName='2theta', LogText='{}'.format(-two_theta),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::arm', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltax', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')

AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::deltay', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='meter',
                     NumberType='Double')
                     
AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::flip', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::roty', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

AddSampleLog(Workspace=raw_data_ws_name, LogName='cal::spin', LogText='{}'.format(0),
                     LogType='Number Series', LogUnit='degree',
                     NumberType='Double')

LoadInstrument('raw', Filename='tests/testdata/XRay_Definition_2K.xml', RewriteSpectraMap=True)

