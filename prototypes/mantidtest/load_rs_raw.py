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


# From mask_util
def load_pyrs_mask(mask_h5):
    """ Load an HDF5 mask file
    :param mask_h5:
    :return: 3-tuple (mask vector, two theta, user note)
    """
    # open
    mask_file = h5py.File(mask_h5, 'r')

    # check
    if 'mask' not in mask_file:
        raise RuntimeError('{} does not have entry "mask"'.format(mask_h5))

    # get mask array
    mask_entry = mask_file['mask']
    mask_vec = mask_entry[()]

    if '2theta' in mask_entry.attrs:
        two_theta = mask_entry.attrs['2theta']   # numpy.float64
        # print ('2theta = {} of type {}'.format(two_theta, type(two_theta)))
    else:
        two_theta = None   # not 2theta-dependant mask/ROI

    if 'note' in mask_entry.attrs:
        note = mask_entry.attrs['note']
    else:
        note = None

    return mask_vec, two_theta, note


file_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5'
mask_name = 'tests/testdata/masks/Chi_Neg30.hdf5'

# Load raw and create workspace
count_vec, two_theta = load_raw_measurement_data(file_name)
count_vec = count_vec.astype('float64')

# Load mask
mask_vec, mask_two_theta, note = load_pyrs_mask(mask_name)

# [Plan 1] Mask from counts vector
count_vec *= mask_vec

vec_x = numpy.zeros(count_vec.shape)
raw = CreateWorkspace(DataX=vec_x, DataY=count_vec, DataE=numpy.sqrt(count_vec), NSpec=1)
raw = Transpose(raw)

# Load instrument
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

# [Plan 2] create workspace for mask
mask_ws = CreateWorkspace(DataX=vec_x, DataY=mask_vec, NSpec=1)
mask_ws = Transpose(mask_ws)
# raw_masked_ws = raw * mask_ws
