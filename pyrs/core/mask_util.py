# Zoo of methods to work with masks including Mantid and PyRS special
import numpy as np
import h5py
from pyrs.utilities import checkdatatypes

"""
Note:

- Mask vector
1. numpy 1D array
2. mask_vec[i] = 1  : ROI
3. mask_vec[i] = 0  : Mask
4. mask_vec * data_vec:   masking
5. mask_vec2 + mask_vec2: OR  (AND for ROI)
6. mask_vec1 * mask_vec2: AND
"""


def load_mantid_mask(pixel_number, mantid_mask_xml, is_mask):
    """ Load Mantid mask file in XML format
    Assumption: PixelID (detector ID) starts from 0 and there is NO gap
    :param mantid_mask_xml:
    :param pixel_number: total pixel number
    :return: a vector
    """
    checkdatatypes.check_file_name(mantid_mask_xml, True, False, False, 'Mantid XML mask file')
    checkdatatypes.check_int_variable('(Total) pixel number', pixel_number, (1024**2, 2048**2+1))

    # load file to lines
    mask_file = open(mantid_mask_xml, 'r')
    mask_lines = mask_file.readlines()
    mask_file.close()

    # get detector ID range line
    det_id_line = None
    for line in mask_lines:
        if line.count('<detid') > 0:
            det_id_line = line.strip()
            break
    # END-FOR

    if det_id_line is None:
        raise RuntimeError('Mask file {} does not have masked detector IDs'.format(mantid_mask_xml))

    # parse
    masked_det_pair_list = det_id_line.split('>')[1].split('<')[0].strip().split(',')
    # print ('[DB...BAT] Masked detectors range: {}'.format(masked_det_pair_list))

    # create vector with 1 (for not masking)
    masking_array = np.zeros((pixel_number,), 'float')
    if is_mask:
        # is given string are mask then default is not masked
        masking_array += 1.
    # is ROI default = 0

    masked_specs = 0
    for masked_det_pair in masked_det_pair_list:
        # get range
        terms = masked_det_pair.split('-')
        start_detid = int(terms[0])
        end_detid = int(terms[1])
        # check range
        if end_detid >= pixel_number:
            raise RuntimeError('Detector ID {} is out of range of given detector size {}'
                               ''.format(end_detid, pixel_number))
        # mask or ROI
        if is_mask:
            masking_array[start_detid:end_detid+1] = 0.
        else:
            masking_array[start_detid:end_detid+1] = 1.
        # stat
        masked_specs += end_detid - start_detid + 1
    # END-FOR

    print ('[DB...CHECK] Masked spectra = {}, Sum of masking array = {}'
           ''.format(masked_specs, sum(masking_array)))

    return masking_array


def load_pyrs_mask(mask_h5):
    """ Load an HDF5 mask file
    :param mask_h5:
    :return: 3-tuple (mask vector, two theta, user note)
    """
    checkdatatypes.check_file_name(mask_h5, True, False, False, 'PyRS mask file (hdf5) to load')

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


def save_mantid_mask(mask_vec,  h5_name, two_theta, note):
    """
    Save a mask vector to
    :param mask_vec:
    :param h5_name:
    :param two_theta:
    :param note:
    :return:
    """
    checkdatatypes.check_numpy_arrays('Mask vector', [mask_vec], dimension=1, check_same_shape=False)
    checkdatatypes.check_file_name(h5_name, False, True, False, 'PyRS masking file to export to')
    if two_theta is not None:
        checkdatatypes.check_float_variable('2-theta', two_theta, (-360., 360))
    if note is not None:
        checkdatatypes.check_string_variable('Mask note', note, None)

    # create file
    mask_file = h5py.File(h5_name, 'w')
    # add data set
    mask_data_set = mask_file.create_dataset('mask', data=mask_vec)
    # add attributes
    if two_theta:
        mask_data_set.attrs['2theta'] = two_theta  # '{}'.format(two_theta)
    if note:
        mask_data_set.attrs['note'] = note
    # close file
    mask_file.close()

    return
