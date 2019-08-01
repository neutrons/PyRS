
# From reduction_manager.py
def load_data(self, data_file_name, sub_run=None, target_dimension=None, load_to_workspace=True):
    """
    Load data set and
    - determine the instrument size (for PyHB2BReduction and initialize the right one if not created)
    :param data_file_name:
    :param sub_run: integer for sun run number
    :param target_dimension: if TIFF, target dimension will be used to bin the data
    :param load_to_workspace: if TIFF, option to create a workspace
    :return: data ID to look up, 2-theta (None if NOT recorded)
    """
    # check inputs
    checkdatatypes.check_file_name(data_file_name, True, False, False, 'Data file to load')

    # check file type
    if data_file_name.endswith('.nxs.h5'):
        file_type = 'nxs.h5'
    else:
        file_type = data_file_name.split('.')[-1].lower()

    # load
    two_theta = None

    if file_type == 'tif' or file_type == 'tiff':
        # TIFF
        data_id = self._load_tif_image(data_file_name, target_dimension, rotate=True,
                                       load_to_workspace=load_to_workspace)

    elif file_type == 'bin':
        # SPICE binary
        data_id = self._load_spice_binary(data_file_name)

    elif file_type == 'hdf5' or file_type == 'h5':
        # PyRS HDF5
        data_id, two_theta = self._load_pyrs_h5(data_file_name, sub_run, load_to_workspace)

    elif file_type == 'nxs.h5' or file_type == 'nxs':
        # Event NeXus
        data_id = self._load_nexus(data_file_name)

    else:
        # not supported
        raise RuntimeError('File type {} from input {} is not supported.'.format(file_type, data_file_name))

    self._last_loaded_data_id = data_id

    return data_id, two_theta
