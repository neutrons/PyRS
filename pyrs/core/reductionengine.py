# Reduction engine including slicing
import os
import numpy as np
from pyrs.utilities import checkdatatypes
from pyrs.utilities import hb2b_utilities


from pyrs.utilities import file_utilities
from pyrs.core import scandataio


class HB2BReductionManager(object):
    """
    A data reduction manager of HB2B

    1. It can work with both PyHB2BReduction and MantidHB2BReduction seamlessly
    2. It can always compare the results between 2 reduction engines
    3. It shall provide an API to calibration optimization
    """
    def __init__(self):
        """ initialization
        """
        # calibration manager
        self._calibration_manager = hb2b_utilities.CalibrationManager()

        # number of bins
        self._num_bins = 1000

        return

    def load_data(self, data_file_name, target_dimension=None):
        """
        Load data set and
        - determine the instrument size (for PyHB2BReduction and initialize the right one if not created)
        :param data_file_name:
        :return:
        """
        file_type = data_file_name.split('.')[-1].lower()

        if file_type == 'tif' or file_type == 'tiff':
            self._laod_tif_image(data_file_name, target_dimension)

        return

    def _load_tif_image(self, image_file, target_dimension):
        """
        Load a TIFF file and resize to target dimension.
        Save the data in a 2D array (raw and for plotting) and 1D (flattened for reduction)
        :param image_file:
        :param target_dimension:
        :return:
        """
        from skimage import io, exposure, img_as_uint, img_as_float
        from PIL import Image

        ImageData = Image.open(ima)
        # im = img_as_uint(np.array(ImageData))
        io.use_plugin('freeimage')
        image_2d_data = np.array(ImageData, dtype=np.int32)
        print(image_2d_data.shape, type(image_2d_data), image_2d_data.min(), image_2d_data.max())
        # image_2d_data.astype(np.uint32)
        image_2d_data.astype(np.float64)
        if rotate:
            image_2d_data = image_2d_data.transpose()

        # Merge data if required
        if pixel_size == 1024:
            counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + image_2d_data[1::2, ::2] + image_2d_data[
                                                                                                         1::2, 1::2]
            pixel_type = '1K'
            # print (DataR.shape, type(DataR))
        else:
            # No merge
            counts_vec = image_2d_data
            pixel_type = '2K'

        counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
        print(counts_vec.min())

        data_ws_name = os.path.basename(raw_tiff_name).split('.')[0] + '_{}'.format(pixel_type)
        CreateWorkspace(DataX=np.zeros((pixel_size ** 2,)), DataY=counts_vec, DataE=np.sqrt(counts_vec),
                        NSpec=pixel_size ** 2,
                        OutputWorkspace=data_ws_name, VerticalAxisUnit='SpectraNumber')

        return data_ws_name, counts_vec


def get_log_value(workspace, log_name):
    """
    get log value
    :param workspace:
    :param log_name:
    :return:
    """
    # TODO - 20181204 - Implement!

    return blabla


def set_log_value(workspace, log_name, log_value):
    """
    set a value to a workspace's sample logs
    :param workspace:
    :param log_name:
    :param log_value:
    :return:
    """
    # TODO - 20181204 - Implement!

    return


def retrieve_workspace(ws_name, must_be_event=True):
    """
    retrieve workspace
    :param ws_name:
    :param must_be_event: throw if not event workspace if this is specified
    :return:
    """
    checkdatatypes.check_string_variable('Workspace name', ws_name)
    if ws_name == '':
        raise RuntimeError('Workspace name cannot be an empty string')

    if not ADS.doesExist(ws_name):
        raise RuntimeError('Worksapce {} does not exist in Mantid ADS'.format(ws_name))

    return ADS.retrieve(ws_name)
