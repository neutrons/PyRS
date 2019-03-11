# Zoo of methods to work with file properties
import time
import os
import checkdatatypes
import platform


# Zoo of methods to work with raw data input and output of processed data



def export_md_array_hdf5(md_array, sliced_dir_list, file_name):
    """
    export 2D data from
    :param md_array:
    :param sliced_dir_list: None or integer last
    :param file_name:
    :return:
    """
    checkdatatypes.check_numpy_arrays('2D numpy array to export', [md_array], 2, False)
    checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True)

    if sliced_dir_list is not None:
        # delete selected columns: axis=1
        checkdatatypes.check_list('Sliced directions', sliced_dir_list)
        try:
            md_array = numpy.delete(md_array, sliced_dir_list, 1)  # axis = 1
        except ValueError as val_err:
            raise RuntimeError('Unable to delete column {} of input numpy 2D array due to {}'
                               ''.format(sliced_dir_list, val_err))
    # END-IF

    # write out
    out_h5_file = h5py.File(file_name, 'w')
    out_h5_file.create_dataset('Sliced-{}'.format(sliced_dir_list), data=md_array)
    out_h5_file.close()

    return



def load_data_from_tif(raw_tiff_name, pixel_size=2048, rotate=True):
    """
    Load data from TIFF
    :param raw_tiff_name:
    :param pixel_size
    :param rotate:
    :return:
    """
    from skimage import io, exposure, img_as_uint, img_as_float
    from PIL import Image
    import numpy as np
    import pylab as plt

    ImageData = Image.open(raw_tiff_name)
    # im = img_as_uint(np.array(ImageData))
    io.use_plugin('freeimage')
    image_2d_data = np.array(ImageData, dtype=np.int32)
    print (image_2d_data.shape, type(image_2d_data), image_2d_data.min(), image_2d_data.max())
    # image_2d_data.astype(np.uint32)
    image_2d_data.astype(np.float64)
    if rotate:
        image_2d_data = image_2d_data.transpose()

    # Merge data if required
    if pixel_size == 1024:
        counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + image_2d_data[1::2, ::2] + image_2d_data[1::2, 1::2]
        pixel_type = '1K'
        # print (DataR.shape, type(DataR))
    else:
        # No merge
        counts_vec = image_2d_data
        pixel_type = '2K'

    counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
    print (counts_vec.min())

    data_ws_name = os.path.basename(raw_tiff_name).split('.')[0] + '_{}'.format(pixel_type)
    CreateWorkspace(DataX=np.zeros((pixel_size**2,)), DataY=counts_vec, DataE=np.sqrt(counts_vec), NSpec=pixel_size**2,
                    OutputWorkspace=data_ws_name, VerticalAxisUnit='SpectraNumber')

    return data_ws_name, counts_vec


def save_mantid_nexus(workspace_name, file_name, title=''):
    """
    save workspace to NeXus for Mantid to import
    :param workspace_name:
    :param file_name:
    :param title:
    :return:
    """
    # check input
    checkdatatypes.check_file_name(file_name, check_exist=False,
                                   check_writable=True, is_dir=False)
    checkdatatypes.check_string_variable('Workspace title', title)

    # check workspace
    checkdatatypes.check_string_variable('Workspace name', workspace_name)
    if AnalysisDataService.doesExist(workspace_name):
        SaveNexusProcessed(InputWorkspace=workspace_name,
                           Filename=file_name,
                           Title=title)
    else:
        raise RuntimeError('Workspace {0} does not exist in Analysis data service. Available '
                           'workspaces are {1}.'
                           ''.format(workspace_name, AnalysisDataService.getObjectNames()))

    # END-IF-ELSE

    return



def check_creation_date(file_name):
    """
    check the create date (year, month, date) for a file
    :except RuntimeError: if the file does not exist
    :param file_name:
    :return:
    """
    checkdatatypes.check_file_name(file_name, check_exist=True)

    # get the creation date in float (epoch time)
    if platform.system() == 'Windows':
        # windows not tested
        epoch_time = os.path.getctime(file_name)
    else:
        # mac osx/linux
        stat = os.stat(file_name)
        try:
            epoch_time = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            epoch_time = stat.st_mtime
        # END-TRY
    # END-IF-ELSE

    # convert epoch time to a string as YYYY-MM-DD
    file_create_time = time.localtime(epoch_time)
    file_create_time_str = time.strftime('%Y-%m-%d', file_create_time)

    return file_create_time_str


def get_temp_directory():
    """
    get a temporary directory to write files
    :return:
    """
    # current workspace first
    temp_dir = os.getcwd()
    if os.access(temp_dir, os.W_OK):
        return temp_dir

    # /tmp/ second
    temp_dir = '/tmp/'
    if os.path.exists(temp_dir):
        return temp_dir

    # last solution: home directory
    temp_dir = os.path.expanduser('~')

    return temp_dir



# testing
print (check_creation_date('__init__.py'))
