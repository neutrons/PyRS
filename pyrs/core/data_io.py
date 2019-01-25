# Zoo of methods to work with raw data input and output of processed data


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


def load_data_from_tif_ver2():
    """ Use matplotlib import TIFF
    :return:
    """
    # TODO - NIGHT - Implement version 2 as the main TIFF reader for test on analysis cluster
    data = matplotlib.image.imread(tiff_name)
    data.astype(np.int32)

    return


def load_data_from_bin(bin_file_name):
    """
    """
    ws_name = os.path.basename(bin_file_name).split('.')[0]
    LoadSpiceXML2DDet(Filename=bin_file_name, OutputWorkspace=ws_name, LoadInstrument=False)

    # get vector of counts
    counts_ws = Transpose(InputWorkspace=ws_name, OutputWorkspace='temp')
    count_vec = counts_ws.readY(0)

    return ws_name, count_vec


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
