#!/usr/bin/python
""""
Convert the synchrotron raw data to test
1. raw counts writing and writing
2. PyRS reduction
3. PyRS calibration

Note: most of the methods to parse HZB data are copied from script pyrscalibration.py
"""
from pyrs.utilities.rs_project_file import HidraConstants, HidraProjectFile, HidraProjectFileMode
import numpy
from skimage import io
from PIL import Image
import numpy as np


def load_data_from_tif(raw_tiff_name, pixel_size=2048, rotate=True):
    """
    Load data from TIFF
    :param raw_tiff_name:
    :param pixel_size
    :param rotate:
    :return:
    """
    ImageData = Image.open(raw_tiff_name)
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
        counts_vec = image_2d_data[::2, ::2] + image_2d_data[::2, 1::2] + \
            image_2d_data[1::2, ::2] + image_2d_data[1::2, 1::2]
        pixel_type = '1K'
        # print (DataR.shape, type(DataR))
    else:
        # No merge
        counts_vec = image_2d_data
        pixel_type = '2K'

    counts_vec = counts_vec.reshape((pixel_size * pixel_size,))
    print('Minimum counts (on pixels) = ', counts_vec.min())

    return counts_vec, pixel_type


def generate_xray_instrument():
    """
    Create an instrument setup for HZB
    :return:
    """
    from pyrs.core import instrument_geometry

    wavelength = 1.239  # A
    x = 0.00020  # meter
    y = 0.00020  # meter
    detector = instrument_geometry.AnglerCameraDetectorGeometry(num_rows=2048, num_columns=2048,
                                                                pixel_size_x=x,
                                                                pixel_size_y=y,
                                                                arm_length=0.416,  # meter
                                                                calibrated=False)

    hzb = instrument_geometry.HydraSetup(detector_setup=detector)  # single wave length
    hzb.set_single_wavelength(wavelength)

    return hzb


def main():
    """
    Main method to do the conversion
    :param argv:
    :return:
    """
    raw_tif_name = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif'
    output_file_name = 'tests/testdata/Hidra_XRay_LaB6_10kev_35deg.h5'

    # Load raw data from image
    xray_count_vec, xray_det_type = load_data_from_tif(raw_tif_name)

    # start project file
    project_file = HidraProjectFile(output_file_name, mode=HidraProjectFileMode.OVERWRITE)

    # add comments
    project_file.set_information({'Raw Data File': raw_tif_name, 'Detector Type': xray_det_type})

    # parse and add counts: only 1 sub run
    project_file.add_raw_counts(sub_run_number=1, counts_array=xray_count_vec)

    # add sample log data & sub runs
    # for log_name in ['sub-run', '2Theta']
    project_file.add_experiment_log(HidraConstants.SUB_RUNS, numpy.array([1]))
    project_file.add_experiment_log(HidraConstants.TWO_THETA, numpy.array([35.]))

    # add instrument information
    instrument_setup = generate_xray_instrument()
    project_file.set_instrument_geometry(instrument_setup)

    # save
    project_file.close()

    return


if __name__ == '__main__':
    main()
