#!/usr/bin/python
""""
Convert the old HZB raw data to test
1. raw counts writing and writing
2. PyRS reduction
3. PyRS calibration

Note: most of the methods to parse HZB data are copied from script convert_hzb_data.py

*How to run*
1. Add PyRS path to python path (refer to pyrsdev.sh)
2. Run this script
"""
import numpy
import os
from pyrs.core.instrument_geometry import HidraSetup
from pyrs.projectfile import HidraConstants, HidraProjectFile, HidraProjectFileMode
from pyrs.utilities import file_util


def parse_hzb_tiff(tiff_file_name):
    """
    Parse HZB TIFF (image) data to numpy array (1D)
    :param tiff_file_name:
    :return: (1) 1D array (column major, from lower left corner) (2) (num_row, num_cols)
    """
    # is it rotated?
    # rotated
    counts_array = file_util.load_rgb_tif(tiff_file_name, True)

    return counts_array


def import_hzb_summary(summary_excel):
    """
    import and parse HZB summary file in EXCEL format
    [    u'E3 file',        u'Tiff',  u'Tiff_Index',       u'Index',
               u'2th',         u'mon',       u'2th.1',  u'Unnamed: 7',
                u'L2',        u'ADET', u'Unnamed: 10', u'Unnamed: 11',
       u'Unnamed: 12', u'Unnamed: 13',        u'SDET'],
    # ['Tiff'][i] = E3-Y2O3 42-50
    # ['Tiff Index] = 1, 2, 3, ...
    # 2th = 2th.1
    # L2: unit == mm
    # tif_name = '{}_{:05}.tiff'.format(df['Tiff'][0], int(df['Tiff_Index'][0]))
    :param summary_excel: experiment summary file in Excel format
    :return:
    """
    # load data
    summary_pandas_data = file_util.load_excel_file(summary_excel)

    # get list of interested items
    scan_index_array = numpy.array(summary_pandas_data['Index'])
    two_theta_array = numpy.array(summary_pandas_data['2th'])
    tiff_header_array = numpy.array(summary_pandas_data['Tiff'])
    tiff_order_array = numpy.array(summary_pandas_data['Tiff_Index'])
    monitor_array = numpy.array(summary_pandas_data['mon'])
    l2_array = numpy.array(summary_pandas_data['L2']) * 1.E-3  # convert to meter

    # combine to files
    tiff_files = list()
    for item_index in range(len(tiff_header_array)):
        tiff_header_i = tiff_header_array[item_index]
        tiff_order_i = tiff_order_array[item_index]
        tiff_file_i = '{}_{:05}.tiff'.format(tiff_header_i, int(tiff_order_i))
        tiff_files.append(tiff_file_i)
    # END-FOR

    # create a dictionary of dictionary for the information
    summary_dict = dict()
    for item_index in range(len(tiff_files)):
        scan_i_dict = {HidraConstants.TWO_THETA: two_theta_array[item_index],
                       'Tiff': tiff_files[item_index],
                       'Monitor': monitor_array[item_index],
                       'L2': l2_array[item_index]}
        if scan_index_array[item_index] in summary_dict:
            raise RuntimeError('Experiment summary file {} contains 2 entries with same (scan) Index {}'
                               ''.format(summary_excel, scan_index_array[item_index]))
        summary_dict[scan_index_array[item_index]] = scan_i_dict
    # END-FOR

    return summary_dict, {HidraConstants.SUB_RUNS: scan_index_array,
                          HidraConstants.TWO_THETA: two_theta_array,
                          'Monitor': monitor_array, 'L2': l2_array}


def generate_hzb_instrument():
    """
    Create an instrument setup for HZB
    :return:
    """
    from pyrs.core import instrument_geometry

    wavelength = 1.222
    arm_length = 1.13268
    x = 0.001171875
    y = 0.001171875
    detector = instrument_geometry.AnglerCameraDetectorGeometry(num_rows=256, num_columns=256,
                                                                pixel_size_x=x,
                                                                pixel_size_y=y,
                                                                arm_length=arm_length,
                                                                calibrated=False)

    hzb = HidraSetup(l1=1.0, detector_setup=detector)  # single wave length
    hzb.set_single_wavelength(wavelength)

    return hzb


def main():
    """
    Main method to do the conversion
    :param argv:
    :return:
    """
    hzb_summary_name = '/SNS/users/wzz/Projects/HB2B/Quasi_HB2B_Calibration/calibration.xlsx'
    output_file_name = 'tests/testdata/HZB_Raw_Project.h5'
    exp_data_dir = '/SNS/users/wzz/Projects/HB2B/Quasi_HB2B_Calibration/'  # raw HZB TIFF exp data directory

    # parse EXCEL spread sheet
    exp_summary_dict, exp_logs_dict = import_hzb_summary(hzb_summary_name)

    # start project file
    project_file = HidraProjectFile(output_file_name, mode=HidraProjectFileMode.OVERWRITE)

    # parse and add counts
    sub_run_list = exp_summary_dict.keys()
    for sub_run_i in sorted(sub_run_list):
        tiff_name = exp_summary_dict[sub_run_i]['Tiff']
        counts_array = parse_hzb_tiff(os.path.join(exp_data_dir, tiff_name))
        # print (counts_array.min(), counts_array.max(), (numpy.where(counts_array > 0.5)[0]).shape)
        project_file.append_raw_counts(sub_run_i, counts_array)
    # END-FOR

    # add sample log data & sub runs
    for log_name in [HidraConstants.SUB_RUNS, HidraConstants.TWO_THETA, 'Monitor', 'L2']:
        project_file.append_experiment_log(log_name, exp_logs_dict[log_name])
    # END-FOR
    # project_file.add_experiment_information('sub-run', sub_run_list)

    # add instrument information
    hzb_instrument_setup = generate_hzb_instrument()
    project_file.write_instrument_geometry(hzb_instrument_setup)

    # save
    project_file.close()


if __name__ == '__main__':
    main()
