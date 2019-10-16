#!/usr/bin/python
# Convert HZB data to standard transformed-HDF5 format considering all the sample log values and instrument information
# This may be executed only once and moved out of build
import numpy
import os
from pyrs.utilities import file_util
from pyrs.utilities import rs_project_file


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
        scan_i_dict = {'2theta': two_theta_array[item_index],
                       'Tiff': tiff_files[item_index],
                       'Monitor': monitor_array[item_index],
                       'L2': l2_array[item_index]}
        if scan_index_array[item_index] in summary_dict:
            raise RuntimeError('Experiment summary file {} contains 2 entries with same (scan) Index {}'
                               ''.format(summary_excel, scan_index_array[item_index]))
        summary_dict[scan_index_array[item_index]] = scan_i_dict
    # END-FOR

    return summary_dict, {'Scan Index': scan_index_array, '2Theta': two_theta_array,
                          'Monitor': monitor_array, 'L2': l2_array}


def main(argv):
    """
    main for the workflow to create the HDF5
    :param argv:
    :return:
    """
    # process inputs ...
    exp_summary_excel = argv[1]
    exp_data_dir = argv[2]
    project_file_name = argv[3]

    # parse EXCEL spread sheet
    exp_summary_dict, exp_logs_dict = import_hzb_summary(exp_summary_excel)

    # start project file
    project_file = rs_project_file.HydraProjectFile(project_file_name,
                                                    mode=rs_project_file.HydraProjectFileMode.OVERWRITE)

    # add sample log data
    for log_name in ['Scan Index', '2Theta', 'Monitor', 'L2']:
        project_file.add_experiment_log(log_name, exp_logs_dict[log_name])
    # END-FOR

    # parse and add counts
    sub_run_list = exp_summary_dict.keys()
    for sub_run_i in sorted(sub_run_list):
        tiff_name = exp_summary_dict[sub_run_i]['Tiff']
        counts_array = parse_hzb_tiff(os.path.join(exp_data_dir, tiff_name))
        print(counts_array.min(), counts_array.max(), (numpy.where(counts_array > 0.5)[0]).shape)
        project_file.add_raw_counts(sub_run_i, counts_array)
    # END-FOR

    # save
    project_file.close()

    return


if __name__ == '__main__':
    main(['Whatever',
          '/SNS/users/wzz/Projects/HB2B/Quasi_HB2B_Calibration/calibration.xlsx',
          '/SNS/users/wzz/Projects/HB2B/Quasi_HB2B_Calibration/',
          'tests/testdata/hzb/hzb_calibration.hdf5'])
