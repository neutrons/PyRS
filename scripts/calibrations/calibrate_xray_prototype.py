#!/user/bin/python
# Template to calibrate X-ray data with calibration algorithm 1
import sys
import os
from pyrs.core import pyrscore
from pyrs.utilities import script_helper


def parse_mask_files(mask_file_name):
    """
    Parse mask file names from an ASCII file
    :param mask_file_name:
    :return:
    """
    if mask_file_name is None:
        return list()

    ifile = open(mask_file_name, 'r')
    lines = ifile.readlines()
    ifile = ifile.close()

    # Parse
    mask_file_list = list()
    for raw_line in lines:
        line = raw_line.strip()
        # remove , and \t
        line = line.replace(',', ' ')
        line = line.replace('\t', ' ')
        terms = line.split()
        for file_name in terms:
            mask_file_list.append(file_name)

    return mask_file_list


def set_mask_files(masks_list_file_name):
    """
    Read an ASCII file containing a list of masks
    :param masks_list_file_name:
    :return:
    """
    temp_list = ['Chi_0_Mask.xml', 'Chi_10_Mask.xml',
                 'Chi_20_Mask.xml', 'Chi_30_Mask.xml', 'NegZ_Mask.xml']
    mask_xml_list = [os.path.join('tests/testdata/masks', xml_name) for xml_name in temp_list]

    return mask_xml_list


def main(argv):
    """
    Main
    :param argv:
    :return:
    """
    #
    # Required: HIDRA
    # Optional: Instrument file
    # Optional: mask files
    # Optional: bins

    # long-name, short-name, target name, type, default value, is mandatory, documen
    args_def_dict = [('input', 'i', 'inputfile', str, None, True, 'Input HIDRA project file'),
                     ('masks', 'm', 'masksfiles', str, None, False,
                      'Path to an ASCI file containing list of path to mask files, separated by ":", ", " or "\n"'),
                     ('instrument', None, 'instrument', str, None, False, 'Path to instrument file'),
                     ('output', 'o', 'outputfile', str, None, True, 'Output calibration in JSON format'),
                     ('binsize', 'b', 'binsize', float, 0.01, False, '2theta step')]

    try:
        param_dict = script_helper.process_arguments(argv, args_def_dict)
    except RuntimeError as run_err:
        print ('Failed to parse inputs due to {}'.format(run_err))
        return

    # In case of help
    if param_dict is None:
        return

    # Create calibration control
    calib_controller = pyrscore.PyRsCore()

    # Load data
    project_name = 'calibration xray'
    calib_controller.load_hidra_project(param_dict['inputfile'], project_name=project_name)

    # Reduce data
    calib_controller.reduce_diffraction_data(project_name, two_theta_step=args_def_dict['binsize'],
                                             pyrs_engine=True)

    # Export reduction data
    calib_controller.export_diffraction_data(project_name, param_dict['inputfile'])

    # Calibration init: import ROI/Mask files
    mask_file_list = parse_mask_files(param_dict['masks'])
    if len(mask_file_list) < 2:
        print ('For X-ray case, user must specify at least 2 masks')
        sys.exit(-1)

    # TODO - to be continued

    return


if __name__ == '__main__':
    print (sys.argv)
    main(sys.argv)
