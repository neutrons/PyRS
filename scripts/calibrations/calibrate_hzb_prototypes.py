#!/user/bin/python
# Template to calibrate HZB data with calibration algorithm 1
import sys
import os
from pyrs.utilities import rs_project_file
from pyrs.core import pyrscore
from pyrs.utilities import script_helper


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
                     ('instrument', None, 'instrument', str, None, False, 'Path to instrument file'),
                     ('output', 'o', 'outputfile', str, None, True, 'Output calibration in JSON format'),
                     ('binsize', 'b', 'binsize', float, 0.01, False, '2theta step')]

    try:
        param_dict = script_helper.process_arguments(argv, args_def_dict)
    except RuntimeError as run_err:
        print('Failed to parse inputs due to {}'.format(run_err))
        return

    # In case of help
    if param_dict is None:
        return

    # Create calibration control
    calib_controller = pyrscore.PyRsCore()

    # Load data
    project_name = 'calibration'
    calib_controller.load_hidra_project(param_dict['inputfile'], project_name=project_name)

    # Reduce data
    calib_controller.reduce_diffraction_data(project_name, two_theta_step=args_def_dict['binsize'],
                                             pyrs_engine=True)

    # Export reduction data
    calib_controller.export_diffraction_data(project_name, param_dict['inputfile'])

    # Calibration init: import ROI/Mask files
    mask_file_list = parse_mask_files(param_dict['masks'])
    if len(mask_file_list) < 2:
        print('For X-ray case, user must specify at least 2 masks')
        sys.exit(-1)

    # TODO - to be continued

    return


if __name__ == '__main__':
    main(sys.argv)
