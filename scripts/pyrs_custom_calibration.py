import numpy as np
from pyrs.projectfile import HidraProjectFile  # type: ignore
from pyrs.calibration.peakfit_calibration import PeakFitCalibration
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.instrument_geometry import DENEXDetectorGeometry


def _load_diff_data(project_file_name):
    test_ws = HidraWorkspace('test_powder_pattern')
    test_project = HidraProjectFile(project_file_name)
    test_ws.load_hidra_project(test_project, load_raw_counts=True, load_reduced_diffraction=False)

    return test_ws


def _run_calibration(calibrator):
    """
    """

    calibrator.singlepeak = False
    calibrator.CalibrateGeometry(ConstrainPosition=True)
    calibrator.print_calibration(print_to_screen=False, refine_step="geometry")

    return calibrator


def _parse_powder_line(material):
    if material == 'LaB6':
        dspace = np.array([4.15692, 2.93939, 2.4, 2.07846, 1.85903, 1.69706, 1.46969, 1.38564, 1.31453,
                           1.25336, 1.2, 1.15292, 1.11098, 1.03923, 1.0082, 0.9798, 0.95366, 0.92952,
                           0.90711, 0.88626, 0.84853, 0.83138, 0.81524, 0.8, 0.77192])
    else:
        dspace = np.array([3.124042, 2.705500, 1.913077, 1.631478, 1.562021, 1.352750, 1.241368, 1.209936,
                           1.104516, 1.041347, 1.041347, 0.956539, 0.914626, 0.901833, 0.901833, 0.855554])

    return dspace


if __name__ == '__main__':
    import sys
    import json

    with open(sys.argv[1], 'r') as json_input:
        try:
            calibration_inputs = json.load(json_input)
        except ValueError:
            raise RuntimeError('template.json was created as an example input')

    POWDER_RUN = _load_diff_data(calibration_inputs['file'])

    detector_ob = DENEXDetectorGeometry(calibration_inputs['num_pix_x'], calibration_inputs['num_pix_y'],
                                        calibration_inputs['pix_size_x'], calibration_inputs['pix_size_y'],
                                        calibration_inputs['det_distance'], False)

    calibrator = PeakFitCalibration(powder_engine=POWDER_RUN, pin_engine=None,
                                    powder_lines=_parse_powder_line(calibration_inputs['calibrant']),
                                    single_material=True, hb2b_inst=detector_ob,
                                    wavelength=calibration_inputs['wavelength'])

    # calibrator.plot_res = True
    calibrator.bins = 1800
    calibrator.inital_width = 1e-2
    calibrator.inital_AMP = 1e4
    calibrator.min_tth = 1.25
    calibrator.max_tth = 4.15
    calibrator.eta_slices = 15

    # calibrator.singleEval()
    # calibrator.CalibrateGeometry(ConstrainPosition=True)
    # calibrator.calibrate_shiftx(ConstrainPosition=True)
    # calibrator.calibrate_distance(ConstrainPosition=True)
    # calibrator.calibrate_shifty(ConstrainPosition=True)
    # calibrator.CalibrateShift(ConstrainPosition=True)
    # _run_calibration(calibrator)

    diff_data = calibrator.get_2d_reduced_data()
    np.savetxt('diff_data', diff_data)
    # # if SAVE_CALIB:
    # #     datatime = time.strftime('%Y-%m-%dT%H-%M', time.localtime())
    # #     if HFIR_CYCLE is not None:
    # #         FolderName = '/HFIR/HB2B/shared/CALIBRATION/cycle{}'.format(HFIR_CYCLE)
    # #         if not os.path.exists(FolderName):
    # #             os.makedirs(FolderName)
    # #         CalibName = '/HFIR/HB2B/shared/CALIBRATION/cycle{}/HB2B_{}_{}.json'.format(HFIR_CYCLE, mono, datatime)
    # #         calibrator.write_calibration(CalibName)

    # #     CalibName = '/HFIR/HB2B/shared/CALIBRATION/HB2B_{}_{}.json'.format(mono, datatime)
    # #     calibrator.write_calibration(CalibName)
    # #     print(calibrator.refinement_summary)
    # # else:
    # # calibrator.print_calibration()
    # print(calibrator.refinement_summary)
