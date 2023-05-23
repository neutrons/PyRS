# This module is to calculate Pole Figure
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int
import numpy as np
from typing import Tuple


class PoleFigureCalculator:
    """
    A calculator for Pole Figure.
    It has the memory and result for the last time it is called to calculate
    """

    def __init__(self):
        """
        initialization
        """
        # initialize class instances
        self._peak_info_dict = dict()   # key: detector ID, value: dict; key: scan log index, value: dict
        self._pole_figure_dict = dict()  # key: detector ID, value: 2-tuple.  scan log indexes (list), 2D array

        # flag
        self._cal_successful = False

    def add_input_data_set(self, peak_id: int, log_dict: dict) -> None:
        """ set peak intensity log and experiment logs that are required by pole figure calculation
        :param peak_id
        :param log_dict: dictionary (key = scan log index (int), value = dictionary (log name, log value))
        :return:
        """

        peak_id = to_int('Detector ID', peak_id, min_value=0)
        checkdatatypes.check_dict('Log values for pole figure', log_dict)

        log_names = list(log_dict.keys())
        checkdatatypes.check_list('Pole figure motor names', log_names,
                                  ['chi', 'phi', 'omega', 'eta', 'center', 'intensity'])

        # check inputs
        if peak_id in self._peak_info_dict:
            # set
            for entry in ['chi', 'omega', 'eta', 'center', 'intensity', 'phi']:

                self._peak_info_dict[peak_id][entry] = np.concatenate((self._peak_info_dict[peak_id][entry],
                                                                       log_dict[entry]), axis=0)

        else:
            # set
            self._peak_info_dict[peak_id] = log_dict

    def calculate_pole_figure(self, peak_id_list=None):
        """ Calculate pole figures
        :param det_id_list:
        :return:
        """
        # check input
        if peak_id_list is None:
            peak_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs to calculate pole figure', peak_id_list,
                                      self.get_detector_ids())
        # END-IF

        for peak_id in peak_id_list:
            # calculator by each detector
            peak_info_dict = self._peak_info_dict[peak_id]

            # construct the output
            num_pts = len(peak_info_dict['chi'])
            pole_figure_array = np.ndarray(shape=(num_pts, 3), dtype='float')

            for index in range(num_pts):
                # check fitting result
                intensity_i = peak_info_dict['intensity'][index]

                # rotate Q from instrument coordinate to sample coordinate
                theta_i = 0.5 * peak_info_dict['center'][index]
                omega_i = peak_info_dict['omega'][index]
                if omega_i < 0.:
                    omega_i += 90.

                chi_i = peak_info_dict['chi'][index]
                phi_i = peak_info_dict['phi'][index]
                eta_i = peak_info_dict['eta'][index]
                alpha, beta = self.rotate_project_q(theta_i, omega_i, chi_i, phi_i, eta_i)

                pole_figure_array[index, 0] = alpha
                pole_figure_array[index, 1] = beta
                pole_figure_array[index, 2] = intensity_i
                # END-FOR
            # END-FOR

            # convert
            self._pole_figure_dict[peak_id] = pole_figure_array

    def get_polefigure_array(self, peak_id):
        '''
        return array with polefigure angles and intensity

        Parameters
        ----------
        det_id : peak number
            DESCRIPTION.

        Returns
        -------
        nd.array

        '''

        try:
            return self._pole_figure_dict[peak_id]
        except KeyError:
            self.calculate_pole_figure()
            return self._pole_figure_dict[peak_id]

    def export_pole_figure(self, output_folder: str = '', peak_id_list: list = [], peak_name_list: list = [],
                           run_number: int = 0, file_type: str = 'mtex', file_header: str = '') -> None:
        """
        exported the calculated pole figure
        :param detector_id_list: list of detector IDs to write the pole figure file
        :param file_name:
        :param file_type: ASCII or MTEX (.jul)
        :param file_header: for MTEX format
        :return:
        """

        # process detector ID list
        if peak_id_list is None:
            peak_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs', peak_id_list)

        for i_peak in range(len(peak_id_list)):
            file_name = '{}/HB2B_{}_{}'.format(output_folder, run_number, peak_name_list[i_peak])
            # check inputs
            checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True)
            checkdatatypes.check_string_variable('Output pole figure file type/format', file_type)

            print('[INFO] Exporting polefigure = {} with peak ID {}'.format(peak_name_list[i_peak],
                                                                            peak_id_list[i_peak]))

            # it is a dictionary now
            if file_type.lower() == 'ascii':
                # export pole figure arrays as ascii column file
                export_arrays_to_ascii(self._pole_figure_dict, peak_id_list, file_name)
            elif file_type.lower() == 'mtex':
                # export to MTEX format
                export_to_mtex(self._pole_figure_dict, [peak_id_list[i_peak]], file_name, header=file_header)

        return

    def get_detector_ids(self):
        """
        get all the detector IDs
        :return: list of integer
        """
        return list(self._peak_info_dict.keys())

    def get_pole_figure_1_pt(self, det_id: int, log_index: int) -> Tuple[float, float]:
        """ get 1 pole figure value determined by detector and sample log index
        :param det_id:
        :param log_index:
        :return:
        """
        det_id = to_int('Detector id', det_id, min_value=0)
        log_index = to_int('Sample log index', log_index)

        # get raw parameters' fitted value
        pole_figure_vec = self._pole_figure_dict[det_id]

        pf_tuple = pole_figure_vec[log_index]
        alpha = pf_tuple[0]
        beta = pf_tuple[1]

        return alpha, beta

    def rotate_project_q(self, theta: float, omega: float, chi: float, phi: float, eta: float) -> Tuple[float, float]:
        """
        Projection of angular dependent data onto pole sphere. Analytical solution taken from
        Chapter 8.3 in Bob He Two-Dimensional X-ray Diffraction

        _______________________
        :param two_theta:
        :param omega:
        :param chi:
        :param phi:
        :return: 2-tuple as the projection (alpha, beta)
        """
        theta = to_float('theta', theta)
        omega = to_float('Omega', omega)
        chi = to_float('chi', chi)
        phi = to_float('phi', phi)
        eta = to_float('eta', eta)

        sp = np.sin(np.deg2rad(phi))
        sw = np.sin(np.deg2rad(omega))
        sc = np.sin(np.deg2rad(chi))
        sg = np.sin(np.deg2rad(eta + 270))
        st = np.sin(np.deg2rad(theta))

        cp = np.cos(np.deg2rad(phi))
        cw = np.cos(np.deg2rad(omega))
        cc = np.cos(np.deg2rad(chi))
        cg = np.cos(np.deg2rad(eta + 270))
        ct = np.cos(np.deg2rad(theta))

        h1 = st*(sp*sc*sw + cp*cw) + ct*cg*sp*cc - ct*sg*(sp*sc*cw-cp*sw)
        h2 = -st*(cp*sc*sw-sp*cw) - ct*cg*cp*cc + ct*sg*(cp*sc*cw+sp*sw)
        h_length = np.sqrt(np.square(h1) + np.square(h2))

        alpha = np.rad2deg(np.arccos(h_length))
        beta = np.rad2deg(np.arccos(h1 / h_length))

        if h2 < 0:
            beta = -1 * beta

        return 90 - alpha, beta

    def reset_calculator(self):
        """ reset the pole figure calculator
        :return:
        """
        self._peak_info_dict = dict()
        self._pole_figure_dict = dict()


def export_arrays_to_ascii(pole_figure_array_dict: dict, peak_id_list: list, file_name: str):
    """
    export a dictionary of arrays to an ASCII file
    :param file_name:
    :param detector_id_list: selected the detector IDs for pole figure
    :param pole_figure_array_dict:
    :return:
    """

    # check for correct file extension
    if '.txt' not in file_name:
        file_name = '{}.txt'.format(file_name.replace('.jul', ''))

    # combine
    pole_figure_array_list = []
    for pf_key in peak_id_list:
        pole_figure_vec = pole_figure_array_dict[pf_key]

        if pf_key not in peak_id_list:
            raise NotImplementedError('The data structure of pole figure array is not clear. '
                                      'Find out how detector IDs are involved.')

        pole_figure_array_list.append(pole_figure_vec)
    # END-FOR

    combined_array = np.concatenate(pole_figure_array_list, axis=0)
    # sort
    combined_array = np.sort(combined_array, axis=0)
    # save
    np.savetxt(file_name, combined_array)   # x,y,z equal sized 1D arrays

    return


def export_to_mtex(pole_figure_array_dict: dict, peak_id_list: list, file_name: str, header: str):
    """
    export to mtex format, which includes
    line 1: NRSF2
    line 2: alpha beta intensity
    line 3: (optional header)
    line 4 and on: alpha\tbeta\tintensity
    :param file_name:
    :param detector_id_list: selected the detector IDs for pole figure
    :param pole_figure_array_dict:
    :param header
    :return:
    """

    # check for correct file extension
    if '.jul' not in file_name:
        file_name = '{}.jul'.format(file_name.replace('.jul', ''))

    # initialize output string: MTEX HEAD
    mtex = 'NRSF2\n'
    mtex += 'alpha beta intensity\n'

    # user optional header
    mtex += '{0}\n'.format(header)

    # writing data
    for pf_key in peak_id_list:

        pole_figure_array = pole_figure_array_dict[pf_key]
        for i_pt in range(pole_figure_array.shape[0]):
            mtex += '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\n' \
                    ''.format(pole_figure_array[i_pt, 0], pole_figure_array[i_pt, 1], pole_figure_array[i_pt, 2])
        # END-FOR (i_pt)
    # END-FOR

    # write file
    p_file = open(file_name, 'w')
    p_file.write(mtex)
    p_file.close()
