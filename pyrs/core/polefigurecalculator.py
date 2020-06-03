# This module is to calculate Pole Figure
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int
import numpy as np
from typing import Tuple


def nice(matrix):
    """
    export a string for print a matrix nicely
    :param matrix:
    :return:
    """
    nice_out = ''
    for i_row in range(matrix.shape[0]):
        row = ''
        for j_col in range(matrix.shape[1]):
            row += '{0:5.5f}\t'.format(matrix[i_row, j_col])
        nice_out += row + '\n'

    return nice_out


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
        self._peak_intensity_dict = dict()   # key: detector ID, scan log index (int, int)
        self._peak_fit_info_dict = dict()  # key: detector ID, value: dict; key: log index, value: dict
        self._pole_figure_dict = dict()  # key: detector ID, value: 2-tuple.  scan log indexes (list), 2D array

        # flag
        self._cal_successful = False

    def add_input_data_set(self, det_id: int,
                           peak_intensity_dict: dict, peak_fit_info_dict: dict, log_dict: dict) -> None:
        """ set peak intensity log and experiment logs that are required by pole figure calculation
        :param det_id
        :param peak_intensity_dict : dictionary (key = scan log index (int), value = peak intensity (float)
        :param peak_fit_info_dict: dictionary (key = scan log index (int), value = peak fitting information (float)
        :param log_dict: dictionary (key = scan log index (int), value = dictionary (log name, log value))
        :return:
        """
        # check inputs
        if det_id in self._peak_intensity_dict:
            raise RuntimeError('Detector ID {0} already been added.  Must be reset calculator.'
                               ''.format(det_id))
        det_id = to_int('Detector ID', det_id, min_value=0)
        checkdatatypes.check_dict('Peak intensities', peak_intensity_dict)
        checkdatatypes.check_dict('Peak fitting information', peak_fit_info_dict)
        checkdatatypes.check_dict('Log values for pole figure', log_dict)

        # check sample log index
        if set(peak_intensity_dict.keys()) != set(log_dict.keys()):
            raise RuntimeError('Sample log indexes from peak intensities and sample logs'
                               ' do not match.')

        # add peak intensity
        self._peak_intensity_dict[det_id] = peak_intensity_dict

        # go through all the values
        for scan_log_index in log_dict:
            # check each log index (entry) whether there are enough 2theta
            log_names = log_dict[scan_log_index].keys()
            checkdatatypes.check_list('Pole figure motor names', log_names, ['2theta', 'chi', 'phi', 'omega'])
        # END-FOR

        # set
        self._peak_info_dict[det_id] = log_dict
        self._peak_fit_info_dict[det_id] = peak_fit_info_dict

    def calculate_pole_figure(self, det_id_list):
        """ Calculate pole figures
        :param det_id_list:
        :return:
        """
        # check input
        if det_id_list is None:
            det_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs to calculate pole figure', det_id_list,
                                      self.get_detector_ids())
        # END-IF

        for det_id in det_id_list:
            # calculator by each detector
            peak_intensity_dict = self._peak_intensity_dict[det_id]
            peak_info_dict = self._peak_info_dict[det_id]

            # construct the output
            scan_log_index_list = sorted(peak_intensity_dict.keys())
            num_pts = len(scan_log_index_list)
            pole_figure_array = np.ndarray(shape=(num_pts, 3), dtype='float')

            for index, scan_index in enumerate(scan_log_index_list):
                # check fitting result
                intensity_i = peak_intensity_dict[scan_index]

                # rotate Q from instrument coordinate to sample coordinate
                theta_i = 0.5 * peak_info_dict[scan_index]['center']
                omega_i = peak_info_dict[scan_index]['omega']
                if omega_i < 0.:
                    omega_i += 90.
                chi_i = peak_info_dict[scan_index]['chi']
                phi_i = peak_info_dict[scan_index]['phi']
                eta_i = peak_info_dict[scan_index]['eta']
                alpha, beta = self.rotate_project_q(theta_i, omega_i, chi_i, phi_i, eta_i)

                pole_figure_array[index, 0] = alpha
                pole_figure_array[index, 1] = beta
                pole_figure_array[index, 2] = intensity_i
                # END-FOR
            # END-FOR

            # convert
            self._pole_figure_dict[det_id] = scan_log_index_list, pole_figure_array

    def export_pole_figure(self, detector_id_list,  file_name, file_type, file_header=''):
        """
        exported the calculated pole figure
        :param detector_id_list: list of detector IDs to write the pole figure file
        :param file_name:
        :param file_type: ASCII or MTEX (.jul)
        :param file_header: for MTEX format
        :return:
        """
        # TESTME - 20180711 - Clean this method and allow user to specifiy header

        # process detector ID list
        if detector_id_list is None:
            detector_id_list = self.get_detector_ids()
        else:
            checkdatatypes.check_list('Detector IDs', detector_id_list)

        # check inputs
        checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True)
        checkdatatypes.check_string_variable('Output pole figure file type/format', file_type)

        # it is a dictionary now
        if file_type.lower() == 'ascii':
            # export pole figure arrays as ascii column file
            export_arrays_to_ascii(self._pole_figure_dict, detector_id_list, file_name)
        elif file_type.lower() == 'mtex':
            # export to MTEX format
            export_to_mtex(self._pole_figure_dict, detector_id_list, file_name, header=file_header)

        return

    def get_detector_ids(self):
        """
        get all the detector IDs
        :return: list of integer
        """
        return self._peak_intensity_dict.keys()

    def get_peak_fit_parameter_vec(self, param_name: str, det_id: int) -> np.ndarray:
        """ get the fitted parameters and return in vector
        :param param_name:
        :param det_id:
        :return:
        """
        det_id = to_int('Detector ID', det_id, min_value=0)

        param_vec = np.ndarray(shape=(len(self._peak_fit_info_dict[det_id]), ), dtype='float')
        log_index_list = sorted(self._peak_fit_info_dict[det_id].keys())
        for i, log_index in enumerate(log_index_list):
            try:
                param_vec[i] = self._peak_fit_info_dict[det_id][log_index][param_name]
            except KeyError:
                raise RuntimeError('Parameter {0} is not a key.  Candidates are {1} ... {2}'
                                   ''.format(param_name, self._peak_fit_info_dict[det_id].keys(),
                                             self._peak_fit_info_dict[det_id][log_index].keys()))
        # END-FOR

        return param_vec

    def get_pole_figure_1_pt(self, det_id: int, log_index: int) -> Tuple[float, float]:
        """ get 1 pole figure value determined by detector and sample log index
        :param det_id:
        :param log_index:
        :return:
        """
        det_id = to_int('Detector id', det_id, min_value=0)
        log_index = to_int('Sample log index', log_index)

        # get raw parameters' fitted value
        log_index_vec, pole_figure_vec = self._pole_figure_dict[det_id]

        # check
        if log_index != int(log_index_vec[log_index]):
            raise RuntimeError('Log index {0} does not match the value in log index vector')

        pf_tuple = pole_figure_vec[log_index]
        alpha = pf_tuple[0]
        beta = pf_tuple[1]

        return alpha, beta

    def get_pole_figure_vectors(self, det_id, max_cost, min_cost=1.):
        """ return Pole figure in a numpy 2D array
        :param det_id:
        :param max_cost:
        :param min_cost:
        :return: 2-tuple: (1) an integer list (2) numpy array with shape (n, 3).  n is the number of data points
        """
        # check input
        if max_cost is None:
            max_cost = 1.E20
        else:
            max_cost = to_float('Maximum peak fitting cost value', max_cost, min_value=0.)

        # get raw parameters' fitted value
        log_index_vec, pole_figure_vec = self._pole_figure_dict[det_id]

        # get costs and filter out isnan()
        cost_vec = self.get_peak_fit_parameter_vec('cost', det_id)

        taken_index_list = list()
        for idx in range(len(cost_vec)):
            if min_cost < cost_vec[idx] < max_cost:
                taken_index_list.append(idx)
        # END-IF

        selected_log_index_vec = np.take(log_index_vec, taken_index_list, axis=0)
        selected_pole_figure_vec = np.take(pole_figure_vec, taken_index_list, axis=0)

        return selected_log_index_vec, selected_pole_figure_vec

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
        sg = np.sin(np.deg2rad(eta))
        st = np.sin(np.deg2rad(theta))

        cp = np.cos(np.deg2rad(phi))
        cw = np.cos(np.deg2rad(omega))
        cc = np.cos(np.deg2rad(chi))
        cg = np.cos(np.deg2rad(eta))
        ct = np.cos(np.deg2rad(theta))

        h1 = st*(sp*sc*sw+cp*cw) + ct*cg*sp*cc - ct*sg*(sp*sc*cw-cp*sw)
        h2 = -st*(cp*sc*sw-sp*cw) - ct*cg*cp*cc + ct*sg*(cp*sc*cw+sp*sw)
        # h3 = st*cc*sw - ct*sg*cc*cw - ct*cg*sc
        h_length = np.sqrt(np.square(h1) + np.square(h2))

        alpha = np.arccos(h_length)
        beta = np.rad2deg(np.arccos(h1 / h_length))
        if h2 < 0:
            beta *= -1.

        return alpha, beta

    def reset_calculator(self):
        """ reset the pole figure calculator
        :return:
        """
        self._peak_info_dict = dict()
        self._peak_intensity_dict = dict()
        self._pole_figure_dict = dict()

# END-OF-CLASS (PoleFigureCalculator)


def export_arrays_to_ascii(pole_figure_array_dict, detector_id_list, file_name):
    """
    export a dictionary of arrays to an ASCII file
    :param file_name:
    :param detector_id_list: selected the detector IDs for pole figure
    :param pole_figure_array_dict:
    :return:
    """
    # check input types
    checkdatatypes.check_dict('Pole figure array dictionary', pole_figure_array_dict)
    checkdatatypes.check_list('Detector ID list', detector_id_list)

    print('[INFO] Export Pole Figure Arrays To ASCII:\nKeys: {0}\nValues[0]: {1}'
          ''.format(pole_figure_array_dict.keys(), pole_figure_array_dict.values()[0]))

    # combine
    pole_figure_array_list = list()
    for pf_key in pole_figure_array_dict.keys():
        index_vec, pole_figure_vec = pole_figure_array_dict[pf_key]

        if pf_key not in detector_id_list:
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


def export_to_mtex(pole_figure_array_dict, detector_id_list, file_name, header):
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
    # check input types
    checkdatatypes.check_dict('Pole figure array dictionary', pole_figure_array_dict)
    checkdatatypes.check_list('Detector ID list', detector_id_list)

    # initialize output string: MTEX HEAD
    mtex = 'NRSF2\n'
    mtex += 'alpha beta intensity\n'

    # user optional header
    mtex += '{0}\n'.format(header)

    # writing data
    pf_keys = sorted(pole_figure_array_dict.keys())
    for pf_key in pf_keys:
        print('[STUDY ]Pole figure key = {}.  It is in detector ID list {} = {}'
              ''.format(pf_key, detector_id_list, pf_key in detector_id_list))
        if pf_key not in detector_id_list:
            raise NotImplementedError('The data structure of pole figure array is not clear. '
                                      'Find out how detector IDs are involved.')

        sample_log_index, pole_figure_array = pole_figure_array_dict[pf_key]
        for i_pt in range(pole_figure_array.shape[0]):
            mtex += '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\n' \
                    ''.format(pole_figure_array[i_pt, 0], pole_figure_array[i_pt, 1], pole_figure_array[i_pt, 2])
        # END-FOR (i_pt)
    # END-FOR

    # write file
    p_file = open(file_name, 'w')
    p_file.write(mtex)
    p_file.close()
