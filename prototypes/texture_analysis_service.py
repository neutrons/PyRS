def calculate_pole_figure():
    # NOTE: This is the original implementation of pole figure calculation API

    # Check
    if project_name not in self._pole_figure_calculator_dict:
        raise KeyError('{} does not exist. Available: {}'
                       ''.format(project_name, self._pole_figure_calculator_dict.keys()))

    # check input
    checkdatatypes.check_string_variable('Data key/ID', data_key)
    if detector_id_list is None:
        detector_id_list = self.get_detector_ids(data_key)
    else:
        checkdatatypes.check_list('Detector IDs', detector_id_list)

    # get peak intensities from fitting
    # peak_intensities = self.get_peak_intensities(data_key, detector_id_list)

    # initialize pole figure
    self._last_pole_figure_calculator = polefigurecalculator.PoleFigureCalculator()
    self._pole_figure_calculator_dict[data_key] = self._last_pole_figure_calculator

    # set up pole figure logs and get it
    log_names = [('2theta', '2theta'),
                 ('omega', 'omega'),
                 ('chi', 'chi'),
                 ('phi', 'phi')]

    for det_id in detector_id_list:
        # get intensity and log value
        log_values = self.data_center.get_scan_index_logs_values((data_key, det_id), log_names)

        try:
            optimizer = self._get_optimizer((data_key, det_id))
            peak_intensities = optimizer.get_peak_intensities()
        except AttributeError as att:
            raise RuntimeError('Unable to get peak intensities. Check whether peaks have been fit.  FYI: {}'
                               ''.format(att))
        fit_info_dict = optimizer.get_peak_fit_parameters()

        # add value to pole figure calculate
        self._last_pole_figure_calculator.add_input_data_set(det_id, peak_intensities, fit_info_dict, log_values)
    # END-FOR

    # do calculation
    self._last_pole_figure_calculator.calculate_pole_figure(detector_id_list)

    return


def get_pole_figure_value(self, data_key, detector_id, log_index):
    """
    get pole figure value of a certain measurement identified by data key and log index
    :param data_key:
    :param detector_id
    :param log_index:
    :return:
    """
    print('[ERROR] Pole figure from {} at detector ID {}/{} is not implemented'
          ''.format(data_key, detector_id, log_index))
    return None, None
    checkdatatypes.check_int_variable('Scan log #', log_index, (0, None))

    alpha, beta = self._last_pole_figure_calculator.get_pole_figure_1_pt(detector_id, log_index)

    # log_index_list, pole_figures = self._last_pole_figure_calculator.get_pole_figure_vectors
    # (detector_id, max_cost=None)
    # if len(pole_figures) < log_index + 1:
    #     alpha = 0
    #     beta = 0
    # else:
    #     try:
    #         alpha = pole_figures[log_index][0]
    #         beta = pole_figures[log_index][1]
    #     except ValueError as val_err:
    #         raise RuntimeError('Given detector {0} scan log index {1} of data IDed as {2} is out of range as '
    #                            '({3}, {4})  (error = {5})'
    #                            ''.format(detector_id, log_index, data_key, 0, len(pole_figures), val_err))
    # # END-IF-ELSE

    return alpha, beta


def get_pole_figure_values(self, data_key, detector_id_list, max_cost):
    """ API method to get the (N, 3) array for pole figures
    :param data_key:
    :param detector_id_list:
    :param max_cost:
    :return:
    """
    pole_figure_calculator = self._pole_figure_calculator_dict[data_key]
    assert isinstance(pole_figure_calculator, polefigurecalculator.PoleFigureCalculator), \
        'Pole figure calculator type mismatched. Input is of type {0} but expected as {1}.' \
        ''.format(type(pole_figure_calculator), 'polefigurecalculator.PoleFigureCalculato')

    if detector_id_list is None:
        detector_id_list = pole_figure_calculator.get_detector_ids()
    else:
        checkdatatypes.check_list('Detector ID list', detector_id_list)

    # get all the pole figure vectors
    vec_alpha = None
    vec_beta = None
    vec_intensity = None
    for det_id in detector_id_list:
        print('[DB...BAt] Get pole figure from detector {0}'.format(det_id))
        # get_pole_figure returned 2 tuple.  we need the second one as an array for alpha, beta, intensity
        sub_array = pole_figure_calculator.get_pole_figure_vectors(det_id, max_cost)[1]
        vec_alpha_i = sub_array[:, 0]
        vec_beta_i = sub_array[:, 1]
        vec_intensity_i = sub_array[:, 2]

        print('Det {} # data points = {}'.format(det_id, len(sub_array)))
        # print ('alpha: {0}'.format(vec_alpha_i))

        if vec_alpha is None:
            vec_alpha = vec_alpha_i
            vec_beta = vec_beta_i
            vec_intensity = vec_intensity_i
        else:
            vec_alpha = numpy.concatenate((vec_alpha, vec_alpha_i), axis=0)
            vec_beta = numpy.concatenate((vec_beta, vec_beta_i), axis=0)
            vec_intensity = numpy.concatenate((vec_intensity, vec_intensity_i), axis=0)
        # END-IF-ELSE
        print('Updated alpha: size = {0}: {1}'.format(len(vec_alpha), vec_alpha))
    # END-FOR

    return vec_alpha, vec_beta, vec_intensity


def save_pole_figure(self, data_key, detectors, file_name, file_type):
    """
    save pole figure/export pole figure
    :param data_key:
    :param detectors: a list of detector (ID)s or None (default for all detectors)
    :param file_name:
    :param file_type:
    :return:
    """

    checkdatatypes.check_string_variable('Data key', data_key)

    if data_key in self._pole_figure_calculator_dict:
        self._pole_figure_calculator_dict[data_key].export_pole_figure(detectors, file_name, file_type)
    else:
        raise RuntimeError('Data key {0} is not calculated for pole figure.  Current data keys contain {1}'
                           ''.format(data_key, self._pole_figure_calculator_dict.keys()))

    return
