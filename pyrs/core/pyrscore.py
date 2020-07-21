# This is the core of PyRS serving as the controller of PyRS and hub for all the data
from pyrs.utilities import checkdatatypes
from pyrs.core import instrument_geometry
from pyrs.utilities import file_util
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.core import reduction_manager
from pyrs.core import polefigurecalculator
import os
import numpy


class PyRsCore:
    """
    PyRS core
    """
    def __init__(self):
        """
        initialization
        """
        # Declare services
        # reduction service
        self._reduction_service = reduction_manager.HB2BReductionManager()

        # pole figure calculation
        self._pole_figure_calculator_dict = dict()
        self._last_pole_figure_calculator = None

        # strain and stress calculator
        self._ss_calculator_dict = dict()   # [session name][strain/stress type: 1/2/3] = ss calculator
        self._curr_ss_session = None
        self._curr_ss_type = None

        # Working environment
        self._working_dir = None

        return

    @property
    def strain_stress_calculator(self):
        """
        return the handler to strain/stress calculator
        :return:
        """
        if self._curr_ss_session is None:
            return None

        return self._ss_calculator_dict[self._curr_ss_session][self._curr_ss_type]

    @property
    def working_dir(self):
        """
        get working directory
        :return:
        """
        w_dir = self._working_dir
        if w_dir is None:
            w_dir = os.getcwd()

        return w_dir

    @working_dir.setter
    def working_dir(self, user_dir):
        """
        set working directory
        :param user_dir:
        :return:
        """
        checkdatatypes.check_file_name(user_dir, check_writable=False, is_dir=True)

        self._working_dir = user_dir

        return

    def calculate_pole_figure(self, project_name, solid_angles):
        """Calculate pole figure

        Previous API: data_key, detector_id_list

        Returns
        -------

        """
        raise NotImplementedError('Project {} of solid angles {} need to be implemented soon!'
                                  ''.format(project_name, solid_angles))

    @staticmethod
    def _get_strain_stress_type_key(is_plane_strain, is_plane_stress):
        """
        blabla
        :param is_plane_strain:
        :param is_plane_stress:
        :return: 1: regular unconstrained, 2: plane strain, 3: plane stress
        """
        if is_plane_strain:
            return 2
        elif is_plane_stress:
            return 3

        return 1

    #  get_detector_ids(self, data_key) removed due to new workflow to calculate pole figure and API

    def get_diffraction_data(self, session_name, sub_run, mask):
        """ get diffraction data of a certain session/wokspace
        :param session_name: name of session for workspace
        :param sub_run: sub run of the diffraction ata
        :param mask: String as mask ID for reduced diffraction data
        :return: tuple: vec_2theta, vec_intensit
        """
        diff_data_set = self._reduction_service.get_reduced_diffraction_data(session_name, sub_run, mask)

        return diff_data_set

    def get_pole_figure_value(self, data_key, detector_id, log_index):
        """
        get pole figure value of a certain measurement identified by data key and log index
        :param data_key:
        :param detector_id
        :param log_index:
        :return:
        """
        assert self
        print('[ERROR] Pole figure from {} at detector ID {}/{} is not implemented'
              ''.format(data_key, detector_id, log_index))
        return None, None

    def get_pole_figure_values(self, data_key, detector_id_list, max_cost):
        """ API method to get the (N, 3) array for pole figures
        :param data_key:
        :param detector_id_list:
        :param max_cost:
        :return:
        """
        pole_figure_calculator = self._pole_figure_calculator_dict[data_key]
        assert isinstance(pole_figure_calculator, polefigurecalculator.PoleFigureCalculator),\
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
            print('Updated alpha: size = {0}: {1}'.format(len(vec_alpha), vec_alpha))

        return vec_alpha, vec_beta, vec_intensity

    def load_hidra_project(self, hidra_h5_name, project_name, load_detector_counts=True, load_diffraction=False):
        """

        Parameters
        ----------
        hidra_h5_name
            name of HIDRA project file in HDF5 format
        project_name
            name of the reduction project specified by user to trace
        load_detector_counts
        load_diffraction

        Returns
        -------
        pyrs.core.workspaces.HidraWorkspace

        """
        # Initialize session
        self._reduction_service.init_session(project_name)

        # Load project
        ws = self._reduction_service.load_hidra_project(project_file_name=hidra_h5_name,
                                                        load_calibrated_instrument=False,
                                                        load_detectors_counts=load_detector_counts,
                                                        load_reduced_diffraction=load_diffraction)

        return ws

    def save_diffraction_data(self, project_name, file_name):
        """ Save (reduced) diffraction data to HiDRA project file
        :param project_name: HiDRA wokspace reference or name
        :param file_name:
        :return:
        """
        self.reduction_service.save_reduced_diffraction(project_name, file_name)

        return

    def save_peak_fit_result(self, project_name, hidra_file_name, peak_tag, overwrite=True):
        """ Save the result from peak fitting to HiDRA project file
        Parameters
        ----------
        project_name: str
            name of peak fitting session
        hidra_file_name: String
            project file to export peaks fitting result to
        peak_tag : str
            peak tag
        overwrite: bool
            Flag to append to an existing file or overwrite it

        Returns
        -------

        """
        if project_name is None:
            optimizer = self._peak_fit_engine
        else:
            optimizer = self._peak_fitting_dict[project_name]

        # Determine the file IO mode
        if os.path.exists(hidra_file_name) and overwrite is False:
            # file exists and user does not want overwrite: READWRITE mode
            file_mode = HidraProjectFileMode.READWRITE
        else:
            # starting as a new file
            file_mode = HidraProjectFileMode.OVERWRITE

        # Create HiDRA project file
        hidra_project_file = HidraProjectFile(hidra_file_name, file_mode)
        # Export peaks
        optimizer.export_to_hydra_project(hidra_project_file, peak_tag)
        # Close
        hidra_project_file.close()

        return

    @staticmethod
    def save_pole_figure(data_key, detectors, file_name, file_type):
        """
        save pole figure/export pole figure
        :param data_key:
        :param detectors: a list of detector (ID)s or None (default for all detectors)
        :param file_name:
        :param file_type:
        :return:
        """
        raise NotImplementedError('{}/{}/{}/{} need to be implemented'
                                  ''.format(data_key, detectors, file_name, file_type))

    def reduce_diffraction_data(self, session_name, num_bins, pyrs_engine, mask_file_name=None,
                                geometry_calibration=None, sub_run_list=None):
        """ Reduce all sub runs in a workspace from detector counts to diffraction data
        :param session_name:
        :param num_bins:
        :param pyrs_engine:
        :param mask_file_name:
        :param geometry_calibration: True/file name/AnglerCameraDetectorShift/None), False, None/(False, )
        :param sub_run_list: list of sub run numbers or None (for all)
        :return:
        """
        # Mask file
        if mask_file_name:
            mask_info = self._reduction_service.load_mask_file(mask_file_name)
            mask_id = mask_info[2]
            print('L650 Mask ID = {}'.format(mask_id))
        else:
            mask_id = None

        # Geometry calibration
        if geometry_calibration is None or geometry_calibration is False:
            # No apply
            apply_calibration = False
        elif isinstance(geometry_calibration, str):
            # From a Json file
            calib_shift = instrument_geometry.AnglerCameraDetectorShift(0, 0, 0, 0, 0, 0)
            calib_shift.from_json(geometry_calibration)
            apply_calibration = calib_shift
        elif isinstance(geometry_calibration, instrument_geometry.AnglerCameraDetectorShift):
            # Already a AnglerCameraDetectorShift instance
            apply_calibration = geometry_calibration
        elif geometry_calibration is True:
            # Use what is loaded from file or set to workspace before
            apply_calibration = True
        else:
            raise RuntimeError('Argument geometry_calibration of value {} and type {} is not supported'
                               ''.format(geometry_calibration, type(geometry_calibration)))

        # Reduce
        # TODO - Mask/MaskID shall be refactored
        self._reduction_service.reduce_diffraction_data(session_name, apply_calibration,
                                                        num_bins, pyrs_engine, sub_run_list,
                                                        mask_id, mask_id)

    @property
    def reduction_service(self):
        """
        get the reference to reduction engine
        :return:
        """
        return self._reduction_service

    def reset_strain_stress(self, is_plane_strain, is_plane_stress):
        """ reset the strain and stress calculation due to change of type

        :param is_plane_strain:
        :param is_plane_stress:
        :return:
        """
        if self._curr_ss_session is None:
            raise RuntimeError('Current session is not named.')
        elif self._curr_ss_session not in self._ss_calculator_dict:
            print('[WARNING] Current strain/stress session does not exist.')
            return

        ss_type_index = self._get_strain_stress_type_key(is_plane_strain, is_plane_stress)
        if ss_type_index == self._curr_ss_type:
            raise RuntimeError('Same strain/stress type (plane strain = {}, plane stress = {}'
                               ''.format(is_plane_strain, is_plane_stress))

        # rename the current strain stress name
        # saved_ss_name = self._curr_ss_session + '_{}_{}'.format(is_plane_strain, is_plane_stress)
        # self._ss_calculator_dict[self._curr_ss_session].rename(saved_ss_name)
        # prev_calculator = self._ss_calculator_dict[self._curr_ss_session]
        # self._ss_calculator_dict[saved_ss_name] = prev_calculator

        # reset new strain/stress calculator
        new_ss_calculator = self.strain_stress_calculator.migrate(is_plane_strain, is_plane_stress)

        self._ss_calculator_dict[self._curr_ss_session][ss_type_index] = new_ss_calculator
        self._curr_ss_type = ss_type_index

        return self._curr_ss_session

    def save_nexus(self, data_key, file_name):
        """
        save data in a MatrixWorkspace to Mantid processed NeXus file
        :param data_key:
        :param file_name:
        :return:
        """
        # Check - peak fitting has been moved from `this`
        optimizer = self._get_peak_fitting_controller(data_key)

        # get the workspace name
        try:
            matrix_name = optimizer.get_mantid_workspace_name()
            # save
            file_util.save_mantid_nexus(matrix_name, file_name)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to write to NeXus because Mantid fit engine is not used.\nError info: {0}'
                               ''.format(run_err))

        try:
            matrix_name = optimizer.get_center_of_mass_workspace_name()
            # save
            dir_name = os.path.dirname(file_name)
            base_name = os.path.basename(file_name)
            file_name = os.path.join(dir_name, base_name.split('.')[0] + '_com.nxs')
            file_util.save_mantid_nexus(matrix_name, file_name)
        except RuntimeError as run_err:
            raise RuntimeError('Unable to write COM to NeXus because Mantid fit engine is not used.\nError info: {0}'
                               ''.format(run_err))

    def slice_data(self, even_nexus, splicer_id):
        """Event split a NeXus file

        Reference: PyVDRive
                    status, message = self.get_controller().slice_data(raw_file_name, self._currSlicerKey,
                                                           reduce_data=False,
                                                           vanadium=None,
                                                           save_chopped_nexus=False,
                                                           output_dir=os.getcwd(),
                                                           export_log_type='loadframe')

        Parameters
        ----------
        even_nexus
        splicer_id: str
            ID to locate an event slicer (splitter) which has been set up
        Returns
        -------

        """
        # TODO - Implement method
        raise NotImplementedError('Implement this ASAP')
