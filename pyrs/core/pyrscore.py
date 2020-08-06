# This is the core of PyRS serving as the controller of PyRS and hub for all the data
from pyrs.core import reduction_manager
import os


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
    def working_dir(self):
        """
        get working directory
        :return:
        """
        w_dir = self._working_dir
        if w_dir is None:
            w_dir = os.getcwd()

        return w_dir

    def get_diffraction_data(self, session_name, sub_run, mask):
        """ get diffraction data of a certain session/wokspace
        :param session_name: name of session for workspace
        :param sub_run: sub run of the diffraction ata
        :param mask: String as mask ID for reduced diffraction data
        :return: tuple: vec_2theta, vec_intensit
        """
        diff_data_set = self._reduction_service.get_reduced_diffraction_data(session_name, sub_run, mask)

        return diff_data_set

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

    @property
    def reduction_service(self):
        """
        get the reference to reduction engine
        :return:
        """
        return self._reduction_service
