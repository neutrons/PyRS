from mantid.api import AnalysisDataService as mtd
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose
from pyrs.utilities import checkdatatypes
from pyrs.core import workspaces

import mantid
from mantid.simpleapi import FitPeaks, CreateWorkspace
from mantid.api import AnalysisDataService


def generate_mantid_workspace(hidra_workspace, workspace_name, mask_id=None):
    """
    Generate a Mantid MatrixWorkspace from a HidraWorkspace
    :param hidra_workspace:
    :param workspace_name: string for output workspace name
    :param mask_id: Mask index for the reduced diffraction data in HidraWorkspace/HidraProjectFile
    :return:
    """
    # Check inputs
    checkdatatypes.check_type('Hidra workspace', hidra_workspace, workspaces.HidraWorkspace)
    # workspace name:
    if workspace_name is None:
        workspace_name = hidra_workspace.name
    else:
        checkdatatypes.check_string_variable('Workspace name', workspace_name)

    two_theta_array, data_y_matrix = hidra_workspace.get_reduced_diffraction_data_set(mask_id)

    matrix_ws = CreateWorkspace(DataX=two_theta_array,
                                DataY=data_y_matrix,
                                NSpec=data_y_matrix.shape[0],
                                OutputWorkspace=workspace_name)

    return matrix_ws


def get_data_y(ws_name, transpose):
    """
    read data Y of a workspace
    :param ws_name:
    :param transpose: if the workspace is N x 1, then dimension == 1 means that workspace
                      will be transposed for exporting data
    :return:
    """
    workspace = retrieve_workspace(ws_name, True)

    if transpose == 1:
        ws_name_temp = '{}_transposed'.format(ws_name)
        if not workspace_exists(ws_name):
            Transpose(InputWorkspace=ws_name, OutputWorkspace=ws_name_temp)
        transpose_ws = retrieve_workspace(ws_name_temp)
        data_y = workspace.readY(0)
    else:
        raise NotImplementedError('It has not been implemented to read 1 X N array')

    return data_y


def retrieve_workspace(ws_name, throw=True):
    """
    retrieve the reference to a workspace
    :param ws_name:
    :param throw:
    :return:
    """
    if not workspace_exists(ws_name):
        if throw:
            raise RuntimeError('Workspace {} does not exist in Mantid ADS'
                               ''.format(ws_name))
        else:
            return None

    return mtd.retrieve(ws_name)


def study_mantid_peak_fitting():
    """
    Save the workspaces used or output from Mantid FitPeaks
    :return:
    """
    # debug mode is disabled
    # find the directory for file
    dir_name = rs_scan_io.get_temp_directory()
    print ('[DEBUG-INFO] Mantid fit debugging data files will be written to {0}'.format(dir_name))

    # workspace for data
    base_name = self._reference_id.replace('.', '_') + '_' + peak_function_name
    raw_file_name = os.path.join(dir_name, '{0}_data.nxs'.format(base_name))
    rs_scan_io.save_mantid_nexus(self._workspace_name, raw_file_name,
                                 title='raw data for {0}'.format(self._reference_id))

    # peak window workspace
    fit_window_name = os.path.join(dir_name, '{0}_fit_window.nxs'.format(base_name))
    rs_scan_io.save_mantid_nexus(peak_window_ws_name, fit_window_name, title='Peak fit window workspace')

    # peak center workspace
    peak_center_file_name = os.path.join(dir_name, '{0}_peak_center.nxs'.format(base_name))
    rs_scan_io.save_mantid_nexus(self._center_of_mass_ws_name, peak_center_file_name,
                                 title='Peak center (center of mass) workspace')


def workspace_exists(ws_name):
    """
    check whether a workspace exists or not
    :param ws_name:
    :return:
    """
    checkdatatypes.check_string_variable('Workspace name', ws_name)

    return mtd.doesExist(ws_name)
