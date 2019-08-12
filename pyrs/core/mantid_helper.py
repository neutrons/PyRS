from mantid.api import AnalysisDataService as mtd
from mantid.simpleapi import LoadSpiceXML2DDet, Transpose
from pyrs.utilities import checkdatatypes
from pyrs.core import workspaces

import mantid
from mantid.simpleapi import FitPeaks, CreateWorkspace
from mantid.api import AnalysisDataService


def generate_mantid_workspace(hidra_workspace, workspace_name, mask_id=None):
    """
    Generate a Mantid workspace from a HidraWorkspace
    :param hidra_workspace:
    :return:
    """
    checkdatatypes.check_type('Hidra workspace', hidra_workspace, workspaces.HidraWorkspace)

    two_theta_array, data_y_matrix = hidra_workspace.get_reduced_diffraction_data(mask_id)

    # workspace name:
    if workspace_name is None:
        workspace_name = hidra_workspace.name()
    else:
        checkdatatypes.check_string_variable('Workspace name', workspace_name)

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


def workspace_exists(ws_name):
    """
    check whether a workspace exists or not
    :param ws_name:
    :return:
    """
    checkdatatypes.check_string_variable('Workspace name', ws_name)

    return mtd.doesExist(ws_name)
