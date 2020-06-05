from mantid.simpleapi import CreateWorkspace
from mantid.simpleapi import Transpose
from mantid.simpleapi import AddSampleLog, SaveNexusProcessed
from mantid.api import MatrixWorkspace
from pyrs.core import workspaces
from pyrs.utilities import checkdatatypes
import numpy as np


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

    # Get data from HiDRA Workspace
    two_theta_matrix, data_y_matrix, data_e_matrix = hidra_workspace.get_reduced_diffraction_data_set(mask_id)

    # Mantid (2019.11) does not accept NaN
    # Convert all NaN to zero.  No good peak will have NaN or Zero
    data_y_matrix[np.where(np.isnan(data_y_matrix))] = 0.
    data_e_matrix[np.where(np.isnan(data_e_matrix))] = 0.

    # Create Mantid workspace
    matrix_ws = CreateWorkspace(DataX=two_theta_matrix,
                                DataY=data_y_matrix,
                                DataE=data_e_matrix,
                                NSpec=data_y_matrix.shape[0],
                                OutputWorkspace=workspace_name, EnableLogging=False)

    return matrix_ws
