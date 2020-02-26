from mantid.simpleapi import CreateWorkspace
from mantid.simpleapi import mtd
from mantid.simpleapi import Transpose
from mantid.simpleapi import AddSampleLog, SaveNexusProcessed
from mantid.api import MatrixWorkspace
import os
from pyrs.core import workspaces
from pyrs.utilities import checkdatatypes
import numpy as np


def export_workspaces(ws_name_list):
    for ws_name in ws_name_list:
        SaveNexusProcessed(InputWorkspace=ws_name, Filename='/tmp/{}.nxs'.format(ws_name))


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
    two_theta_matrix, data_y_matrix = hidra_workspace.get_reduced_diffraction_data_set(mask_id)

    # Mantid (2019.11) does not accept NaN
    # Convert all NaN to zero.  No good peak will have NaN or Zero
    data_y_matrix[np.where(np.isnan(data_y_matrix))] = 0.
    data_e_matrix = np.sqrt(data_y_matrix)
    data_e_matrix[np.where(np.isnan(data_e_matrix))] = 0.

    # Create Mantid workspace
    matrix_ws = CreateWorkspace(DataX=two_theta_matrix,
                                DataY=data_y_matrix,
                                DataE=data_e_matrix,  # TODO this is wrong, but better than zeros
                                NSpec=data_y_matrix.shape[0],
                                OutputWorkspace=workspace_name, EnableLogging=False)

    return matrix_ws


def get_data_y(ws_name, transpose):
    """
    read data Y of a workspace
    :param ws_name:
    :param transpose: if the workspace is N x 1, then dimension == 1 means that workspace
                      will be transposed for exporting data
    :return:
    """
    if transpose == 1:
        ws_name_temp = '{}_transposed'.format(ws_name)
        if not workspace_exists(ws_name):
            Transpose(InputWorkspace=ws_name, OutputWorkspace=ws_name_temp)
        transpose_ws = retrieve_workspace(ws_name_temp)
        data_y = transpose_ws.readY(0)
    else:
        raise NotImplementedError('It has not been implemented to read 1 X N array')

    return data_y


def get_log_value(workspace, log_name):
    """
    get log value from workspace
    :param workspace:
    :param log_name:
    :return:
    """
    try:
        sample_log_property = workspace.run().getProperty(log_name)
    except KeyError:
        raise RuntimeError('Workspace {} does not have property {}'.format(workspace, log_name))

    log_value = sample_log_property.value()

    return log_value


def is_matrix_workspace(workspace):
    """Check an object is a MantidWorkspace

    Parameters
    ----------
    workspace

    Returns
    -------

    """
    return isinstance(workspace, MatrixWorkspace)


def set_log_value(workspace, log_name, log_value, unit='meter'):
    """
    set a value to a workspace's sample logs
    :param workspace:
    :param log_name:
    :param log_value:
    :param unit:
    :return:
    """
    AddSampleLog(Workspace=workspace, LogName=log_name,
                 LogText='{}'.format(log_value),
                 LogType='Number Series', LogUnit=unit,
                 NumberType='Double')

    return


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


def study_mantid_peak_fitting(workspace_name, param_table_ws_name, cal_peak_ws_name, output_ws_name,
                              peak_function_name, info):
    """
    Save the workspaces used or output from Mantid FitPeaks
    :return:
    """
    from pyrs.utilities import file_util

    # debug mode is disabled
    # find the directory for file
    dir_name = file_util.get_temp_directory()
    print('[DEBUG-INFO] Mantid fit debugging data files will be written to {0}'.format(dir_name))

    # workspace for data
    base_name = workspace_name + '_' + peak_function_name
    raw_file_name = os.path.join(dir_name, '{0}_data.nxs'.format(base_name))
    file_util.save_mantid_nexus(workspace_name, raw_file_name,
                                title='raw data for {0}'.format(info))

    # peak window workspace
    fit_window_file_name = os.path.join(dir_name, '{0}_param_table.nxs'.format(base_name))
    file_util.save_mantid_nexus(param_table_ws_name, fit_window_file_name, title='Peak fit window workspace')

    # peak center workspace
    if cal_peak_ws_name is not None:
        peak_center_file_name = os.path.join(dir_name, '{0}_model.nxs'.format(base_name))
        file_util.save_mantid_nexus(cal_peak_ws_name, peak_center_file_name,
                                    title='Peak center (center of mass) workspace')

    # Output workspace
    if output_ws_name is not None:
        output_file_name = os.path.join(dir_name, '{0}_peak_fit_output.nxs'.format(base_name))
        file_util.save_mantid_nexus(output_ws_name, output_file_name,
                                    title='Peak fit output (center of mass) workspace')

    return


def workspace_exists(ws_name):
    """
    check whether a workspace exists or not
    :param ws_name:
    :return:
    """
    checkdatatypes.check_string_variable('Workspace name', ws_name)

    return mtd.doesExist(ws_name)
