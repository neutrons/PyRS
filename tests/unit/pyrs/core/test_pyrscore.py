#!/usr/bin/python
import os
from pyrs.core import pyrscore
import pytest


def broken_test_pole_figure_calculation():
    """
    main testing body to test the workflow to calculate pole figure
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()
    assert rs_core

    # import data file: detector ID and file name
    test_data_set = [(1, 'tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5'),
                     (2, 'tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5'),
                     (3, 'tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5'),
                     (4, 'tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5'),
                     (5, 'tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5'),
                     (6, 'tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5'),
                     (7, 'tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5')]
    assert test_data_set


def test_main():
    """
    test main
    :return:
    """
    rs_core = pyrscore.PyRsCore()

    # pre-requisite is that the data file exists
    test_data = 'tests/data/Hidra_16-1_cor_log.h5'
    assert os.path.exists(test_data), 'File {} does not exist'.format(test_data)

    # load data
    test_hd_ws = rs_core.load_hidra_project(test_data, 'test core', False, True)
    assert test_hd_ws is not None

    # Get sub runs
    sub_runs = rs_core.reduction_service.get_sub_runs('test core')
    assert sub_runs is not None

    # Get sample logs
    log_names = rs_core.reduction_service.get_sample_logs_names('test core')
    assert isinstance(log_names, list)

    return

# In order to test the core methods for peak fitting and thus strain/stress calculation
# default testing directory is ..../PyRS/


print(os.getcwd())
# therefore it is not too hard to locate testing data
# TODO: convert BD_Data_Log.hdf5 to Hidra_BD_Data.hdf
test_data = 'tests/testdata/BD_Data_Log.hdf5'
print('Data file {0} exists? : {1}'.format(test_data, os.path.exists(test_data)))


def create_test_data(rs_core, src_data_set, target_data_set):
    """
    fit peaks and output to h5 data sets as the input of strain/stress
    :param rs_core
    :param src_data_set:
    :param target_data_set:
    :return:
    """
    for direction in ['e11', 'e22', 'e33']:
        # load
        src_h5_file = src_data_set[direction]
        data_key, message = rs_core.load_rs_raw(src_h5_file)
        print('Load {}: {}'.format(src_h5_file, message))
        # fit
        rs_core.fit_peaks(data_key, None, 'Gaussian', 'Linear', [80, 90])
        # save
        target_h5_file = target_data_set[direction]
        rs_core.save_peak_fit_result(data_key, src_h5_file, target_h5_file)
    # END-FOR

    return


def modify_test_strain_calculation():
    """
    main testing body to test the workflow to calculate strain
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {'e11': 'tests/temp/LD_Data_Log.hdf5',
                       'e22': 'tests/temp/BD_Data_Log.hdf5',
                       'e33': 'tests/temp/ND_Data_Log.hdf5'}

    # start a session
    rs_core.new_strain_stress_session('test strain/stress module', is_plane_strain=False,
                                      is_plane_stress=False)

    # TODO - 20180810 - d0 might not be a single value but changes along grids.
    # TODO   continue   So make it possible to accept d0 in a n x 3 matrix as (x, y, z)
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e11'], direction='e11')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e22'], direction='e22')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e33'], direction='e33')

    # convert peak positions to d-spacing
    rs_core .strain_stress_calculator.convert_peaks_positions()

    # check and align measurement points around
    # TODO FIXME - 20181001 - Make an individual method for ....  ---> New workflow!
    if True:
        # TODO FIXME - 20181001 - This is a new suite of methods to analyze the sample grids
        # ... ...
        # set the name of the sample log for grid positions
        pos_x_sample_names = {'e11': 'vx', 'e22': 'vx', 'e33': 'vx'}
        pos_y_sample_names = {'e11': 'vy', 'e22': 'vy', 'e33': 'vy'}
        pos_z_sample_names = {'e11': 'vz', 'e22': 'vz', 'e33': 'vz'}
        rs_core.strain_stress_calculator.set_grid_log_names(pos_x_sample_names, pos_y_sample_names, pos_z_sample_names)

        rs_core.strain_stress_calculator.check_grids_alignment()  # rename method
        rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)
    # END-Align-Grid

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute()

    # save
    rs_core.strain_stress_calculator.save_strain_stress('tests/temp/ss1.dat')

    # generate strain/stress grids
    grid_dict = {'Min': {'X': -130, 'Y': None, 'Z': 9.},
                 'Max': {'X': 130., 'Y': None, 'Z': None},
                 'NotUsed': {}}
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        'e22', user_defined=False, grids_dimension_dict=grid_dict)

    # convert to image of slice view...
    rs_core.strain_stress_calculator.export_strain_2d(grids)

    # export
    #  rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


def modify_test_copy_from_unconstrained_to_plane():
    """
    testing the use case such that the user copy a strain/stress calculator to another one
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {'e11': 'tests/temp/LD_Data_Log.hdf5',
                       'e22': 'tests/temp/BD_Data_Log.hdf5',
                       'e33': 'tests/temp/ND_Data_Log.hdf5'}

    # start a session
    rs_core.new_strain_stress_session('test strain/stress module', is_plane_strain=False,
                                      is_plane_stress=False)

    # set parameters
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e11'], direction='e11')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e22'], direction='e22')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e33'], direction='e33')

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print('Measuring points are not aligned: {}'.format(run_err))

    # create a plane strain/stress calculator
    ss_calculator = rs_core.strain_stress_calculator

    new_ss_calculator = ss_calculator.migrate(plane_strain=True, plane_stress=False)

    # re check
    new_ss_calculator.check_grids_alignment()
    new_ss_calculator.located_matched_grids(resolution=0.001)
    new_ss_calculator.execute()

    return


def modify_test_strain_stress_user_defined_grid():
    """
    Same as test_strain_calculation but using user-defined strain/stress grid
    :return:
    """
    # TESTME - 20180817 - Implement ASAP
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {'e11': 'tests/temp/LD_Data_Log.hdf5',
                       'e22': 'tests/temp/BD_Data_Log.hdf5',
                       'e33': 'tests/temp/ND_Data_Log.hdf5'}

    # start a session
    rs_core.new_strain_stress_session('test strain/stress module', is_plane_strain=False,
                                      is_plane_stress=False)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e11'], direction='e11')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e22'], direction='e22')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e33'], direction='e33')

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print('Measuring points are not aligned: {}'.format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core .strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {'Min': {'X': -130, 'Y': 0., 'Z': 9.},
                 'Max': {'X': 130., 'Y': 0., 'Z': 19},
                 'Resolution': {'X': 5.0, 'Y': None, 'Z': 1.0}}
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict)
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, 'center_d', maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids,
                                                                      peak_pos_d_vec=center_d_vec)

    # export
    #  rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


def modify_test_plane_strain():
    """
    Same as test_strain_calculation but using plane strain
    :return:
    """
    # TODO - 20180817 - Continue to implement
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {'e11': 'tests/temp/LD_Data_Log.hdf5',
                       'e22': 'tests/temp/BD_Data_Log.hdf5',
                       'e33': 'tests/temp/ND_Data_Log.hdf5'}

    # start a session
    rs_core.new_strain_stress_session('test strain/stress module', is_plane_strain=True,
                                      is_plane_stress=False)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e11'], direction='e11')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e22'], direction='e22')

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print('Measuring points are not aligned: {}'.format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core .strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {'Min': {'X': -130, 'Y': 0., 'Z': 9.},
                 'Max': {'X': 130., 'Y': 0., 'Z': 19},
                 'Resolution': {'X': 5.0, 'Y': None, 'Z': 1.0}}
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict)
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, 'center_d', maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids,
                                                                      peak_pos_d_vec=center_d_vec)

    assert strain_vec
    assert stress_vec

    return


def modify_test_plane_stress():
    """
    Same as test_strain_calculation but using plane stress
    :return:
    """
    # TODO - 20180817 - Continue to implement
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {'e11': 'tests/temp/LD_Data_Log.hdf5',
                       'e22': 'tests/temp/BD_Data_Log.hdf5',
                       'e33': 'tests/temp/ND_Data_Log.hdf5'}

    # start a session
    rs_core.new_strain_stress_session('test strain/stress module', is_plane_strain=False,
                                      is_plane_stress=True)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e11'], direction='e11')
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set['e22'], direction='e22')

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print('Measuring points are not aligned: {}'.format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core .strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {'Min': {'X': -130, 'Y': 0., 'Z': 9.},
                 'Max': {'X': 130., 'Y': 0., 'Z': 19},
                 'Resolution': {'X': 5.0, 'Y': None, 'Z': 1.0}}
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict)
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, 'center_d', maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids,
                                                                      peak_pos_d_vec=center_d_vec)

    assert strain_vec
    assert stress_vec

    return


if __name__ == '__main__':
    pytest.main()
