#!/usr/bin/python
""""
Convert the "old" HB2B data, now used to test peak fitting, to new HydraProject format

*How to run*
1. Add PyRS path to python path (refer to pyrsdev.sh)
1. Run this script

""" 
from pyrs.utilities import rs_scan_io
from pyrs.utilities import rs_project_file


def main():
    """
    Main method to convert diffraction data in the old HDF5 format
    :return:
    """
    # Load source data in 'old' format
    source_h5 = 'tests/testdata/16-1_TD.cor_Log.hdf5'
    reader = rs_scan_io.DiffractionDataFile()
    diff_data_dict, sample_logs = reader.load_rs_file(source_h5)

    # Create a Hydra project
    target_project_file_name = 'tests/testdata/Hydra_16-1_cor_log.hdf5'
    target_file = rs_project_file.HydraProjectFile(target_project_file_name,
                                                   rs_project_file.HydraProjectFileMode.OVERWRITE)

    # Add (reduced) diffraction data
    for sub_run_index in diff_data_dict.keys():
        two_theta_vector, intensity_vector = diff_data_dict[sub_run_index]

        print ('[DB...BAT] {}; {}'.format(two_theta_vector, intensity_vector))

        target_file.set_2theta_diffraction_data(sub_run_index, two_theta_vector, intensity_vector)

    # END-FOR

    target_file.save_hydra_project()

    return


if __name__ == '__main__':
    main()