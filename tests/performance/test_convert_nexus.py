import os
from pyrs.nexus.split_sub_runs import load_split_nexus_python
import pytest
import datetime


def test_convert_nexus():
    """Test performance on converting NeXus to HidraWorkspace

    Using IPTS-???? Run ??? to ???.

    Returns
    -------

    """
    ipts_number = 22731
    run_start = 1102
    run_stop = 1155

    ipts_dir = '/HFIR/HB2B/IPTS-{}/nexus'.format(ipts_number)
    if os.path.exists(ipts_dir) is False:
        pytest.pytest.skip('Performance test is skipped due to no access to {}'.format(ipts_dir))

    # a record dictionary for reduction time, key = time (second)
    time_run_dict = dict()
    non_exist_nexuses = list()
    for run_number in range(run_start, run_stop):
        # create NeXus file name
        nexus_name = os.path.join(ipts_dir, 'HB2B_{}.nxs.h5'.format(run_number))
        # skip non-existing NeXus
        if not os.path.exists(nexus_name):
            non_exist_nexuses.append((run_number, nexus_name))

        # load and split
        start_time = datetime.datetime.now()
        counts_dict, sample_log_dict = load_split_nexus_python(nexus_name)
        stop_time = datetime.datetime.now()

        time_run_dict[(stop_time - start_time).total_seconds()] = run_number, nexus_name, counts_dict.keys()
    # END-FOR

    # Non-existing NeXus
    if len(non_exist_nexuses) > 0:
        print(non_exist_nexuses)

    # Sort by time
    run_time_list = sorted(time_run_dict.keys(), reverse=True)
    for i in range(min(10, len(run_time_list))):
        run_i, nexus_i, sub_runs_i = time_run_dict[run_time_list[i]]
        print('Run {}: Time = {}, Number of sub runs = {}'.format(run_i, run_time_list[i], len(sub_runs_i)))

    return
