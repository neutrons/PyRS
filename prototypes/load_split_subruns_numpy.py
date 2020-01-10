# This is a numpy version for prototyping to load NeXus and split events for sub runs
# by numpy and hdf5
import h5py
import numpy as np
from matplotlib import pyplot as plt
import datetime
from mantid.simpleapi import Load, FilterByLogValue, DeleteWorkspace
import os


def load_nexus(nexus_file_name):
    nexus_h5 = h5py.File(nexus_file_name, 'r')
    return nexus_h5

def get_scan_indexes(nexus_h5):
    scan_index_times = nexus_h5['entry']['DASlogs']['scan_index']['time'].value
    scan_index_values = nexus_h5['entry']['DASlogs']['scan_index']['value'].value

    if scan_index_values[0] == 0:
        scan_index_times = scan_index_times[1:]
        scan_index_values = scan_index_values[1:]

    if scan_index_times.shape != scan_index_values.shape:
        raise RuntimeError('Scan index time and value not in same shape')
    if scan_index_times.shape[0] % 2 == 1:
        raise RuntimeError("Scan index are not in (1, 0) pair")

    return scan_index_times, scan_index_values


def split_sub_runs(nexus_h5, scan_index_times, scan_index_values):
    # pulse times
    pulse_time_array = nexus_h5['entry']['bank1_events']['event_time_zero'].value

    # search scan index boundaries in pulse time array
    subrun_pulseindex_array = np.searchsorted(pulse_time_array, scan_index_times)

    # get event  index array: same size as pulse times
    event_index_array = nexus_h5['entry']['bank1_events']['event_index'].value
    event_id_array = nexus_h5['entry']['bank1_events']['event_id'].value

    # histogram boundaries
    # bound_x = np.arange(1024 ** 2 + 1).astype(float) - 0.1

    # split data
    num_sub_runs = scan_index_values.shape[0] / 2
    # sub_run_data_set = np.ndarray((num_sub_runs, 1024**2), dtype=float)
    sub_run_counts_dict = dict()

    for i_sub_run in range(num_sub_runs):
        # get the start and stop index in pulse array
        start_pulse_index = subrun_pulseindex_array[2 * i_sub_run]
        stop_pulse_index = subrun_pulseindex_array[2 * i_sub_run + 1]

        # get start andn stop event ID from event index array
        start_event_id = event_index_array[start_pulse_index]
        if stop_pulse_index >= event_index_array.size:
            print('[WARNING] for sub run {} out of {}, stop pulse index {} is out of boundary of {}'
                  ''.format(i_sub_run, num_sub_runs, stop_pulse_index, event_index_array.shape))
            # stop_pulse_index = event_index_array.size - 1
            # supposed to be the last pulse and thus use the last + 1 event ID's index
            stop_event_id = event_index_array.shape[0]
        else:
            # natural one
            stop_event_id = event_index_array[stop_pulse_index]

        # get sub set of the events falling into this range
        sub_run_events = event_id_array[start_event_id:stop_event_id]
        # convert to float
        counts = sub_run_events.astype(float)

        # histogram
        # hist = np.histogram(counts, bound_x)[0]
        hist = np.bincount(sub_run_events, minlength=1024**2)
        # sub_run_data_set[i_sub_run] = hist
        sub_run_counts_dict[int(scan_index_values[2 * i_sub_run])] = hist

    return sub_run_counts_dict


def verify(nexus, sub_run_list, sub_run_data_dict):
    # Load
    ws = Load(Filename=nexus, OutputWorkspace=os.path.basename(nexus).split('.')[0])

    pp_str = ''

    for sub_run in sub_run_list:
        sub_run_ws = FilterByLogValue(InputWorkspace=ws,
                                      OutputWorkspace=str(ws) + '_{}'.format(sub_run),
                                      LogName='scan_index',
                                      LogBoundary='Left',
                                      MinimumValue=float(sub_run) - .25,
                                      MaximumValue=float(sub_run) + .25)
        counts_vec = sub_run_ws.extractY().reshape((1024**2,))

        diff = np.abs(sub_run_data_dict[sub_run] - counts_vec)
        print('sub run {}:  counts = {}, total difference = {}, maximum difference = {}'
              ''.format(sub_run, np.sum(counts_vec), diff.sum(), diff.max()))

        pp_str += 'sub run {}:  counts = {}, total difference = {}, maximum difference = {}\n' \
                  ''.format(sub_run, np.sum(counts_vec), diff.sum(), diff.max())

        DeleteWorkspace(Workspace=sub_run_ws)
    # END

    print(pp_str)


def main():

    start_time = datetime.datetime.now()

    nexus = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1060.nxs.h5'
    nexus_h5 = load_nexus(nexus)

    scan_index_times, scan_index_values = get_scan_indexes(nexus_h5)

    frames = split_sub_runs(nexus_h5, scan_index_times, scan_index_values)

    stop_time = datetime.datetime.now()

    duration = (stop_time - start_time).total_seconds()
    print('Processing time = {}; Per sub run = {}'.format(duration, duration * 1. / len(frames)))

    verify(nexus, range(100, 117), frames)


if __name__ == '__main__':
    main()
