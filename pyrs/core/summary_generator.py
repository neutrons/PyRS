"""
This module generates reduction summary for user in plain text CSV file
"""
from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS  # TODO get from the first peak collection
# import numpy as np

# Default summary titles shown in the CSV file. This is a list of tuples ot enforce order
# things to be found in the output file with
# key = logname, value=name in csv
HEADER_MAPPING = [('experiment_identifier', 'IPTS number'),
                  ('run_number', 'Run'),
                  ('run_title', 'Scan title'),
                  ('sample_name', 'Sample name'),
                  ('item_number', 'Item number'),  # BL11A:CS:ITEMS on powgen
                  ('hkl', 'HKL phase'),
                  ('StrainDirection', 'Strain direction'),  # was suggested to be "strain_dir"
                  ('mono_set', 'Monochromator setting'),
                  ('cal_file', 'Calibration file'),
                  ('project', 'Hidra project file'),
                  ('reduction', 'Manual vs auto reduction')]

# Default field for values - log_name:csv_name
DEFAULT_BODY_TITLES = ['vx', 'vy', 'vz', 'sx', 'sy', 'sz', 'phi', 'chi', 'omega', '2theta', 'S1width',
                       'S1height', 'S1distance', 'RadialDistance']


class SummaryGenerator:
    """
    Generate summary to user about peak

    Format is like

    Header:
      # IPTS number
      # Run
      # Scan title
      # Sample name
      # Item number
      # HKL phase
      # Strain direction
      # Monochromator setting (such as Si111... for wave length)
      # Calibration file, Hidra project file
      # Manual vs auto reduction
    Body:
      sub-run, vx, vy, vz, sx, sy, sz, phi, chi, omega, 2theta, S1 width, S1 height, S1 Distance, Radial distance,
      effective peak parameters

    """
    def __init__(self, filename, log_list=None, separator=','):
        """Initialization

        Parameters
        ----------
        filename: str
            Name of the ``.csv`` file to write
        log_list: list
            Names of the logs to write out
        separator: str
            The column separator in the output file
        """
        # Check input
        # self._check_header_title_value_match(header_title, header_values)

        if log_list is None:
            self._sample_log_list = DEFAULT_BODY_TITLES
        else:
            self._sample_log_list = log_list

        if not filename:
            raise RuntimeError('Failed to supply output filename')
        self._filename = str(filename)
        if not self._filename.endswith('.csv'):
            raise RuntimeError('Filename "{}" must end with ".csv"'.format(self._filename))

        self.separator = separator

        # name values that appear in the header
        self._header_information = dict()

        # logs that don't change within tolerance
        self._constant_logs = list()
        # logs that were requested but don't exist
        self._missing_logs = []
        # logs that were requested and do exist
        self._present_logs = []

    def setHeaderInformation(self, headervalues):
        '''This sets up the supplied information for the header without actually writing it'''

        for logname, _ in HEADER_MAPPING:
            if logname in headervalues.keys():
                self._header_information[logname] = headervalues[logname]

    def write_csv(self, sample_logs, peak_collections, tolerance=1E-10):
        """Export the CSV file

        Parameters
        ----------
        sample_logs: ~pyrs.dataobjects.SampleLogs
        peak_collections: list
            list of :py:obj:`~pyrs.peaks.PeakCollection`
        tolerance : float
            relative tolerance of variance to treat a sample log as a constant value
            and bring into extended header
        """
        # verify the same number of subruns everywhere
        for peak_collection in peak_collections:
            subruns = peak_collection.sub_runs
            if not sample_logs.matching_subruns(subruns):
                raise ValueError('Subruns from sample logs and peak {} do not match'.format(peak_collection.peak_tag))

        # determine what is constant and what is missing
        self._classify_logs(sample_logs, tolerance)

        # header has already been put together
        with open(self._filename, 'w') as handle:
            self._write_header_information(handle, sample_logs)
            self._write_header_missing(handle)
            self._write_header_constants(handle, sample_logs)
            self._write_column_names(handle, peak_collections)
            self._write_data(handle, sample_logs, peak_collections)

    def _classify_logs(self, sample_logs, tolerance):
        self._constant_logs = [logname for logname in sample_logs.constant_logs(tolerance)
                               if logname in self._sample_log_list]
        self._constant_logs.sort()  # keep the order stable for python3 tests

        # loop through all of the requested logs and classify as present or missing
        for logname in self._sample_log_list:
            if logname in sample_logs:
                self._present_logs.append(logname)
            else:
                self._missing_logs.append(logname)

    def _write_header_information(self, handle, sample_logs):
        '''Things that are supplied to SummaryGenerator.setHeaderInformation win out over what
        is found in the sample_logs'''
        # get the values that weren't specified from the logs
        for logname, _ in HEADER_MAPPING:
            # leave it alone if it was already set
            if logname not in self._header_information:
                # try to get the value from the logs or set it to empty string
                value = ''
                if logname in sample_logs:
                    value = sample_logs[logname][0]  # only use first value
                try:  # for python 3
                    value.decode()
                except (UnicodeDecodeError, AttributeError):
                    pass
                self._header_information[logname] = value

            # fix up particular values
            if self._header_information[logname]:
                if logname == 'run_number':
                    self._header_information[logname] = int(self._header_information[logname])
                elif logname == 'experiment_identifier':
                    try:  # for python 3
                        experiment_identifier = self._header_information[logname].decode()
                    except (UnicodeDecodeError, AttributeError):
                        experiment_identifier = self._header_information[logname]
                    self._header_information[logname] = experiment_identifier.split('-')[-1]

        # write out the text
        for logname, label in HEADER_MAPPING:
            value = self._header_information[logname]
            try:  # for python 3
                value = value.decode()
            except (UnicodeDecodeError, AttributeError):
                pass
            if value:
                line = ' = '.join((label, str(value)))
            else:
                line = label
            handle.write('# {}\n'.format(line))

    def _write_header_missing(self, handle):
        '''Add to the header a list of all missing logs'''
        if self._missing_logs:
            handle.write('# missing: {}\n'.format(', '.join(self._missing_logs)))

    def _write_header_constants(self, handle, sample_logs):
        '''Write only the sample logs that are constants into the header. These do not appear in the body.
        '''
        for name in self._constant_logs:
            try:  # for python 3
                value = sample_logs[name].decode()
            except (UnicodeDecodeError, AttributeError):
                value = sample_logs[name]
            try:
                handle.write('# {} = {:.5g} +/- {:.2g}\n'.format(name, value.mean(),
                                                                 value.std()))
            except TypeError:
                # strings don't have a "mean" or "std" so use the first value
                # this is intended for strings
                handle.write('# {} = {}\n'.format(name, value[0]))

    def _write_column_names(self, handle, peak_collections):
        '''This writes the names of all of the columns'''
        # the header line from the sample logs
        column_names = [name for name in self._present_logs
                        if name not in self._constant_logs]

        # the contribution from each peak
        for peak_collection in peak_collections:
            tag = peak_collection.peak_tag  # name of the peak
            # values first
            column_names.append('{}_dspacing_center'.format(tag))
            column_names.append('{}_strain'.format(tag))
            for param in EFFECTIVE_PEAK_PARAMETERS:
                column_names.append('{}_{}'.format(tag, param))
            # errors after values
            column_names.append('{}_dspacing_center_error'.format(tag))
            column_names.append('{}_strain_error'.format(tag))
            for param in EFFECTIVE_PEAK_PARAMETERS:
                column_names.append('{}_{}_error'.format(tag, param))
            column_names.append('{}_chisq'.format(tag))

        # subrun number goes in the very front
        column_names.insert(0, 'sub-run')

        handle.write(self.separator.join(column_names) + '\n')

    def _write_data(self, handle, sample_logs, peak_collections):
        '''Write out the actual data fields, ignoring what is constant'''
        log_names = [name for name in self._present_logs
                     if name not in self._constant_logs]

        for subrun_index in range(len(sample_logs.subruns)):
            line = []

            # sub-run goes in first
            line.append(str(sample_logs.subruns[subrun_index]))

            # then sample logs
            for name in log_names:
                line.append(str(sample_logs[name][subrun_index]))  # get by index rather than subrun

            for peak_collection in peak_collections:
                if peak_collection.get_exclude_subrun(subrun_index) is False:
                    fit_cost = peak_collection.fitting_costs
                    dspacing_center, dspacing_center_error = peak_collection.get_dspacing_center()
                    strain, strain_error = peak_collection.get_strain(units='microstrain')
                    values, errors = peak_collection.get_effective_params()
                    line.append(str(dspacing_center[subrun_index]))
                    line.append(str(strain[subrun_index]))
                    for value in values[subrun_index]:
                        line.append(str(value))
                    line.append(str(dspacing_center_error[subrun_index]))
                    line.append(str(strain_error[subrun_index]))
                    for value in errors[subrun_index]:
                        line.append(str(value))
                    line.append(str(fit_cost[subrun_index]))
                else:
                    values, errors = peak_collection.get_effective_params()
                    line.append('-')
                    line.append('-')
                    for value in values[subrun_index]:
                        line.append('-')
                    line.append('-')
                    line.append('-')
                    for value in errors[subrun_index]:
                        line.append('-')
                    line.append('-')

            handle.write(self.separator.join(line) + '\n')
