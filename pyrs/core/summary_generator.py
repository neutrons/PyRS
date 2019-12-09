"""
This module generates reduction summary for user in plain text CSV file
"""
from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS  # TODO get from the first peak collection

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


class SummaryGenerator(object):
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

        # To write
        self._fit_engine = None
        self._header_log_section = None
        self._body_section = None

    def setHeaderInformation(self, headervalues):
        '''This creates a string to write to the file without actually writing it'''

        for logname, _ in HEADER_MAPPING:
            if logname in headervalues.keys():
                self._header_information[logname] = headervalues[logname]

    def write_csv(self, sample_logs, peak_collections, tolerance=1E-10):
        """Export the CSV file

        Parameters
        ----------
        sep: str
            string to separate fields in the body of the file
        tolerance : float
            relative tolerance of variance to treat a sample log as a constant value
            and bring into extended header
        """
        # verify the same number of subruns everywhere
        for peak_collection in peak_collections:
            _, subruns, _, _, _ = peak_collection.get_effective_parameters_values()
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

        # loop through all of the requested logs and classify as present or missing
        for logname in self._sample_log_list:
            if logname in sample_logs:
                self._present_logs.append(logname)
            else:
                self._missing_logs.append(logname)

    def _write_header_information(self, handle, sample_logs):
        # get the values that weren't specified from the logs
        for logname, _ in HEADER_MAPPING:
            # leave it alone if it was already set
            if logname not in self._header_information:
                # try to get the value from the logs or set it to empty string
                value = ''
                if logname in sample_logs:
                    value = sample_logs[logname][0]  # only use first value
                self._header_information[logname] = value

            # fix up particular values
            if self._header_information[logname]:
                if logname == 'run_number':
                    self._header_information[logname] = int(self._header_information[logname])
                elif logname == 'experiment_identifier':
                    self._header_information[logname] = self._header_information[logname].split('-')[-1]

        # write out the text
        for logname, label in HEADER_MAPPING:
            value = self._header_information[logname]
            if value:
                line = ' = '.join((label, str(value)))
            else:
                line = label
            handle.write('# {}\n'.format(line))

    def _write_header_missing(self, handle):
        if self._missing_logs:
            handle.write('# missing: {}\n'.format(', '.join(self._missing_logs)))

    def _write_header_constants(self, handle, sample_logs):
        """Write only the sample logs that are constants into the header. These should not appear in the body.
        """
        for name in self._constant_logs:
            handle.write('# {} = {:.5g} +/- {:.2g}\n'.format(name, sample_logs[name].mean(), sample_logs[name].std()))

    def _write_column_names(self, handle, peak_collections):
        # the header line from the sample logs
        column_names = [name for name in self._present_logs
                        if name not in self._constant_logs]

        # the contribution from each peak
        for peak_collection in peak_collections:
            tag = peak_collection.peak_tag  # name of the peak
            # values first
            for param in EFFECTIVE_PEAK_PARAMETERS:
                column_names.append('{}_{}'.format(tag, param))
            # errors after values
            for param in EFFECTIVE_PEAK_PARAMETERS:
                column_names.append('{}_{}_error'.format(tag, param))
            column_names.append('{}_chisq'.format(tag))

        # subrun number goes in the very front
        column_names.insert(0, 'sub-run')

        handle.write(self.separator.join(column_names) + '\n')

    def _write_data(self, handle, sample_logs, peak_collections):
        log_names = [name for name in self._present_logs
                     if name not in self._constant_logs]

        for i in range(len(sample_logs.subruns)):
            line = []

            # sub-run goes in first
            line.append(str(sample_logs.subruns[i]))

            # then sample logs
            for name in log_names:
                line.append(str(sample_logs[name][i]))  # get by index rather than subrun

            for peak_collection in peak_collections:
                _, _, fit_cost, values, errors = peak_collection.get_effective_parameters_values()
                for j in range(values.shape[0]):  # number of parameters
                    line.append(str(values[j, i]))
                for j in range(errors.shape[0]):  # number of parameters
                    line.append(str(errors[j, i]))
                line.append(str(fit_cost[i]))

            handle.write(self.separator.join(line) + '\n')
