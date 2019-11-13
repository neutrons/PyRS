"""
This module generates reduction summary for user in plain text CSV file
"""
import collections
import numpy as np

# Default summary headers
SummaryHeaders = collections.namedtuple('SummaryHeader', 'ipts run title sample_name item_number hkl strain_dir '
                                                         'mono_set cal_file project reduction')

# Default summary titles shown in the CSV file
Header_Titles = [('IPTS number', 'ipts'),
                 ('Run', 'run'),
                 ('Scan title', 'title'),
                 ('Sample name', 'sample_name'),
                 ('Item number', 'item_number'),
                 ('HKL phase', 'hkl'),
                 ('Strain direction', 'strain_dir'),
                 ('Monochromator setting', 'mono_set'),
                 ('Calibration file', 'cal_file'),
                 ('Hidra project file', 'project'),
                 ('Manual vs auto reduction', 'reduction')]

# Default field for values
Body_Titles = [
    ('sub-run', 'sub_run'),
    ('vx', 'vx'),
    ('vy', 'vy'),
    ('vz', 'vz'),
    ('sx', 'sx'),
    ('sy', 'sy'),
    ('sz', 'sz'),
    ('phi', 'phi'),
    ('chi', 'chi'),
    ('omega', 'omega'),
    ('2theta', '2theta'),
    ('S1 width', 'Unknown'),
    ('S1 height', 'Unknown'),
    ('S1 Distance', 'Unknown'),
    ('Radial distance', 'Unknown')]


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
    def __init__(self, header_title, header_values, log_tup_list):
        """Initialization

        Parameters
        ----------
        header_title : List of 2-tuples
            Ordered list of 2-tuple (as parameter titles and field in namedtuple for values) written in CSV header.
        header_values : ~collections.namedtuple
            Containing value
        log_tup_list : List
            sample log tuples as sample log name in title title and sample log name in file
        """
        # Check input
        self._check_header_title_value_match(header_title, header_values)

        # Set
        self._header_title = header_title
        self._header_value = header_values
        self._sample_log_list = log_tup_list

        # To write
        self._fit_engine = None
        self._header_info_section = None
        self._header_log_section = None
        self._body_section = None

        return

    @staticmethod
    def _check_header_title_value_match(header_title_tuples, header_values):
        """Check whether all the titles have the field existing in header_values

        Exceptions
        RuntimeError : if not match

        Parameters
        ----------
        header_title_tuples : list
            list of 2-tuple
        header_values : ~collections.namedtuple
            sample log value

        Returns
        -------
        None

        """
        for header_title, value_field in header_title_tuples:
            if value_field not in header_values._fields:
                raise RuntimeError('Header item {} does not have correct field {} in given namedtuple ({})'
                                   ''.format(header_title, value_field, header_values._fields))

        return

    def export_to_csv(self, peak_fit_engine, csv_file_name, sep=',', tolerance=1E-10):
        """Export the CSV file

        Parameters
        ----------
        peak_fit_engine :  ~core.PeakFitEngine
            PeakFitEngine where the peaks' parameter are found
        csv_file_name
        sep
        tolerance : float
            tolerance of variance to treat a sample log as a non-changed value

        Returns
        -------

        """
        # Set fit engine
        self._fit_engine = peak_fit_engine

        #
        self._write_info_header()

        #
        self._write_sample_logs(sep, tolerance)

        return

    def _write_info_header(self):
        """Write

        Returns
        -------

        """
        # Set start
        self._header_info_section = ''

        for index, field in enumerate(self._header_title):
            self._header_info_section += '# {} = {}\n'.format(field, self._header_value[index])

        return

    def _write_sample_logs(self, sep, tolerance):
        """Write sample logs

        Exceptions
        ----------

        Parameters
        ----------
        sep : str
            separator in CSV
        tolerance : float
            tolerance of variance to treat a sample log as a non-changed value

        Returns
        -------
        None

        """
        # Init
        self._header_log_section = ''

        # Examine the values to write out as column or field
        column_list = list()  # item: log title, log value array

        for index in range(len(self._sample_log_list)):
            log_title, log_name = self._sample_log_list[index]

            if log_title == 'sub-run':
                # sub run is a special case
                column_list.append((log_title, self._fit_engine.get_hidra_workspace().get_sub_runs()))
            else:
                # regular sample log value
                try:
                    # get the log value as array
                    log_value_array = self._fit_engine.get_hidra_workspace().get_sample_log_values()
                    # calculate the variance
                    log_var = np.var(log_value_array)
                    if log_var < tolerance:
                        # less than tolerance: fixed value and add to heder
                        self._header_log_section += '# {} = {} +/- {}\n'.format(log_title, np.average(log_value_array),
                                                                              log_var)
                    else:
                        # add the column list for future processing
                        column_list.append((log_title, log_value_array))
                    # END-IF
                except RuntimeError:
                    # sample log value does not exist
                    # add note to header
                    self._header_log_section += '# {} = N/A\n'.format(log_title)
            # END-IF-ELSE
        # END-FOR

        # Convert to body
        num_columns = len(column_list)
        num_rows = column_list[0][1].size
        # CSV body header
        body_header = '# '
        for index in range(num_columns):
            body_header += column_list[index][0]
            if index < num_columns - 1:
                body_header += '{}'.format(sep)
        # END-FOR

        # CSV body
        csv_body = ''
        for row_index in range(num_rows):
            # construct a row
            line_buf = ''
            for col_index in range(num_columns):
                # add value
                line_buf += column_list[col_index][1][row_index]
                # add separator unless last one
                if col_index < num_columns - 1:
                    line_buf += sep
            # END-FOR (COL)

            # add to body
            csv_body += line_buf
            # add new line unless last row
            if row_index < num_rows - 1:
                csv_body += '\n'
        # END-FOR

        self._body_section = body_header + '\n' + csv_body

        return
