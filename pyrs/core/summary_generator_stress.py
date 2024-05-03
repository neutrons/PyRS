"""
This module generates reduction summary for stress in plain text CSV file
"""
from typing import Optional, Dict, Tuple
from builtins import int, isinstance
import math

from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from pyrs.dataobjects.fields import ScalarFieldSample
from pyrs.peaks.peak_collection import PeakCollection

import numpy as np
from pyrs.core.stress_facade import StressFacade


class SummaryGeneratorStress:
    """
        Generates a CSV summary from stress fields inputs from multiple project files on a grid point basis.
        From user story `for Grid Information CSV Output
        <github.com/neutrons/PyRS/blob/master/docs/planning/UserStory_GridInformation_CSVOutputs.md>`_
    """

    directions = ['11', '22', '33']
    # order of these fields matter on the CSV order
    fields_3dir = ['d', 'FWHM', 'Height', 'Strain', 'Stress']
    decimals = {'d0': 7, 'd': 7, 'FWHM': 2, 'Height': 12, 'Strain': 0, 'Stress': 0}

    def __init__(self, filename: str, stress_input):
        """Initialization

        Parameters
        ----------
        filename: str
            Name of the '.csv' file to write
        stress_input: can either be StressField or StressFacade
            stress field input to generate summary or full csv files
        """
        # do file name checks
        if not filename:
            raise RuntimeError('Failed to supply output filename for Stress CSV output')

        self._error_postfix = ' when creating stress file: ' + str(filename) + '\n'

        if not filename.endswith('.csv'):
            raise RuntimeError('File name must end with extension ".csv"' + self._error_postfix)

        # check for length of lists
        self._filename: str = str(filename)

        if isinstance(stress_input, StressField):
            self._stress_facade: StressFacade = StressFacade(stress_input)
            self._stress = stress_input
        elif isinstance(stress_input, StressFacade):
            self._stress_facade: StressFacade = stress_input  # type: ignore
            self._stress = self._stress_facade._stress
        else:
            raise RuntimeError(
                'Error: SummaryGeneratorStress stress input must be of type StressField or StressFacade'
                + self._error_postfix)

        self._strain33_is_calculated = False

        # check for filenames in StrainField per direction
        for direction in SummaryGeneratorStress.directions:
            strain = self._get_strain_field(direction)
            assert isinstance(strain, StrainField)

            # add exception if filenames is empty for 11 and 22 directions
            if not strain.filenames:
                if direction == '11' or direction == '22':
                    raise RuntimeError('StrainField filenames in direction ' + str(direction) +
                                       ' can\'t be empty for Stress CSV output ' + self._filename)
                elif direction == '33':
                    self._strain33_is_calculated = True

        # placeholder for caching peak_collection data as peak_collection.get_
        # function calls are expensive
        # key: type ( 'd0', 'd', 'FWHM', 'Peak_Height' )
        # value: Dict
        #        key: direction ( ''1' , '22', '33' )
        #        value: [0]-> value [1]-> error 1D numpy arrays
        self._peak_colllections_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        # initialize empty dictionaries keys
        self._peak_colllections_data['d'] = dict()
        for field in SummaryGeneratorStress.fields_3dir:
            self._peak_colllections_data[field] = dict()
            for direction in SummaryGeneratorStress.directions:
                self._peak_colllections_data[field][direction] = (np.ndarray, np.ndarray)  # type: ignore

        # used to cache summary csv fields
        self._stress_field: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._strain_field: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for direction in SummaryGeneratorStress.directions:
            self._stress_field[direction] = (np.ndarray, np.ndarray)  # type: ignore
            self._strain_field[direction] = (np.ndarray, np.ndarray)  # type: ignore

    def _write_csv_header(self, handle):
        """
          write projects names, peak_tags, Young's modulus and Poisson ratio
        """
        header = ''

        for direction in SummaryGeneratorStress.directions:

            if direction == '33' and self._strain33_is_calculated:
                line = '# Direction 33: calculated\n'

            else:
                line = '# Direction ' + str(direction) + ': '

                runs = self._stress_facade.runs(direction)
                for run in runs:
                    line += str(run) + ', '

                line = line[:-2] + '\n'

            header += line

        header += '# E: ' + str(self._stress.youngs_modulus) + '\n'
        header += '# v: ' + str(self._stress.poisson_ratio) + '\n'

        handle.write(header)

    def write_summary_csv(self):
        """
            Public function to generate a summary csv file for stress and input fields
        """
        def _write_summary_csv_column_names(handle):
            column_names = 'vx, vy, vz, d0, d0_error, '
            # directional variables
            for field_3dir in SummaryGeneratorStress.fields_3dir:

                field_name = field_3dir if field_3dir != 'Height' else 'Peak_Height'

                for direction in SummaryGeneratorStress.directions:
                    column_names += field_name + '_Dir' + direction + ', '
                    column_names += field_name + '_Dir' + direction + '_error, '

            column_names = column_names[:-2] + '\n'
            handle.write(column_names)
            return

        def _write_summary_csv_body(handle):

            def _write_field_3d(row: int, field: str):
                """
                   Writes 3 dimensional entries as value, error pairs per dimension
                   for an input field in a row
                   Args:
                       row: row index for a particular value and error array
                       field: name of the field from SummaryGeneratorStress.fields_3dir
                """

                entries = ''
                decimals = SummaryGeneratorStress.decimals[field]

                for direction in SummaryGeneratorStress.directions:

                    if field == 'Strain':

                        strain_value = self._strain_field[direction][0][row]
                        strain_error = self._strain_field[direction][1][row]
                        entries += self._write_number(strain_value, decimals) + \
                            self._write_number(strain_error, decimals)

                    elif field == 'Stress':

                        stress_value = self._stress_field[direction][0][row]
                        stress_error = self._stress_field[direction][1][row]
                        entries += self._write_number(stress_value, decimals) + \
                            self._write_number(stress_error, decimals)

                    else:
                        # by pass calculated direction as there are no measured quantities
                        if (self._stress_facade.stress_type == 'in-plane-strain' or
                            self._stress_facade.stress_type == 'in-plane-stress') and \
                                direction == '33':
                            entries += ', , '
                            continue

                        field_data = self._peak_colllections_data[field][direction]
                        if row >= len(field_data[0]):
                            entries += ', , '
                            continue

                        value = field_data[0][row]
                        error = field_data[1][row]
                        entries += self._write_number(value, decimals) + \
                            self._write_number(error, decimals)

                return entries

            # Function starts here
            self._recalc_peak_collections_data()

            body = ''

            # write for each row of the CSV body, first coordinates, d0 and
            # then fields in SummaryGeneratorStress.fields_3dir value, error per dimension
            for row, coordinate in enumerate(self._stress.coordinates):

                line = ''
                for coord in coordinate:
                    line += self._write_number(coord, 2)

                # d0 doesn't depend on direction so just picking the first peak_collection
                for direction in SummaryGeneratorStress.directions:
                    peak_collection = self._get_peak_collection(direction)
                    if not peak_collection:
                        continue

                    if row >= len(peak_collection.get_d_reference()[0]):
                        line += ', , '
                    else:
                        d0_value = peak_collection.get_d_reference()[0][row]
                        d0_error = peak_collection.get_d_reference()[1][row]
                        decimals = SummaryGeneratorStress.decimals['d0']
                        line += self._write_number(d0_value, decimals) + \
                            self._write_number(d0_error, decimals)
                        break

                # value error for fields_3dir = ['d', 'FWHM', 'Peak_Height', 'Strain', 'Stress']
                for field_3dir in SummaryGeneratorStress.fields_3dir:
                    line += _write_field_3d(row, field_3dir)

                line = line[:-2] + '\n'

                body += line

            handle.write(body)
            return

        # function starts here
        with open(self._filename, 'w') as handle:
            self._write_csv_header(handle)
            _write_summary_csv_column_names(handle)
            _write_summary_csv_body(handle)

        return

    def write_full_csv(self) -> None:
        """
            Public function to generate a full csv file for stress and input fields.
            Provides info for each run.
        """

        def _write_full_csv_column_names(handle):

            column_names = 'vx, vy, vz, d0, d0_error, '

            # directional variables
            for field_3dir in SummaryGeneratorStress.fields_3dir:

                field_name = field_3dir if field_3dir != 'Height' else 'Peak_Height'

                for direction in SummaryGeneratorStress.directions:

                    if field_name == 'Stress':
                        entry_base = field_name + '_Dir' + direction
                        column_names += entry_base + ', '
                        column_names += entry_base + '_error, '
                    else:
                        runs = self._stress_facade.runs(direction)
                        for run in runs:
                            entry_base = field_name + '_Dir' + direction + '_' + str(run)
                            column_names += entry_base + ', '
                            column_names += entry_base + '_error, '

            column_names = column_names[:-2] + '\n'
            handle.write(column_names)

            return

        def _write_full_csv_body(handle) -> None:

            body = ''

            # retrieve d_reference once as it implies calculations
            d0_scalar_field: ScalarFieldSample = self._stress_facade.d_reference
            d0_values = d0_scalar_field.values
            d0_errors = d0_scalar_field.errors

            # [direction][0] = values, [direction][1] = errors
            stress_fields: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            # [direction][run][0] = values, [direction][run][1] = errors
            strain_fields: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
            peak_data: Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

            for field_3dir in SummaryGeneratorStress.fields_3dir:

                for direction in SummaryGeneratorStress.directions:

                    self._stress_facade.selection = direction

                    if field_3dir == 'Stress':
                        stress_fields[direction] = (self._stress_facade.stress.values,
                                                    self._stress_facade.stress.errors)
                    else:
                        strain_fields[direction] = {}

                        if direction not in peak_data.keys():
                            peak_data[direction] = {}

                        runs = self._stress_facade.runs(direction)
                        for run in runs:

                            self._stress_facade.selection = run

                            if field_3dir == 'Strain':
                                strain_fields[direction][run] = (self._stress_facade.strain.values,
                                                                 self._stress_facade.strain.errors)
                            else:
                                if run not in peak_data[direction].keys():
                                    peak_data[direction][run] = {}

                                field_data = self._stress_facade.peak_parameter(field_3dir)
                                peak_data[direction][run][field_3dir] = (field_data.values,
                                                                         field_data.errors)
            # write for each row of the CSV body, first coordinates, d0 and
            # then fields in SummaryGeneratorStress.fields_3dir value, error per dimension
            for row, coordinate in enumerate(self._stress.coordinates):
                line = ''
                for coord in coordinate:
                    line += self._write_number(coord, 2)

                # d0
                if row >= len(d0_values):
                    line += ', , '
                else:
                    d0_value = d0_values[row]
                    d0_error = d0_errors[row]
                    decimals = SummaryGeneratorStress.decimals['d0']
                    line += self._write_number(d0_value, decimals) + \
                        self._write_number(d0_error, decimals)

                # value error for fields_3dir = ['d', 'FWHM', 'Peak_Height', 'Strain', 'Stress']
                for field_3dir in SummaryGeneratorStress.fields_3dir:
                    decimals = SummaryGeneratorStress.decimals[field_3dir]

                    for direction in SummaryGeneratorStress.directions:

                        self._stress_facade.selection = direction

                        if field_3dir == 'Stress':
                            stress_value = stress_fields[direction][0][row]
                            stress_error = stress_fields[direction][1][row]
                            line += self._write_number(stress_value, decimals) + \
                                self._write_number(stress_error, decimals)
                        else:
                            runs = self._stress_facade.runs(direction)
                            for run in runs:
                                self._stress_facade.selection = run

                                if field_3dir == 'Strain':

                                    strain_value = strain_fields[direction][run][0][row]
                                    strain_error = strain_fields[direction][run][1][row]
                                    line += self._write_number(strain_value, decimals) + \
                                        self._write_number(strain_error, decimals)
                                else:

                                    peak_parameter_value = peak_data[direction][run][field_3dir][0][row]
                                    peak_parameter_error = peak_data[direction][run][field_3dir][1][row]
                                    line += self._write_number(peak_parameter_value, decimals) + \
                                        self._write_number(peak_parameter_error, decimals)

                line = line[:-2] + '\n'

                body += line

            handle.write(body)

            return

        # function starts here
        with open(self._filename, 'w') as handle:
            self._write_csv_header(handle)
            _write_full_csv_column_names(handle)
            _write_full_csv_body(handle)

        return

    def _write_number(self, number, decimal_digits=12) -> str:

        if math.isnan(number):
            return ', '

        output = ''
        if decimal_digits == 12:
            output = f'{number:.12f}' + ', '
        elif decimal_digits == 0:
            output = f'{number:.0f}' + ', '
        elif decimal_digits == 2:
            output = f'{number:.2f}' + ', '
        elif decimal_digits == 7:
            output = f'{number:.7f}' + ', '
        elif decimal_digits == 6:
            output = f'{number:.6f}' + ', '
        else:
            raise RuntimeError('ERROR: ' + str(decimal_digits) + ' decimal digits not supported in CSV file')

        return output

    def _get_strain_field(self, direction: str) -> Optional[StrainField]:
        """
            Returns a StrainField for a particular direction from self._stress
        """
        stress: StressField = self._stress
        stress.select(direction)
        strain = self._stress.strain
        if isinstance(strain, StrainField):
            return strain

        return None

    def _get_peak_collection(self, direction: str) -> Optional[PeakCollection]:
        """
            Returns a peak_collection for a particular direction from self._stress
        """
        strain = self._get_strain_field(direction)
        if isinstance(strain, StrainField):
            return strain.peak_collections[0]

        return None

    def _recalc_peak_collections_data(self):

        for field in SummaryGeneratorStress.fields_3dir:
            for direction in SummaryGeneratorStress.directions:

                self._stress_facade.selection = direction

                if field == 'Stress':
                    self._stress_field[direction] = (self._stress_facade.stress.values,
                                                     self._stress_facade.stress.errors)

                elif field == 'Strain':
                    self._strain_field[direction] = (self._stress_facade.strain.values,
                                                     self._stress_facade.strain.errors)

                else:
                    if (self._stress_facade.stress_type == 'in-plane-strain' or
                        self._stress_facade.stress_type == 'in-plane-stress') and \
                            direction == '33':
                        continue

                    field_data = self._stress_facade.peak_parameter(field)
                    self._peak_colllections_data[field][direction] = (field_data.values,
                                                                      field_data.errors)
