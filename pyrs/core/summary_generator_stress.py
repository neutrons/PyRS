"""
This module generates reduction summary for stress in plain text CSV file
"""
from typing import Optional, Dict, Tuple
from builtins import int, isinstance
import math

from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from pyrs.peaks.peak_collection import PeakCollection

import numpy as np
from pyrs.core.stress_facade import StressFacade
from unicodedata import decimal


class SummaryGeneratorStress:
    """
        Generates a CSV summary from stress fields inputs from multiple project files on a grid point basis.
        From user story `for Grid Information CSV Output
        <github.com/neutrons/PyRS/blob/master/docs/planning/UserStory_GridInformation_CSVOutputs.md>`_
    """

    directions = ['11', '22', '33']
    fields_3dir = ['d', 'FWHM', 'Height', 'Strain', 'Stress']
    decimals = {'d0': 7, 'd': 7, 'FWHM': 2, 'Height': 12, 'Strain': 6, 'Stress': 6}

    def __init__(self, filename: str, stress: StressField):
        """Initialization

        Parameters
        ----------
        filename: str
            Name of the '.csv' file to write
        stress: StressField
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

        if isinstance(stress, StressField):
            self._stress_facade = StressFacade(stress)
            self._stress = stress
        elif isinstance(stress, StressFacade):
            self._stress_facade = stress
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
                self._peak_colllections_data[field][direction] = (np.ndarray, np.ndarray)

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
                        # TODO add check for strain?
                        strain = self._get_strain_field(direction)
                        assert(isinstance(strain, StrainField))
                        strain_field = strain.field
                        strain_value = strain_field.values[row]
                        strain_error = strain_field.errors[row]
                        entries += self._write_number(strain_value, decimals) + \
                            self._write_number(strain_error, decimals)

                    elif field == 'Stress':
                        self._stress.select(direction)
                        stress_value = self._stress.values[row]
                        stress_error = self._stress.errors[row]
                        entries += self._write_number(stress_value, decimals) + \
                            self._write_number(stress_error, decimals)

                    else:
                        peak_collection = self._get_peak_collection(direction)
                        if not isinstance(peak_collection, PeakCollection):
                            entries += ', , '
                            continue

                        field_data = self._peak_colllections_data[field][direction]
                        if row >= len(field_data[0]):
                            entries += ', , '
                        else:
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

    def write_full_csv(self):

        def _write_full_csv_column_names(handle):

            column_names = 'vx, vy, vz, d0, d0_error, '

            # directional variables
            for field_3dir in SummaryGeneratorStress.fields_3dir:

                field_name = field_3dir if field_3dir != 'Height' else 'Peak_Height'

                for direction in SummaryGeneratorStress.directions:
                    runs = self._stress_facade.runs(direction)

                    for run in runs:
                        entry_base = field_name + '_Dir' + direction + '_' + str(run)
                        column_names += entry_base + ', '
                        column_names += entry_base + '_error, '

            column_names = column_names[:-2] + '\n'
            handle.write(column_names)

            return

        def _write_full_csv_body(handle):

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

        # initialize
        peak_collection11 = self._get_peak_collection('11')
        assert isinstance(peak_collection11, PeakCollection)
        self._peak_colllections_data['d']['11'] = peak_collection11.get_d_reference()

        for field in SummaryGeneratorStress.fields_3dir:
            for direction in SummaryGeneratorStress.directions:

                if field == 'Stress' or field == 'Strain':
                    continue

                peak_collection = self._get_peak_collection(direction)
                if not isinstance(peak_collection, PeakCollection):
                    continue

                if field == 'd':
                    self._peak_colllections_data[field][direction] = peak_collection.get_dspacing_center()
                else:
                    self._peak_colllections_data[field][direction] = \
                        (peak_collection.get_effective_params()[0][field],
                         peak_collection.get_effective_params()[1][field])
