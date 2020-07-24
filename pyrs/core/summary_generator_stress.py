"""
This module generates reduction summary for stress in plain text CSV file
"""
from typing import Optional

from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from builtins import int, isinstance
from pyrs.peaks.peak_collection import PeakCollection
import math


class SummaryGeneratorStress:
    """
        Generates a CSV summary from stress fields inputs from multiple project files on a grid point basis.
    """

    directions = ['11', '22', '33']
    fields_3dir = ['d', 'FWHM', 'Peak_Height', 'Strain', 'Stress']

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

        if not stress:
            raise RuntimeError(
                'Error: stress of type StressField input can\'t be invalid' + self._error_postfix)

        self._stress = stress

        # check for filenames in StrainField per direction
        for direction in SummaryGeneratorStress.directions:
            strain = self._get_strain_field(direction)
            assert isinstance(strain, StrainField)

            # add exception if filenames is empty
            if not strain.filenames:
                raise RuntimeError('StrainField filenames in direction ' + str(direction) +
                                   'can\'t be empty for Stress CSV output ' + self._filename)

    def _write_csv_header(self, handle):
        """
          write projects names, peak_tags, Young's modulus and Poisson ratio
        """
        header = ''

        for direction in SummaryGeneratorStress.directions:
            strain = self._get_strain_field(direction)
            assert isinstance(strain, StrainField)

            line = '# Direction ' + str(direction) + ': '
            for filename in strain.filenames:
                run_number = filename[filename.index('HB2B_') + 5: filename.index('.h5')]
                line += str(run_number) + ', '

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
            column_names = 'vx, vy, vz, d0, d0_error '
            # directional variables
            for field_3dir in SummaryGeneratorStress.fields_3dir:
                for direction in SummaryGeneratorStress.directions:
                    column_names += field_3dir + '_Dir' + direction + ', '
                    column_names += field_3dir + '_Dir' + direction + '_error, '

            column_names = column_names[:-2] + '\n'
            handle.write(column_names)
            return

        def _write_summary_csv_body(handle):

            def _write_number(number) -> str:
                if math.isnan(number):
                    return ', '
                return str(number) + ', '

            def _write_field_3d(row: int, field: str):
                entries = ''
                for direction in SummaryGeneratorStress.directions:
                    if field == 'Strain':
                        # TODO add check for strain?
                        strain = self._get_strain_field(direction)
                        assert(isinstance(strain, StrainField))

                        strain_value = strain.values[row]
                        strain_error = strain.errors[row]
                        entries += _write_number(strain_value) + _write_number(strain_error)

                    elif field == 'Stress':
                        self._stress.select(direction)
                        stress_value = self._stress.values[row]
                        stress_error = self._stress.errors[row]
                        entries += _write_number(stress_value) + _write_number(stress_error)

                    else:
                        peak_collection = self._get_peak_collection(direction)
                        assert(isinstance(peak_collection, PeakCollection))

                        if field == 'd':
                            d_value = peak_collection.get_dspacing_center()[0][row]
                            d_error = peak_collection.get_dspacing_center()[1][row]
                            entries += _write_number(d_value) + _write_number(d_error)

                        elif field == 'FWHM':
                            fwhm_value = peak_collection.get_effective_params()[0]['FWHM'][row]
                            fwhm_error = peak_collection.get_effective_params()[1]['FWHM'][row]
                            entries += _write_number(fwhm_value) + _write_number(fwhm_error)

                        elif field == 'Peak_Height':
                            height_value = peak_collection.get_effective_params()[0]['Height'][row]
                            height_error = peak_collection.get_effective_params()[1]['Height'][row]
                            entries += _write_number(height_value) + _write_number(height_error)

                return entries

            # Function starts here
            body = ''
            for row, coordinate in enumerate(self._stress.coordinates):

                line = str(coordinate[0]) + ', ' + str(coordinate[1]) + ', ' + str(coordinate[2]) + ', '

                # d0 doesn't depend on direction so just picking the first peak_collection
                for direction in SummaryGeneratorStress.directions:
                    peak_collection = self._get_peak_collection(direction)
                    if not peak_collection:
                        continue
                    d0_value = peak_collection.get_d_reference()[0][row]
                    d0_error = peak_collection.get_d_reference()[1][row]

                    line += _write_number(d0_value) + _write_number(d0_error)
                    break

                # fields_3dir = ['d', 'FWHM', 'Peak_Height', 'Strain', 'Stress']
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
            return strain.peak_collection

        return None
