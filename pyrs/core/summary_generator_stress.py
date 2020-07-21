"""
This module generates reduction summary for stress in plain text CSV file
"""
from typing import List, Tuple, Dict

from pyrs.peaks import peak_collection
from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from builtins import int
from pyrs.peaks.peak_collection import PeakCollection


class SummaryGeneratorStress:
    """
    Generate summary to user about peak

    Format is like

    Header: p is the peak index
      # Hidra project files 
      # Peak Tags (PeakLabel[p-1])
      # Young\'s Modulus (E)[GPa] (PeakLabel[p-1])
      # Poisson\'s Ratio (PeakLabel[p-1]) 
    Body:
      vx_`PeakLabel[0]`, vy_`PeakLabel[p-1]`, vz_`PeakLabel[p-1]`, d-spacing_, d_spacing_22, d_spacing_33,  

      sub-run, vx, vy, vz, sx, sy, sz, phi, chi, omega, 2theta, S1 width, S1 height, S1 Distance, Radial distance,
      effective peak parameters

    """
    directions = ['11', '22', '33']
    fields_3dir = ['d', 'FWHM', 'Peak_Height', 'Strain', 'Stress']
    dir_tags = ['_Dir11_', '_Dir22_', '_Dir33_']

    def __init__(self, filename: str, stress_fields: [StressField], separator=','):
        """Initialization

        Parameters
        ----------
        filename: str
            Name of the ``.csv`` file to write
        stress_fields: list[StressField]
            list of stress fields to summarize, they must include desired directions
        separator: str
            The column separator in the output file
        """

        # do file name checks
        if not filename:
            raise RuntimeError('Failed to supply output filename for Stress CSV output')

        if not filename.endswith('.csv'):
            raise RuntimeError('Filename "{}" must end with ".csv"'.format(self._filename))

        # check for length of lists
        self._filename = str(filename)

        if not stress_fields:
            raise RuntimeError(
                'Error: stress_fields list can\'t be empty when creating stress file: ' + self._filename)

        self._stress_fields = stress_fields
        self._separator = separator

        # derived members
        self._project_tags, self._peak_tags, self._sorted_entries = self._init_project_peak_tags()

        # 0 -> 11, 1-> 22, 2-> 33
        # sorted_entries[dir][tag] -> List of stress IDs
        # self._sorted_entries:Dict[Dict[str,List[int]]]

        print('Projects:')
        print(self._project_tags)
        print('Peaks:')
        print(self._peak_tags)
        print('Sorted entries:')
        print(self._sorted_entries)

        self._global_coordinates, self._stress_field_index = self._init_global_coordinates()
        # print(self._global_coordinates)
        # print(self._stress_field_index)

    def _init_project_peak_tags(self) -> Tuple[List[str], List[str], Dict[str, Dict[str, List[int]]]]:
        """
            Init peak tags from strain fields in stress fields
            @return: peak_tags: List[str]
        """

        def _check_add_project_tag(filenames, global_project_tags):
            """
                Inline function to check existence of project tags in global_project_tags, 
                if not adds to it
            """
            for filename in filenames:
                tag = filename[filename.index('HB2B_') + 5: filename.index('.h5')]
                if tag not in global_project_tags:
                    global_project_tags.append(tag)

        def _check_add_peak_tag(peak_collections, global_peak_tags):
            """
                Inline function to check existence of collection peak tags in global_peak_tags,
                if not adds to it
            """
            for peak_collection in peak_collections:
                peak_tag = peak_collection.peak_tag
                if peak_tag not in global_peak_tags:
                    global_peak_tags.append(peak_tag)

        project_tags: List[str] = []
        peak_tags: List[str] = []
        sorted_entries: Dict[str, Dict[str, List[int]]] = {}

        for stress_field_id, stress_field in enumerate(self._stress_fields):
            for direction in SummaryGeneratorStress.directions:

                sorted_entries[direction] = {}  # initialize

                stress_field.select(direction)

                _check_add_project_tag(stress_field.strain.filenames, project_tags)
                _check_add_peak_tag(stress_field.strain.peak_collections, peak_tags)

                for filename in stress_field.strain.filenames:
                    tag = filename[filename.index('HB2B_') + 5: filename.index('.h5')]

                    if tag not in sorted_entries[direction].keys():
                        sorted_entries[direction][tag] = []

                    sorted_entries[direction][tag].append(stress_field_id)

        return project_tags, peak_tags, sorted_entries

    def _init_global_coordinates(self) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """
            Init global coordinates as list of [vx, vy, vz] and list of included stress fields per point
            Merge points if necessary for multiple meshes
            @return: Tuple: 1st -> global coordinate, 2nd -> list of original stress mesh per point
        """

        global_coordinates = self._stress_fields[0].coordinates
        stress_field_index = [[] for i in range(len(global_coordinates))]

        for i in range(0, len(global_coordinates)):
            stress_field_index[i].append(0)

        nfields = len(self._stress_fields)

        # no need to merge, only one field
        if nfields == 1:
            return global_coordinates, stress_field_index

        # TODO
        # merge looping through the rest of the fields
        for i in range(1, nfields):
            new_coordinates = self._stress_fields[i].coordinates

    def _write_csv_header(self, handle):

        # write projects names, peak_tags,
        # Young's modulus and Poisson ratio for each stress field

        # write projects names
        header = '# Hidra Project Names = '
        for project_tag in self._project_tags:
            header += str(project_tag) + ', '
        header = header[:-2] + '\n'

        # write peak_tags
        header += '# Peak Tags = '
        for peak_tag in self._peak_tags:
            header += str(peak_tag) + ', '
        header = header[:-2] + '\n'

        # write Young's Modulus for each peak
        header += '# Young\'s Modulus (E)[GPa] = '
        for stress_field in self._stress_fields:
            header += str(stress_field.youngs_modulus) + ', '
        header = header[:-2] + '\n'

        # write Poisson Ratio for each peak
        header += '# Poisson Ratio = '
        for stress_field in self._stress_fields:
            header += str(stress_field.poisson_ratio) + ', '
        header = header[:-2] + '\n'

        handle.write(header)

    def _write_csv_column_names(self, handle):

        column_names = 'vx, vy, vz, '

        # d0 doesn't depend on direction
        for project_tag in self._project_tags:
            column_names += 'd0_' + project_tag + ', '

        def __column_3dir(field, dir_tags, project_tags) -> str:
            names = ''
            for dir_tag in dir_tags:
                for project_tag in project_tags:
                    names += field + dir_tag + project_tag + ', '
            return names

        # directional variables

        for field_3dir in SummaryGeneratorStress.fields_3dir:
            column_names += __column_3dir(field_3dir, SummaryGeneratorStress.dir_tags, self._project_tags)

        column_names = column_names[:-2] + '\n'
        handle.write(column_names)

    def _write_csv_body(self, handle):

        def __write_d0(row: int) -> str:
            for direction in SummaryGeneratorStress.directions:
                for project_tag in self._project_tags:
                    peak_collection: PeakCollection = self._get_peak_collection(row, direction, project_tag)

                    if not peak_collection:
                        continue

                    # d0 doesn't depend on direction so just picking the first peak_collection
                    return str(peak_collection.get_d_reference()[0][row]) + ', '

        def __write_3dir(row: int, field) -> str:

            values = ''
            for direction in SummaryGeneratorStress.directions:
                for project_tag in self._project_tags:

                    if field == 'Strain':
                        strain_field = self._get_strain_field(row, direction, project_tag)
                        if not strain_field:
                            values += ', '
                            continue

                        values += str(strain_field.values[row])

                    elif field == 'Stress':
                        stress_field = self._get_stress_field(row, direction, project_tag)
                        if not stress_field:
                            values += ', '
                            continue

                        values += str(stress_field.values[row])

                    else:
                        peak_collection: PeakCollection = self._get_peak_collection(row, direction, project_tag)

                        if not peak_collection:
                            values += ', '
                            continue

                        if field == 'd':
                            values += str(peak_collection.get_dspacing_center()[0][row])
                        elif field == 'FWHM':
                            values += str(peak_collection.get_effective_params()[0]['FWHM'][row])
                        elif field == 'Peak_Height':
                            values += str(peak_collection.get_effective_params()[0]['Height'][row])

                    values += ', '

            return values

        lines = ''
        nrows = len(self._global_coordinates)
        for row in range(0, nrows):

            for global_coordinate in self._global_coordinates[row]:
                lines += str(global_coordinate) + ', '

            lines += __write_d0(row)
            lines += __write_3dir(row, 'd')
            lines += __write_3dir(row, 'FWHM')
            lines += __write_3dir(row, 'Peak_Height')
            lines += __write_3dir(row, 'Strain')
            lines += __write_3dir(row, 'Stress')

            lines = lines[:-2] + '\n'

        handle.write(lines)

    def write_csv(self):

        with open(self._filename, 'w') as handle:
            self._write_csv_header(handle)
            self._write_csv_column_names(handle)
            self._write_csv_body(handle)

    def _get_stress_field(self, row: int, direction: str, project_tag: str) -> StressField:
        """
            Returns a StressField for a particular row, direction and project_tag
        """
        stress_ids = self._sorted_entries[direction][project_tag]
        for stress_id in stress_ids:
            if stress_id in self._stress_field_index[row]:
                stress_field: StressField = self._stress_fields[stress_id]
                stress_field.select(direction)
                return stress_field

        return StressField()

    def _get_strain_field(self, row: int, direction: str, project_tag: str) -> StrainField:
        """
            Returns a StrainField for a particular row, direction and project_tag
        """
        stress_ids = self._sorted_entries[direction][project_tag]
        for stress_id in stress_ids:
            if stress_id in self._stress_field_index[row]:
                stress_field: StressField = self._stress_fields[stress_id]
                stress_field.select(direction)
                return stress_field.strain

        return StrainField()

    def _get_peak_collection(self, row: int, direction: str, project_tag: str) -> PeakCollection:
        """
            Returns a peak collection for a particular row, direction and project_tag
        """
        stress_ids = self._sorted_entries[direction][project_tag]
        for stress_id in stress_ids:
            if stress_id in self._stress_field_index[row]:
                stress_field = self._stress_fields[stress_id]
                stress_field.select(direction)
                return stress_field.strain.peak_collection

        return PeakCollection()
