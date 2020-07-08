"""
This module generates reduction summary for stress in plain text CSV file
"""
from typing import List, Tuple

from pyrs.peaks import peak_collection
from pyrs.dataobjects.fields import StressField
from importlib_metadata.docs.conf import project
from pyrs.peaks.peak_collection import PeakCollection
from builtins import isinstance


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
        self._project_tags, self._peak_tags = self._init_project_peak_tags()

        # print('Projects:')
        # print(self._project_tags)
        # print('Peaks:')
        # print(self._peak_tags)

        self._global_coordinates, self._stress_field_index = self._init_global_coordinates()
        # print(self._global_coordinates)
        # print(self._stress_field_index)

    def _init_project_peak_tags(self) -> Tuple[List[str], List[str]]:
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
                tag = filename[filename.index('HB2B_') + 4: filename.index('.h5')]
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

        directions = ['11', '22', '33']

        for stress_field in self._stress_fields:
            for direction in directions:
                stress_field.select(direction)

                # if stress_field.strain.filenames:
                #     _check_add_project_tag(stress_field.strain.filenames, global_project_tags)

                if stress_field.strain.peak_collections:
                    _check_add_peak_tag(stress_field.strain.peak_collections, peak_tags)

        # FIXME Mocking project and peak tags for now
        if not project_tags:
            project_tags = ['1320']

        if not peak_tags:
            peak_tags = ['peak0']

        return project_tags, peak_tags

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

    def write_csv_column_names(self, handle):

        line = ''
        for peak_collection in self._peak_tags[0]:
            peak_tag = peak_collection._tag
            line += 'vx_' + peak_tag
            line += 'vy_' + peak_tag
            line += 'vz_' + peak_tag

    def write_csv(self):

        with open(self._filename, 'w') as handle:
            self._write_csv_header(handle)
            # self._write_csv_column_names(handle)
