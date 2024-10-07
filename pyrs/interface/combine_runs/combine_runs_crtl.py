import numpy as np
from pyrs.utilities import get_input_project_file  # type: ignore


class CombineRunsCrtl:
    def __init__(self, _model):
        self._model = _model

    def parse_file_path(self, runs):
        filepaths = []
        for run in runs:
            try:
                filepaths.append(get_input_project_file(int(run)))
            except FileNotFoundError:
                pass

        return filepaths

    def parse_entry_list(self, entry_list):
        entry_list = entry_list.replace(' ', '')
        split_text = entry_list.split(',')
        ranges = [entry for entry in split_text if (':' in entry) or (';' in entry)]
        entries = [entry for entry in split_text if (':' not in entry) and (';' not in entry)]

        runs = np.array([np.int16(entry) for entry in entries])
        for entry in ranges:
            split_entry = None
            if ':' in entry:
                split_entry = np.array(entry.split(':'), dtype=np.int16)
            elif ';' in entry:
                split_entry = np.array(entry.split(';'), dtype=np.int16)

            if split_entry is not None:
                split_entry = np.sort(split_entry)
                runs = np.append(runs, np.arange(split_entry[0],
                                                 split_entry[1]))

        return self.parse_file_path(runs)

    def load_combine_projects(self, project_files):
        if len(project_files) > 1:
            self._model.combine_project_files(project_files)
            return 1
        else:
            return 0
