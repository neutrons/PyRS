import os

from pyrs.utilities import get_input_project_file  # type: ignore


class CombineRunsCrtl:
    def __init__(self, _model):
        self._model = _model

    def parse_file_path(self, runs):
        """Return each run's project file; a run that cannot be found raises, never silently drops."""
        filepaths = []
        for run in runs:
            # the lookup returns the folder holding the run's file, not the file (#931)
            folder = get_input_project_file(int(run))
            filepaths.append(os.path.join(folder, "HB2B_{}.h5".format(int(run))))

        return filepaths

    def parse_entry_list(self, entry_list):
        """Parse run numbers such as ``"1327, 1328"`` and ranges ``"1327:1332"`` (or ``;``).

        A range includes both ends. Run numbers are Python ints: ``np.int16`` overflowed above
        32767 (#931).

        Raises:
            ValueError: If an entry is not a whole number.
            RuntimeError: If a run's project file cannot be found.
        """
        entry_list = entry_list.replace(" ", "")
        split_text = entry_list.split(",")
        ranges = [entry for entry in split_text if (":" in entry) or (";" in entry)]
        entries = [entry for entry in split_text if (":" not in entry) and (";" not in entry)]

        runs = [int(entry) for entry in entries]
        for entry in ranges:
            ends = entry.replace(";", ":").split(":")
            if len(ends) != 2:
                raise ValueError("Range {!r} needs exactly two run numbers".format(entry))
            first, last = sorted(int(end) for end in ends)
            runs.extend(range(first, last + 1))

        return self.parse_file_path(runs)

    def load_combine_projects(self, project_files, accept_reduction_differences=False):
        if len(project_files) > 1:
            self._model.combine_project_files(project_files, accept_reduction_differences=accept_reduction_differences)
            return 1
        else:
            return 0

    def export_combined_projectfile(self, export_file):
        # a cancelled save dialog returns "" (not None)
        if export_file:
            self._model.export_project_files(export_file)
