import os
from typing import Iterable, List

from pyrs.utilities import get_input_project_file  # type: ignore


class CombineRunsCrtl:
    def __init__(self, _model):
        self._model = _model

    def parse_file_path(self, runs: Iterable[int]) -> List[str]:
        """Return each run's project file; a run that cannot be found raises, never silently drops."""
        filepaths = []
        for run in runs:
            # the lookup returns the folder holding the run's file, not the file (#931)
            folder = get_input_project_file(int(run))
            filepaths.append(os.path.join(folder, "HB2B_{}.h5".format(int(run))))

        return filepaths

    def parse_entry_list(self, entry_list: str) -> List[str]:
        """Parse run numbers such as ``"1327, 1328"`` and ranges ``"1327:1332"`` (or ``;``).

        The runs keep the order they were typed in, since that order numbers the combined file's
        sub-runs; a repeated run is combined once. A range includes both ends; empty entries (a
        trailing comma) are skipped. Run numbers are Python ints: ``np.int16`` overflowed above
        32767 (#931).

        Raises:
            ValueError: If an entry is not a whole number, or a range has more than two ends.
            RuntimeError: If a run's project file cannot be found.
        """
        runs: List[int] = []
        for entry in entry_list.replace(" ", "").split(","):
            if not entry:
                continue
            if ":" in entry or ";" in entry:
                ends = entry.replace(";", ":").split(":")
                if len(ends) != 2:
                    raise ValueError("Range {!r} needs exactly two run numbers".format(entry))
                first, last = sorted(int(end) for end in ends)
                runs.extend(range(first, last + 1))
            else:
                runs.append(int(entry))

        return self.parse_file_path(dict.fromkeys(runs))

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
