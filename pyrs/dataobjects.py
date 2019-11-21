# extentable version of dict https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
from collections import MutableMapping


class SampleLogs(MutableMapping):
    SUBRUN_KEY = 'sub-runs'  # TODO should be pyrs.utilities.rs_project_file.HidraConstants.SUB_RUNS

    def __init__(self, **kwargs):
        self._data = dict(kwargs)
        self.subruns = None

    def __delitem__(self, key):
        if key == self.SUBRUN_KEY:
            self.subruns = None
        else:
            del self._data[key]

    def __getitem__(self, key):
        if key == self.SUBRUN_KEY:
            return self.subruns
        else:
            return self._data[key]

    def __iter__(self):
        # does not include subruns
        return iter(self._data)

    def __len__(self):
        # does not include subruns
        return len(self._data)

    def __setitem__(self, key, value):
        if key == self.SUBRUN_KEY:
            self.subruns = value
        else:
            self._data[key] = value
