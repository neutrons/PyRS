# This is rs_scan_io.DiffractionFile's 2.0 version
from enum import Enum

__all__ = ['HidraProjectFileMode']


class HidraProjectFileMode(Enum):
    '''
    Enumeration for file access mode

    These values match the strings of :py:obj:`h5py.File`
    '''
    READONLY = 'r'   # read-only
    READWRITE = 'a'  # read and write
    OVERWRITE = 'w'  # new file

    def __str__(self):
        return self.value

    @staticmethod
    def getMode(mode):
        '''Private function to convert anything into :py:obj:`HidraProjectFileMode`'''
        if isinstance(mode, HidraProjectFileMode):
            return mode  # already a member of the enum
        else:
            # all other checks are for strings
            # the first checks are against nominal values
            # the last check is against the enum names
            mode = str(mode).lower()
            if mode in ['a', 'rw']:
                return HidraProjectFileMode.READWRITE
            elif mode == 'r':
                return HidraProjectFileMode.READONLY
            elif mode == 'w':
                return HidraProjectFileMode.OVERWRITE
            else:
                return HidraProjectFileMode[mode.upper()]
