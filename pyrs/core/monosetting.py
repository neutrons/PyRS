from enum import Enum

__all__ = ['MonoSetting']


class MonoSetting(Enum):
    '''Hold the default values for the nominal wavelengths of the monochromator'''
    Si333 = 1.452
    Si511 = 1.452  # same as Si333
    Si422 = 1.540
    Si331 = 1.731
    Si400 = 1.886
    Si311 = 2.275
    Si220 = 2.667

    def __str__(self):
        '''The label of the peak'''
        return self.name

    def __float__(self):
        '''The nominal wavelength'''
        return self._value_

    @staticmethod
    def getFromIndex(index):
        '''The ``MonoSetting`` log in the Nexus file holds an index into this ordered array'''
        index = int(index)
        settings = [MonoSetting.Si333, MonoSetting.Si511, MonoSetting.Si422, MonoSetting.Si331, MonoSetting.Si400,
                    MonoSetting.Si311, MonoSetting.Si220]
        if index < 0 or index >= len(settings):
            raise IndexError('Index must be between 0 and {} (supplied index={})'.format(len(settings) - 1, index))
        return settings[index]

    @staticmethod
    def getFromRotation(mrot):
        '''The ``mrot`` (monochromator rotation) log in the NeXus file can be converted into a specific wavelength'''
        mrot = mrot
        if -41.0 < mrot < -38.0:
            return MonoSetting.Si333
        elif -1.0 < mrot < 1.0:
            return MonoSetting.Si511
        elif -25.0 < mrot < -14.0:
            return MonoSetting.Si422
        elif -170.0 < mrot < -166.0:
            return MonoSetting.Si331
        elif 14.0 < mrot < 18.0:
            return MonoSetting.Si400
        elif -11.5 < mrot < -6.0:
            return MonoSetting.Si311
        elif -200.0 < mrot < -175.0:
            return MonoSetting.Si220
        else:
            raise ValueError('Unable to determine monosetting from the monochromator rotation angle {}'.format(mrot))
