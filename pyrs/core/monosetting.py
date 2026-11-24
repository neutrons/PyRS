from enum import Enum

__all__ = ["MonoSetting"]


class MonoSetting(Enum):
    """The monochromator settings, each with its nominal wavelength.

    Si333 and Si511 are different monochromator positions (different ``mrot``) with the same wavelength. Each
    member's enum value is therefore ``(name, wavelength)``: were it the wavelength alone, ``Enum`` would make Si511
    an alias of Si333 and a Si511 run would be reported as Si333. ``value`` and ``float()`` give the
    wavelength.
    """

    Si333 = ("Si333", 1.452)
    Si511 = ("Si511", 1.452)  # same wavelength as Si333, a different monochromator position
    Si422 = ("Si422", 1.540)
    Si331 = ("Si331", 1.731)
    Si400 = ("Si400", 1.886)
    Si311 = ("Si311", 2.275)
    Si220 = ("Si220", 2.667)

    @property
    def value(self):  # type: ignore[override]
        """The nominal wavelength"""
        return self._value_[1]

    def __str__(self):
        """The label of the peak"""
        return self.name

    def __float__(self):
        """The nominal wavelength"""
        return self._value_[1]

    @staticmethod
    def getFromIndex(index):
        """The ``MonoSetting`` log in the Nexus file holds an index into this ordered array"""
        index = int(index)
        settings = [
            MonoSetting.Si333,
            MonoSetting.Si511,
            MonoSetting.Si422,
            MonoSetting.Si331,
            MonoSetting.Si400,
            MonoSetting.Si311,
            MonoSetting.Si220,
        ]
        if index < 0 or index >= len(settings):
            raise IndexError("Index must be between 0 and {} (supplied index={})".format(len(settings) - 1, index))
        return settings[index]

    @staticmethod
    def getFromRotation(mrot):
        """The ``mrot`` (monochromator rotation) log in the NeXus file can be converted into a specific wavelength"""
        mrot = mrot
        if -41.0 < mrot < -38.0:
            return MonoSetting.Si333
        elif -4.0 < mrot < 1.0:
            return MonoSetting.Si511
        elif -25.0 < mrot < -14.0:
            return MonoSetting.Si422
        elif -175.0 < mrot < -166.0:
            return MonoSetting.Si331
        elif 12.0 < mrot < 18.0:
            return MonoSetting.Si400
        elif -13.5 < mrot < -6.0:
            return MonoSetting.Si311
        elif -200.0 < mrot < -177.0:
            return MonoSetting.Si220
        else:
            raise ValueError("Unable to determine monosetting from the monochromator rotation angle {}".format(mrot))
