# Peak fitting engine
from pyrs.utilities import checkdatatypes


SupportedPeakProfiles = ['Gaussian', 'PseudoVoigt', 'Voigt']
SupportedBackgroundTypes = ['Flat', 'Linear', 'Quadratic']

__all__ = ['PeakFitEngineFactory', 'SupportedPeakProfiles', 'SupportedBackgroundTypes']


class PeakFitEngineFactory(object):
    """
    Peak fitting engine factory
    """
    @staticmethod
    def getInstance(name):
        """ Get instance of Peak fitting engine
        """
        checkdatatypes.check_string_variable('Peak fitting engine', name, ['Mantid', 'PyRS'])

        # this must be here for now to stop circular imports
        from .mantid_fit_peak import MantidPeakFitEngine

        if name == 'Mantid':
            return MantidPeakFitEngine
        else:
            raise RuntimeError('Implement general scipy peak fitting engine')
