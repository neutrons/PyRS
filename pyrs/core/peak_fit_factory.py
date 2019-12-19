# Peak fitting engine
from pyrs.peaks.mantid_fit_peak import MantidPeakFitEngine
from pyrs.utilities import checkdatatypes


SupportedPeakProfiles = ['Gaussian', 'PseudoVoigt', 'Voigt']
SupportedBackgroundTypes = ['Flat', 'Linear', 'Quadratic']


class PeakFitEngineFactory(object):
    """
    Peak fitting engine factory
    """
    @staticmethod
    def getInstance(engine_name):
        """ Get instance of Peak fitting engine
        :param engine_name:
        :return:
        """
        checkdatatypes.check_string_variable('Peak fitting engine', engine_name, ['Mantid', 'PyRS'])

        if engine_name == 'Mantid':
            engine_class = MantidPeakFitEngine
        else:
            raise RuntimeError('Implement general scipy peak fitting engine')

        return engine_class
