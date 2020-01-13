# Peak fitting engine
SupportedPeakProfiles = ['Gaussian', 'PseudoVoigt', 'Voigt']
SupportedBackgroundTypes = ['Flat', 'Linear', 'Quadratic']

__all__ = ['FitEngineFactory', 'SupportedPeakProfiles', 'SupportedBackgroundTypes']


class FitEngineFactory(object):
    """
    Peak fitting engine factory
    """
    @staticmethod
    def getInstance(name, hidraworkspace, out_of_plane_angle=None):
        """Get instance of Peak fitting engine
        """
        name = str(name).lower()

        # this must be here for now to stop circular imports
        from .mantid_fit_peak import MantidPeakFitEngine
        from .scipypeakfitengine import ScipyPeakFitEngine

        if name == 'mantid':
            return MantidPeakFitEngine(hidraworkspace, out_of_plane_angle)
        elif name == 'pyrs':
            return ScipyPeakFitEngine(hidraworkspace, out_of_plane_angle)
        else:
            raise RuntimeError('Cannot create a fit engine name="{}"'.format(name))
