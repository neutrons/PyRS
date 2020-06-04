# Peak fitting engine
SupportedPeakProfiles = ['Gaussian', 'PseudoVoigt', 'Voigt']
SupportedBackgroundTypes = ['Flat', 'Linear', 'Quadratic']

__all__ = ['FitEngineFactory', 'SupportedPeakProfiles', 'SupportedBackgroundTypes']


class FitEngineFactory:
    """
    Peak fitting engine factory
    """
    @staticmethod
    def getInstance(hidraworkspace, peak_function_name, background_function_name,
                    wavelength, out_of_plane_angle=None, engine_name='mantid'):
        """Get instance of Peak fitting engine
        """
        engine_name = str(engine_name).lower()

        # this must be here for now to stop circular imports
        from .mantid_fit_peak import MantidPeakFitEngine

        if engine_name == 'mantid':
            return MantidPeakFitEngine(hidraworkspace, peak_function_name, background_function_name,
                                       wavelength=wavelength, out_of_plane_angle=out_of_plane_angle)
        else:
            raise RuntimeError('Cannot create a fit engine name="{}"'.format(engine_name))
