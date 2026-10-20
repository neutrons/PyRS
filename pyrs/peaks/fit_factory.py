# Peak fitting engine
SupportedPeakProfiles = ["Gaussian", "PseudoVoigt"]
SupportedBackgroundTypes = ["Flat", "Linear", "Quadratic"]

__all__ = ["FitEngineFactory", "SupportedPeakProfiles", "SupportedBackgroundTypes"]


class FitEngineFactory:
    """
    Peak fitting engine factory
    """

    @staticmethod
    def getInstance(
        hidraworkspace,
        peak_function_name,
        background_function_name,
        wavelength,
        out_of_plane_angle=None,
        engine_name="mantid",
        high_background: bool = True,
    ):
        """Get instance of Peak fitting engine

        ``high_background`` is forwarded to Mantid FitPeaks' ``HighBackground`` option; see
        :class:`~pyrs.peaks.mantid_fit_peak.MantidPeakFitEngine`.
        """
        engine_name = str(engine_name).lower()

        # this must be here for now to stop circular imports
        from .mantid_fit_peak import MantidPeakFitEngine

        if engine_name == "mantid":
            return MantidPeakFitEngine(
                hidraworkspace,
                peak_function_name,
                background_function_name,
                wavelength=wavelength,
                out_of_plane_angle=out_of_plane_angle,
                high_background=high_background,
            )
        else:
            raise RuntimeError('Cannot create a fit engine name="{}"'.format(engine_name))
