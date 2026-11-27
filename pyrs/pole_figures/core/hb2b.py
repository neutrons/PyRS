"""
HB-2B (HIDRA, HFIR) instrument constants used by the pole-figure core.

They are the monochromator wavelengths, the 2theta range for pole figures and the table-scan process-variable (PV)
names.

Example:
    >>> two_theta_allowed(104.0), two_theta_allowed(130.0)
    (True, False)
"""

from pyrs.core.monosetting import MonoSetting

#: HB-2B monochromator setting -> nominal wavelength (Å), from PyRS's
#: ``pyrs.core.monosetting.MonoSetting`` (every name, so Si511 is listed with the same wavelength as Si333). Calibrated
#: values per run come from the data files.
HB2B_WAVELENGTHS: dict[str, float] = {name: float(m.value) for name, m in MonoSetting.__members__.items()}

# 2theta range allowed for pole-figure reflections on HB-2B. Outside it the peak is not used.
TWO_THETA_MIN, TWO_THETA_MAX = 50.0, 120.0


def two_theta_allowed(two_theta: float) -> bool:
    """
    True when a reflection's 2theta is inside the allowed HB-2B range (inclusive).

    Args:
        two_theta: 2theta of the reflection (deg).

    Returns:
        Whether ``TWO_THETA_MIN <= two_theta <= TWO_THETA_MAX``.
    """
    return TWO_THETA_MIN <= two_theta <= TWO_THETA_MAX


#: table-scan columns that set the run title and the phase label (EPICS process variables)
RUN_TITLE_PV = "HB2B:SMS:RunInfo:RunTitle"
PHASE_PV = "HB2B:CS:hklPhase"
