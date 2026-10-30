"""
tests/util/instrument_helpers.py

Helpers for ensuring a HidraWorkspace has instrument geometry set,
for use in tests and demo scripts that load Hidra project files which
may not contain an instrument section.
"""

from pyrs.core.instrument_geometry import DENEXDetectorGeometry
from pyrs.core.workspaces import HidraWorkspace

# Import the canonical HB2B engineering constants from the one place they
# are defined, so there is no duplication.
from pyrs.core.nexus_conversion import NUM_PIXEL_1D, PIXEL_SIZE, ARM_LENGTH


def default_hb2b_geometry() -> DENEXDetectorGeometry:
    """Return the standard HB2B DENEX detector geometry using nominal engineering values.

    These are the same hardcoded values used by ``NeXusConvertingApp`` when it
    initialises a fresh ``HidraWorkspace`` from a NeXus file.  The geometry is
    *not* calibrated (``calibrated=False``) and carries no detector shift, i.e.
    it represents the as-built instrument with no alignment corrections applied.
    """
    return DENEXDetectorGeometry(
        NUM_PIXEL_1D,
        NUM_PIXEL_1D,
        PIXEL_SIZE,
        PIXEL_SIZE,
        ARM_LENGTH,
        False,  # not calibrated — nominal engineering values, no shift applied
    )


def ensure_instrument_geometry(ws: HidraWorkspace) -> None:
    """Set the default HB2B geometry on *ws* if none is already present.

    Most Hidra project files do not include an instrument section, but
    ``NXstress`` requires one.  Calling this function after
    ``ws.load_hidra_project(...)`` guarantees the workspace has a geometry
    without overwriting one that was stored in the file.

    The geometry installed here is the nominal (uncalibrated) HB2B engineering
    geometry; no ``DENEXDetectorShift`` is applied, so
    ``ws._instrument_geometry_shift`` is left untouched.

    Parameters
    ----------
    ws:
        The ``HidraWorkspace`` to patch in-place.
    """
    if ws._instrument_setup is None:
        ws.set_instrument_geometry(default_hb2b_geometry())
