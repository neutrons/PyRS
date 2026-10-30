"""
tests/util/input_data_helpers.py

Helper for ensuring a HidraWorkspace has its raw detector counts (_raw_counts)
populated, for use in smoke-test scripts that load HiDRA project files which do
not retain the raw pixel data.

NOTE: this module imports from ``pyrs.core.nexus_conversion`` which depends on
Mantid.  It should only be used from smoke-test scripts or integration tests
that run in a Mantid-enabled environment, *not* from pure-Python unit tests.
"""

from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.file_util import get_nexus_file


def ensure_input_data(ws: HidraWorkspace, run_number: int) -> None:
    """Populate *ws._raw_counts* from the HFIR archive if not already loaded.

    Locates the raw NeXus event file for *run_number* via Mantid's
    ``FileFinder`` (with ``finddata`` as a fallback), converts the pixel
    event data with ``NeXusConvertingApp.convert()``, and copies the resulting
    per-sub-run detector counts into *ws* using ``ws.set_raw_counts()``.

    This is a no-op when *ws._raw_counts* is already non-empty (e.g. the
    project file was loaded with ``load_raw_counts=True``).

    The sample logs, reduced diffraction data, peak fits, and instrument
    geometry already present in *ws* are not touched.

    Parameters
    ----------
    ws:
        ``HidraWorkspace`` to populate in-place.
    run_number:
        HB2B run number (integer) whose raw NeXus file should be loaded.
        The IPTS directory is resolved automatically via Mantid's archive
        search, which looks under ``/HFIR/HB2B/IPTS-*/nexus/HB2B_{run}.nxs.h5``.
    """
    if ws._raw_counts:
        # Already populated -- nothing to do.
        return

    nexus_path = get_nexus_file(run_number)
    converter = NeXusConvertingApp(nexus_path)
    converted = converter.convert()

    for sub_run, counts in converted._raw_counts.items():
        ws.set_raw_counts(sub_run, counts)
