"""
tests/util/mask_helpers.py

Helpers for adding masks to a HidraWorkspace in tests.
"""

import numpy as np

from pyrs.core.workspaces import HidraWorkspace


def add_named_detector_mask(ws: HidraWorkspace, mask_name: str) -> np.ndarray:
    """Add a named (non-default) all-pass detector mask to a workspace and return it.

    The mask array is shaped to match the workspace's instrument detector size.
    All pixels are set to 1 (unmasked).

    Parameters
    ----------
    ws : HidraWorkspace
        The workspace to add the mask to.
    mask_name : str
        A non-empty, non-default name for the mask.

    Returns
    -------
    np.ndarray
        The mask array that was added.
    """
    setup = ws.get_instrument_setup()
    nrows, ncols = setup.detector_size
    mask_array = np.ones(nrows * ncols, dtype=np.int64)
    ws.set_detector_mask(mask_array, False, mask_name)
    return mask_array
