"""Tests for which reduced-data layouts `TextureFittingCrtl.fit_peaks` accepts.

Unit tier: the controller module imports headless, and the layout is checked
from the model's `ws.reduction_masks` before anything is fitted, so a plain
namespace stands in for the model. Fitting the accepted layouts is covered by
`tests/integration/test_texture_fit_layouts.py`.
"""

from types import SimpleNamespace

import pytest

from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl


@pytest.mark.parametrize(
    "masks",
    [
        [None, "eta_0.0"],
        [None, "some_other_detector_mask"],
        ["some_other_detector_mask_eta_-5.0"],
        ["eta_0.0", "some_other_detector_mask"],
        [],
    ],
    ids=["default-and-eta", "in-plane-mask", "eta-of-in-plane-mask", "eta-and-in-plane-mask", "no-data"],
)
def test_fit_peaks_unsupported_layout_raises(masks: list) -> None:
    """Refused, naming the layout, until what fitting it should mean is settled."""
    # Arrange
    crtl = TextureFittingCrtl(SimpleNamespace(ws=SimpleNamespace(reduction_masks=masks)))

    # Act / Assert
    with pytest.raises(NotImplementedError, match="does not yet support"):
        crtl.fit_peaks(fit_range_table=None, peak_function_name="Gaussian", background_function_name="Linear")
