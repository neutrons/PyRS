from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction
import pytest


def test_peak_shape_enum():
    assert PeakShape.getShape('gaussian') == PeakShape.GAUSSIAN
    assert PeakShape.getShape('GAUSSIAN') == PeakShape.GAUSSIAN
    with pytest.raises(KeyError):
        PeakShape.getShape('non-existant-peak-shape')

    assert len(PeakShape.getShape('gaussian').native_parameters) == 3


def test_background_enum():
    assert BackgroundFunction.getFunction('linear') == BackgroundFunction.LINEAR
    assert BackgroundFunction.getFunction('LINEAR') == BackgroundFunction.LINEAR

    with pytest.raises(KeyError):
        BackgroundFunction.getFunction('non-existant-peak-shape')

    assert len(BackgroundFunction.getFunction('linear').native_parameters) == 2


if __name__ == '__main__':
    pytest.main([__file__])
