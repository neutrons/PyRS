from pyrs.core import MonoSetting  # type: ignore
import pytest


def test_monosetting_from_index():
    for index, mono_exp in enumerate([MonoSetting.Si333, MonoSetting.Si511, MonoSetting.Si422, MonoSetting.Si331,
                                      MonoSetting.Si400, MonoSetting.Si311, MonoSetting.Si220]):
        mono_obs = MonoSetting.getFromIndex(index)
        assert mono_obs == mono_exp

    with pytest.raises(IndexError):
        MonoSetting.getFromIndex(-1)

    with pytest.raises(IndexError):
        MonoSetting.getFromIndex(8)


def test_monosetting_from_rotation():
    monosetting = MonoSetting.getFromRotation(-182.0)
    assert monosetting == MonoSetting.Si220

    with pytest.raises(ValueError):
        monosetting.getFromRotation(178)


def test_monosetting_conversions():
    assert str(MonoSetting.Si422) == 'Si422'
    assert float(MonoSetting.Si422) == 1.540


if __name__ == '__main__':
    pytest.main([__file__])
