from pyrs.core import MonoSetting  # type: ignore
import pytest


def test_monosetting_from_index():
    for index, mono_exp in enumerate(
        [
            MonoSetting.Si333,
            MonoSetting.Si511,
            MonoSetting.Si422,
            MonoSetting.Si331,
            MonoSetting.Si400,
            MonoSetting.Si311,
            MonoSetting.Si220,
        ]
    ):
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
    assert str(MonoSetting.Si422) == "Si422"
    assert float(MonoSetting.Si422) == 1.540


def test_si333_and_si511_are_distinct_members():
    """Same wavelength, different monochromator position: not aliases."""
    assert MonoSetting.Si511 is not MonoSetting.Si333
    assert len(MonoSetting) == 7
    assert MonoSetting.Si511.name == "Si511"
    assert float(MonoSetting.Si511) == float(MonoSetting.Si333) == 1.452


def test_value_is_the_wavelength():
    """peak_fitting_model reads ``.value`` as the wavelength."""
    assert [m.value for m in MonoSetting] == [float(m) for m in MonoSetting]
    assert MonoSetting.Si422.value == 1.540


@pytest.mark.parametrize("mrot, mono", [(-39.5, MonoSetting.Si333), (-1.5, MonoSetting.Si511)])
def test_monosetting_from_rotation_tells_si333_si511_apart(mrot, mono):
    assert MonoSetting.getFromRotation(mrot) is mono


def test_monosetting_from_index_tells_si333_si511_apart():
    assert MonoSetting.getFromIndex(0) is MonoSetting.Si333
    assert MonoSetting.getFromIndex(1) is MonoSetting.Si511


if __name__ == "__main__":
    pytest.main([__file__])
