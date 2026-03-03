"""
Tests for pyrs/utilities/NXstress/_definitions.py
"""

import numpy as np
import pytest

from pyrs.utilities.NXstress._definitions import (
    CHUNK_SHAPE,
    FIELD_DTYPE,
    GROUP_NAME,
    group_naming_scheme,
    allowed_identifier,
    is_ISO_8601,
    DEFAULT_TAG,
)


class TestDefinitions:
    """Test suite for _definitions.py utility functions and enums"""

    def test_CHUNK_SHAPE(self):
        """Verify CHUNK_SHAPE returns correct tuples for ranks 1-3"""
        assert CHUNK_SHAPE(1) == (100,)
        assert CHUNK_SHAPE(2) == (1, 100)
        assert CHUNK_SHAPE(3) == (1, 1, 100)

    def test_FIELD_DTYPE_call(self):
        """Verify calling a FIELD_DTYPE enum member returns expected NumPy dtype"""
        # Test that calling enum members constructs values of the expected type
        float_val = FIELD_DTYPE.FLOAT_DATA(3.14)
        assert isinstance(float_val, np.float32)
        assert float_val == np.float32(3.14)

        int_val = FIELD_DTYPE.INT_DATA(42)
        assert isinstance(int_val, np.int32)
        assert int_val == np.int32(42)

    def test_FIELD_DTYPE_is_instance(self):
        """Verify FIELD_DTYPE.is_instance correctly identifies instances"""
        assert FIELD_DTYPE.FLOAT_DATA.is_instance(np.float32(1.0)) is True
        assert FIELD_DTYPE.INT_DATA.is_instance(np.float32(1.0)) is False

        assert FIELD_DTYPE.INT_DATA.is_instance(np.int32(42)) is True
        assert FIELD_DTYPE.FLOAT_DATA.is_instance(np.int32(42)) is False

    def test_FIELD_DTYPE_is_subclass(self):
        """Verify FIELD_DTYPE.is_subclass correctly identifies subclasses"""
        assert FIELD_DTYPE.FLOAT_DATA.is_subclass(np.float32) is True
        assert FIELD_DTYPE.FLOAT_DATA.is_subclass(np.int32) is False

        assert FIELD_DTYPE.INT_DATA.is_subclass(np.int32) is True
        assert FIELD_DTYPE.INT_DATA.is_subclass(np.float32) is False

    def test_FIELD_DTYPE_str(self):
        """Verify str(FIELD_DTYPE) returns the underlying type's __name__"""
        assert str(FIELD_DTYPE.FLOAT_DATA) == "float32"
        assert str(FIELD_DTYPE.INT_DATA) == "int32"
        assert str(FIELD_DTYPE.FLOAT_CONSTANT) == "float64"

    def test_GROUP_NAME_attributes(self):
        """Verify GROUP_NAME enum members have allowMultiple and nxClass attributes"""
        # Check that all enum members have the required attributes
        for group in GROUP_NAME:
            assert hasattr(group, "allowMultiple")
            assert hasattr(group, "nxClass")
            assert isinstance(group.allowMultiple, bool)

        # Verify specific known members
        assert GROUP_NAME.ENTRY.allowMultiple is True
        assert GROUP_NAME.DETECTOR.allowMultiple is True
        assert GROUP_NAME.FIT.allowMultiple is True

        assert GROUP_NAME.INSTRUMENT.allowMultiple is False
        assert GROUP_NAME.SAMPLE_DESCRIPTION.allowMultiple is False

    def test_group_naming_scheme_int_first(self):
        """Verify group_naming_scheme with int suffix=1 omits suffix"""
        assert group_naming_scheme("entry", 1) == "entry"

    def test_group_naming_scheme_int_second(self):
        """Verify group_naming_scheme with int suffix>1 adds suffix"""
        assert group_naming_scheme("entry", 2) == "entry_2"
        assert group_naming_scheme("entry", 3) == "entry_3"

    def test_group_naming_scheme_str_default(self):
        """Verify group_naming_scheme with DEFAULT_TAG omits suffix"""
        assert group_naming_scheme("DIFFRACTOGRAM", DEFAULT_TAG) == "DIFFRACTOGRAM"

    def test_group_naming_scheme_str_nondefault(self):
        """Verify group_naming_scheme with non-default string adds suffix"""
        assert group_naming_scheme("DIFFRACTOGRAM", "custom_mask") == "DIFFRACTOGRAM_custom_mask"
        assert group_naming_scheme("FIT", "mask_2") == "FIT_mask_2"

    def test_group_naming_scheme_invalid_suffix(self):
        """Verify group_naming_scheme raises RuntimeError for invalid suffix type"""
        with pytest.raises(RuntimeError, match=r".*not implemented for suffix.*"):
            group_naming_scheme("entry", 3.14)

    def test_allowed_identifier(self):
        """Verify allowed_identifier replaces : with _ and leaves other chars unchanged"""
        assert allowed_identifier("HB2B:CS:Wavelength") == "HB2B_CS_Wavelength"
        assert allowed_identifier("simple_name") == "simple_name"
        assert allowed_identifier("name.with.dots") == "name.with.dots"
        assert allowed_identifier("A:B:C:D") == "A_B_C_D"

    def test_is_ISO_8601_valid(self):
        """Verify is_ISO_8601 returns True for valid ISO 8601 strings"""
        assert is_ISO_8601("2024-01-15T10:30:00") is True
        assert is_ISO_8601("2024-12-31T23:59:59") is True
        assert is_ISO_8601("2024-01-01T00:00:00") is True

    def test_is_ISO_8601_invalid(self):
        """Verify is_ISO_8601 returns False for invalid date strings"""
        assert is_ISO_8601("not-a-date") is False
        assert is_ISO_8601("2024/01/15 10:30:00") is False
        assert is_ISO_8601("invalid") is False
