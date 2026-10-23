"""
Tests for pyrs/utilities/NXstress/_definitions.py
"""

import itertools
import re

import numpy as np
import pytest

from pyrs.utilities.NXstress._definitions import (
    CHUNK_SHAPE,
    FIELD_DTYPE,
    GROUP_NAME,
    group_naming_scheme,
    allowed_identifier,
    diffractogram_detector_mask,
    nxstress_diffractogram_keys,
    nxstress_mask_names,
    decode_identifier,
    is_ISO_8601,
    DEFAULT_TAG,
    MAX_IDENTIFIER_LENGTH,
    VALID_ITEM_NAME,
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

    # The identifier tests below iterate VALID_ITEM_NAME rather than restating it,
    # so that a change to the rule extends the guarantee automatically -- the idiom
    # used by test_GROUP_NAME_attributes above.
    #
    # They replace a test that pinned the previous behaviour ("replaces : with _ and
    # leaves other chars unchanged"). That conversion was many-to-one: 'HB2B:CS:X' and
    # 'HB2B_CS_X' both became 'HB2B_CS_X', so one log silently overwrote the other,
    # including the local_name attribute meant to preserve the original.

    @staticmethod
    def _every_byte_value_name() -> list[str]:
        """One name per byte value, exercising lead, interior and trail positions."""
        names = [f"a{chr(c)}b" for c in range(1, 256)]
        names += [f"{chr(c)}ab" for c in range(1, 256)]
        names += [f"ab{chr(c)}" for c in range(1, 256)]
        return names + ["2theta", "2thetaSetpoint", "ü", "日本", "_", "."]

    def test_allowed_identifier_output_matches_rule(self):
        """Every character, at every position, encodes into the NeXus alphabet."""
        # Arrange
        rule = re.compile(f"^{VALID_ITEM_NAME}$")

        # Act / Assert
        for name in self._every_byte_value_name():
            encoded = allowed_identifier(name)
            assert rule.match(encoded), f"{name!r} -> {encoded!r} violates VALID_ITEM_NAME"

    def test_allowed_identifier_round_trips_every_byte_value(self):
        """Reversibility over the same broad input set, not a hand-picked few.

        This asserts the property the identifier policy rests on across ~770
        names rather than the dozen in `test_allowed_identifier_round_trip`
        below, which stays because its inputs are *chosen* -- escape-shaped
        strings and real HB2B log names -- and would be diluted by generation.

        It replaces a `len(encoded) <= MAX_IDENTIFIER_LENGTH` assertion that sat
        in `test_allowed_identifier_output_matches_rule` and could never fail:
        these inputs are three characters, and the longest encoding they produce
        is 14 against a cap of 63. The cap is exercised by the two boundary tests
        below instead.
        """
        # Act / Assert
        for name in self._every_byte_value_name():
            assert decode_identifier(allowed_identifier(name)) == name

    @pytest.mark.parametrize("length", [MAX_IDENTIFIER_LENGTH - 1, MAX_IDENTIFIER_LENGTH])
    def test_allowed_identifier_at_the_length_cap_is_accepted(self, length: int):
        """A name *at* the cap is legal. The edge is what an off-by-one moves."""
        name = "a" * length

        assert allowed_identifier(name) == name

    def test_allowed_identifier_one_past_the_length_cap_raises(self):
        """And one character further raises, rather than truncating silently.

        Distinct from `test_allowed_identifier_too_long_raises` below, which
        passes a *short* name whose **encoding** overruns the cap. This pair
        pins the input-length edge; that one pins the encoded-length path.
        """
        with pytest.raises(ValueError, match=r".*exceeds the NeXus limit.*"):
            allowed_identifier("a" * (MAX_IDENTIFIER_LENGTH + 1))

    def test_allowed_identifier_round_trip(self):
        """Encoding is reversible, including for inputs shaped like escapes."""
        # Arrange: adversarial inputs plus real HB2B log-name shapes.
        names = [
            "a_3Ab",  # looks like a single-underscore escape
            "__",
            "_3A",
            "a__b",
            "a___b",
            "_DEFAULT_",
            "HB2B:Mot:sz_real",
            "HB2B:Mot:IS:Y:Center.RBV",
            "Scan Index",
            "2theta",
            "a$b",
            "a/b",
            "ü",
        ]

        # Act / Assert
        for name in names:
            assert decode_identifier(allowed_identifier(name)) == name

    def test_allowed_identifier_is_injective(self):
        """No two distinct names may share an encoding.

        Regression for the silent log loss described above: injectivity is what
        makes the collision impossible rather than merely detectable.
        """
        # Arrange
        seen: dict[str, str] = {}

        # Act / Assert
        for length in range(1, 5):
            for tup in itertools.product("A_:.3", repeat=length):
                name = "".join(tup)
                encoded = allowed_identifier(name)
                assert encoded not in seen or seen[encoded] == name, (
                    f"{seen.get(encoded)!r} and {name!r} both encode to {encoded!r}"
                )
                seen[encoded] = name

    def test_allowed_identifier_keeps_legal_names(self):
        """Names already legal under the rule pass through untouched."""
        # '.' in the interior and a leading digit are both legal NeXus -- a narrower
        # rule would escape them needlessly. See the sourced comment in _definitions.
        for name in ["simple_name", "name.with.dots", "2theta", "my_log_value", "_DEFAULT_"]:
            assert allowed_identifier(name) == name

    def test_allowed_identifier_empty_raises(self):
        """An empty log name is a bug upstream, not a name to encode."""
        with pytest.raises(ValueError, match=r".*empty string.*"):
            allowed_identifier("")

    def test_allowed_identifier_too_long_raises(self):
        """NeXus caps names at 63 characters, and encoding lengthens them."""
        # Arrange: 32 colons encode to 4 characters each, well past the cap.
        name = ":" * 32

        # Act / Assert
        with pytest.raises(ValueError, match=r".*exceeds the NeXus limit.*"):
            allowed_identifier(name)

    def test_decode_identifier_malformed_raises(self):
        """A truncated or non-hex escape is reported, not silently mangled."""
        with pytest.raises(ValueError, match=r".*Malformed escape.*"):
            decode_identifier("a__ZZb")

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


class TestDiffractogramKeysAreNotMaskNames:
    """Two namespaces, kept apart.

    A **detector mask** is a per-pixel array; `_DEFAULT_` is always one, because
    every reduction uses the default mask. A **diffractogram key** is a
    `_diff_data_set` key: a reduction identifier that *references* a detector mask
    and optionally intersects it with an eta region. Conflating them is what put
    composites like `eta_-5.0` into `masks/names` with no array behind them.
    """

    def test_mask_names_always_include_the_default(self):
        """Every reduction uses the default mask, so it is always in the namespace."""
        assert nxstress_mask_names([]) == {DEFAULT_TAG}
        assert nxstress_mask_names(["mask_a"]) == {DEFAULT_TAG, "mask_a"}
        assert nxstress_mask_names([None]) == {DEFAULT_TAG}

    def test_diffractogram_keys_do_not_invent_the_default(self):
        """A texture workspace reduces `eta_*` only; nothing used the default alone."""
        assert nxstress_diffractogram_keys([]) == set()
        assert nxstress_diffractogram_keys(["eta_-5.0", "eta_0.0"]) == {"eta_-5.0", "eta_0.0"}
        # `None` is the key for "the default mask alone", and maps to the tag.
        assert nxstress_diffractogram_keys([None, "mask_a"]) == {DEFAULT_TAG, "mask_a"}

    def test_a_key_resolves_to_the_detector_mask_it_references(self):
        known = {DEFAULT_TAG, "mask_a"}
        assert diffractogram_detector_mask(DEFAULT_TAG, known) == DEFAULT_TAG
        assert diffractogram_detector_mask("mask_a", known) == "mask_a"
        assert diffractogram_detector_mask("eta_-5.0", known) == DEFAULT_TAG
        assert diffractogram_detector_mask("mask_a_eta_-5.0", known) == "mask_a"

    def test_resolution_prefers_the_longest_matching_mask_name(self):
        """`mask_a_b_eta_0` belongs to `mask_a_b`, not to `mask_a`."""
        known = {DEFAULT_TAG, "mask_a", "mask_a_b"}
        assert diffractogram_detector_mask("mask_a_b_eta_0.0", known) == "mask_a_b"
        assert diffractogram_detector_mask("mask_a_eta_0.0", known) == "mask_a"

    def test_a_mask_named_like_a_composite_is_not_misparsed(self):
        """Resolution matches against known names rather than parsing the string."""
        known = {DEFAULT_TAG, "foo_eta_1.0"}
        assert diffractogram_detector_mask("foo_eta_1.0", known) == "foo_eta_1.0"

    def test_a_key_referencing_an_absent_mask_raises(self):
        with pytest.raises(RuntimeError, match=r"references no available detector mask"):
            diffractogram_detector_mask("mask_b_eta_0.0", {DEFAULT_TAG, "mask_a"})
