import numpy as np
import pytest

from pyrs.utilities.convertdatatypes import is_text_array, to_float, to_text, to_text_array


def test_convert_to_float():
    GOOD = "good"
    BAD = "bad"

    # simple checks
    assert to_float(GOOD, 42.0) == 42.0
    assert to_float(GOOD, 42) == 42.0
    assert to_float(GOOD, "42") == 42.0
    assert to_float(GOOD, 42.0, 41, 43) == 42.0
    assert to_float(BAD, 42.0, 42, 43, min_inclusive=True, max_inclusive=False) == 42.0
    assert to_float(BAD, 42.0, 41, 42, min_inclusive=False, max_inclusive=True) == 42.0

    # check invalid range
    with pytest.raises(ValueError) as err:
        to_float(BAD, 42.0, 43, 41)
        assert BAD in str(err.value)

    # check value outside of range
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42.0, 42, 43, min_inclusive=False, max_inclusive=True)
        assert BAD in str(err.value)
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42.0, 41, 42, min_inclusive=True, max_inclusive=False)
        assert BAD in str(err.value)

    # check with non-convertable values
    for value in [None, "strings cannot be converted", (1, 2)]:
        with pytest.raises(TypeError) as err:
            assert not to_float(BAD, value)
            assert BAD in str(err.value)


def test_convert_to_integer():
    GOOD = "good"
    BAD = "bad"

    # simple checks
    assert to_float(GOOD, 42.0) == 42
    assert to_float(GOOD, 42) == 42
    assert to_float(GOOD, "42") == 42
    assert to_float(GOOD, 42.0, 41, 43) == 42
    assert to_float(BAD, 42.0, 42, 43, min_inclusive=True, max_inclusive=False) == 42
    assert to_float(BAD, 42.0, 41, 42, min_inclusive=False, max_inclusive=True) == 42

    # check invalid range
    with pytest.raises(ValueError) as err:
        to_float(BAD, 42.0, 43, 41)
        assert BAD in str(err.value)

    # check value outside of range
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42.0, 42, 43, min_inclusive=False, max_inclusive=True)
        assert BAD in str(err.value)
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42.0, 41, 42, min_inclusive=True, max_inclusive=False)
        assert BAD in str(err.value)

    # check with non-convertable values
    for value in [None, "strings cannot be converted", (1, 2)]:
        with pytest.raises(TypeError) as err:
            assert not to_float(BAD, value)
            assert BAD in str(err.value)


if __name__ == "__main__":
    pytest.main([__file__])


class TestToText:
    """`as_text` is the one place NXstress copes with a string's two spellings.

    `SampleLogs.__setitem__` normalizes string log values to text, so an in-memory
    value should already be `str` -- but a value can be read straight off an
    `NXfield` or an h5py dataset, and the cost of assuming one spelling is a
    crash on the other. That is what `NXstress._entryTimes`' unconditional
    `t.decode("utf-8")` did: a `str` log raised `AttributeError`, which the
    surrounding `except ValueError` (there to substitute `NO_LOG`) could not
    catch.
    """

    @pytest.mark.parametrize(
        "value",
        [b"11", "11", np.bytes_(b"11"), np.str_("11"), np.array([b"11"])[0], np.array(["11"])[0]],
        ids=["bytes", "str", "np.bytes_", "np.str_", "from-S-array", "from-U-array"],
    )
    def test_every_spelling_answers_the_same_str(self, value):
        # Act
        result = to_text(value)

        # Assert
        assert result == "11"
        assert type(result) is str

    def test_a_non_ascii_value_round_trips(self):
        assert to_text("Fe-\u03b1".encode("utf-8")) == "Fe-\u03b1"


class TestToTextArray:
    """The dtype that comes off HDF5 is `object`, not `|S` -- and that is the catch.

    A conversion written against `|S` alone looks right and passes a test built
    with `np.array([b"a", b"b"])`, while leaving untouched every value that came
    off a real file: h5py and `nexusformat` both hand back an **object** array of
    Python `bytes` for the variable-length UTF-8 dtype, whose `dtype.kind` is
    `"O"`. That is exactly how the first version of this normalization shipped
    green across all three test tiers while doing nothing for `start_time`,
    `end_time`, `Filename` or any discriminator column.
    """

    @staticmethod
    def _object_array(values):
        """What HDF5 actually yields -- not what `np.array([...])` produces."""
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    def test_fixed_width_bytes_become_text(self):
        assert list(to_text_array(np.array([b"11", b"22"]))) == ["11", "22"]

    def test_object_array_of_bytes_becomes_text(self):
        """The HDF5 case. `dtype.kind` is 'O', so an `|S`-only check misses it."""
        # Arrange
        values = self._object_array([b"11", b"22"])
        assert values.dtype.kind == "O"

        # Act
        result = to_text_array(values)

        # Assert
        assert result.dtype.kind == "U"
        assert list(result) == ["11", "22"]

    def test_object_array_of_str_becomes_text(self):
        result = to_text_array(self._object_array(["11", "22"]))
        assert result.dtype.kind == "U"

    def test_text_is_returned_unchanged(self):
        values = np.array(["11", "22"])
        assert to_text_array(values) is values

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.5, 2.5]),
            np.array([1, 2]),
            np.array([True, False]),
        ],
        ids=["float", "int", "bool"],
    )
    def test_a_numeric_array_raises(self, values):
        """Converting numbers to text is a usage error, not a pass-through.

        Returning the array unchanged would make a misdirected call look like a
        working one, and the symptom would surface later as a `bytes` value in
        whichever consumer did not expect one -- the exact failure this
        conversion exists to end. A caller that may hold either asks
        `is_text_array` first.
        """
        with pytest.raises(TypeError, match=r"expects an array of strings"):
            to_text_array(values)

    def test_a_heterogeneous_object_array_raises(self):
        """Convert only when every element is text; never guess at a mixed array."""
        with pytest.raises(TypeError, match=r"expects an array of strings"):
            to_text_array(self._object_array([b"11", 22]))

    def test_an_empty_object_array_becomes_an_empty_text_array(self):
        """An empty string column is legitimate, on disk and in memory.

        An empty HDF5 dataset keeps its own dtype on read -- `float64` stays
        `float64`, `|S4` stays `|S4` -- and only a **variable-length string**
        comes back as `object`. So an empty object array is unambiguously an
        empty string column, and it must not come back as `np.array([])`, which
        is `float64`: that is precisely what the hand-rolled conversion this
        replaced produced for `peaks/phase_name` in an entry written with no
        peak collections.
        """
        # Act
        result = to_text_array(np.empty(0, dtype=object))

        # Assert
        assert result.dtype.kind == "U"
        assert len(result) == 0

    def test_an_empty_numeric_array_still_raises(self):
        """Empty does not make `float64` ambiguous -- its dtype still says numbers."""
        with pytest.raises(TypeError, match=r"expects an array of strings"):
            to_text_array(np.array([], dtype=float))

    def test_an_empty_text_array_keeps_its_shape(self):
        for values in (np.empty(0, dtype="S4"), np.empty(0, dtype="<U4"), np.empty(0, dtype=object)):
            assert to_text_array(values).shape == (0,)

    def test_a_non_array_raises(self):
        with pytest.raises(TypeError, match=r"got list"):
            to_text_array([b"11"])

    def test_the_error_names_what_arrived(self):
        """A usage error has to say what was passed, or it cannot be acted on."""
        with pytest.raises(TypeError, match=r"dtype dtype\('float64'\)"):
            to_text_array(np.array([1.5]))


class TestIsTextArray:
    """The predicate callers use to decide whether `to_text_array` applies."""

    @staticmethod
    def _object_array(values):
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    @pytest.mark.parametrize(
        "values",
        [np.array(["11"]), np.array([b"11"]), np.empty(0, dtype="<U1"), np.empty(0, dtype="S1")],
        ids=["U", "S", "empty-U", "empty-S"],
    )
    def test_text_dtypes_are_text(self, values):
        assert is_text_array(values)

    def test_an_object_array_of_text_is_text(self):
        assert is_text_array(self._object_array([b"11", "22"]))

    @pytest.mark.parametrize(
        "values",
        [np.array([1.5]), np.array([1]), [b"11"], b"11", None],
        ids=["float", "int", "list", "bytes", "none"],
    )
    def test_everything_else_is_not(self, values):
        assert not is_text_array(values)

    def test_a_heterogeneous_object_array_is_not_text(self):
        assert not is_text_array(self._object_array([b"11", 22]))

    def test_an_empty_object_array_is_text(self):
        """Nothing else reads back as an empty object array; see `TestToTextArray`."""
        assert is_text_array(np.empty(0, dtype=object))

    def test_an_empty_numeric_array_is_not_text(self):
        assert not is_text_array(np.array([], dtype=float))

    def test_it_agrees_with_what_to_text_array_accepts(self):
        """The predicate's whole purpose: a True must mean the conversion works."""
        # Arrange
        candidates = [
            np.array(["11"]),
            np.array([b"11"]),
            self._object_array([b"11"]),
            self._object_array([b"11", 22]),
            np.empty(0, dtype=object),
            np.array([], dtype=float),
            np.array([1.5]),
            [b"11"],
        ]

        # Act / Assert
        for values in candidates:
            if is_text_array(values):
                assert to_text_array(values).dtype.kind == "U"
            else:
                with pytest.raises(TypeError):
                    to_text_array(values)
