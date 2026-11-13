"""
Tests for `pyrs/utilities/restorable_property.py`.

The module had no test file at all. Coverage measured it at 87%, and the three
uncovered lines were its three `raise` guardrails -- so every way it refuses a
misuse was unverified, which is the half that fails *silently* when it breaks.
An independent mutation pass confirmed it: replacing `restore`'s type check
with `if False` left the entire 675-test suite green.

These tests cover the guardrails and the two properties that could previously
lose a value without raising: a backing attribute that does not exist, and the
`property` methods that rebuild a property and could drop the mark.
"""

import pytest

from pyrs.utilities.restorable_property import is_restorable, restore, restores


class Thing:
    """A class exercising both spellings: derived-looking and clearly distinct."""

    def __init__(self) -> None:
        self._colour = "blue"
        self._file_path = "a.h5"

    @property
    @restores("_colour")
    def colour(self) -> str:
        return self._colour

    @property
    @restores("_file_path")
    def source_file(self) -> str:
        """Backing attribute deliberately unlike the property name."""
        return self._file_path

    @property
    def plain(self) -> str:
        """An ordinary read-only property, for the negative cases."""
        return self._colour


class TestReadOnlyIsPreserved:
    """The module exists to keep a value read-only while still restorable."""

    def test_assignment_still_raises(self):
        with pytest.raises(AttributeError):
            Thing().colour = "red"  # type: ignore[misc]

    def test_the_property_is_an_ordinary_property(self):
        """Load-bearing, and the reason the mark is on the getter.

        `mypy` special-cases the literal `@property` decorator syntactically. A
        `property` **subclass** used as a decorator is not recognised, so the
        attribute degrades to `Any` and read-only assignment stops being an
        error -- measured, on the subclass implementation this replaced. Anything
        that makes these no longer plain `property` objects takes the static
        guarantee away again, which no runtime test would notice.
        """
        assert type(Thing.colour) is property
        assert type(Thing.source_file) is property


class TestRestore:
    def test_restores_the_declared_attribute(self):
        t = Thing()

        restore(t, "source_file", "b.h5")

        assert t.source_file == "b.h5"
        assert t._file_path == "b.h5"

    def test_a_non_restorable_property_raises(self):
        with pytest.raises(TypeError, match="is not a restorable property"):
            restore(Thing(), "plain", "red")

    def test_an_unknown_name_raises(self):
        with pytest.raises(TypeError, match="is not a restorable property"):
            restore(Thing(), "no_such_name", "red")

    def test_a_class_raises_and_says_why(self):
        """`is_restorable` accepts a class, so this asymmetry needs naming.

        The module docstring pairs the two as one idiom, and
        `_discriminator._validate_round_trippable` really does call
        `is_restorable(HidraWorkspace, name)` with a class -- so the mistake is
        available. Previously this raised "is not a restorable property", which
        is true but blames the wrong thing.
        """
        assert is_restorable(Thing, "colour")

        with pytest.raises(TypeError, match="needs an instance, not the class"):
            restore(Thing, "colour", "red")

    def test_a_misspelled_backing_attribute_raises(self):
        """Silent value loss, and exactly what this module exists to prevent.

        `setattr` on an ordinary object always succeeds, so without this guard a
        typo in `@restores(...)` creates a second attribute that the property
        never reads: no error, and the restored value is gone.
        """

        class Typo:
            def __init__(self) -> None:
                self._colour = "blue"

            @property
            @restores("_color")  # US spelling; the attribute is `_colour`
            def colour(self) -> str:
                return self._colour

        t = Typo()

        with pytest.raises(AttributeError, match="has no attribute '_color'"):
            restore(t, "colour", "red")

        assert not hasattr(t, "_color"), "the bad attribute must not be created"
        assert t.colour == "blue"


class TestIsRestorable:
    def test_instance_and_class_agree(self):
        t = Thing()

        assert is_restorable(t, "colour") is True
        assert is_restorable(Thing, "colour") is True

    def test_plain_property_and_unknown_name_are_false(self):
        assert is_restorable(Thing, "plain") is False
        assert is_restorable(Thing, "no_such_name") is False

    def test_a_subclass_inherits_the_mark(self):
        class Sub(Thing):
            pass

        assert is_restorable(Sub, "colour") is True


class TestTheMarkSurvivesPropertyRebuilding:
    """`property.getter`/`setter`/`deleter` each rebuild the property object.

    The implementation this replaced was a `property` subclass, which had to
    re-implement all three to carry its state across -- and `deleter` was
    missed, so `restore()` silently wrote to a re-derived attribute name. The
    mark now lives on the getter function, which every one of these carries
    across unchanged, so the hazard is gone by construction rather than by
    three overrides being remembered. Asserted because "by construction" is a
    claim about CPython's `property_copy`, not a law.
    """

    @pytest.mark.parametrize("method", ["setter", "deleter"])
    def test_rebuilt_property_keeps_the_backing_attribute(self, method):
        rebuilt = getattr(Thing.source_file, method)(lambda self, *a: None)

        class Rebuilt:
            def __init__(self) -> None:
                self._file_path = "a.h5"

            source_file = rebuilt

        assert is_restorable(Rebuilt, "source_file")
        restore(Rebuilt(), "source_file", "b.h5")  # must not raise, and must find `_file_path`

        t = Rebuilt()
        restore(t, "source_file", "b.h5")
        assert t._file_path == "b.h5"
        assert not hasattr(t, "_source_file"), "the backing name must not be re-derived"

    def test_replacing_the_getter_drops_the_mark(self):
        """`getter` is the one that does *not* carry it across, deliberately.

        `setter` and `deleter` keep the existing `fget`, so the mark rides along.
        `getter` installs a **different** function -- a new getter may well read
        a different attribute, so inheriting the old backing name would be a
        guess, and a wrong guess writes to the wrong place silently. Dropping
        it makes `is_restorable` answer False and `restore` raise, which is the
        loud failure.

        The superseded `property`-subclass implementation did the opposite: it
        overrode `getter` to preserve the name. Recorded here because the change
        in behaviour is intentional and would otherwise look like a regression.
        """
        rebuilt = Thing.source_file.getter(lambda self: "something-else")

        class Rebuilt:
            def __init__(self) -> None:
                self._file_path = "a.h5"

            source_file = rebuilt

        assert is_restorable(Rebuilt, "source_file") is False
        with pytest.raises(TypeError, match="is not a restorable property"):
            restore(Rebuilt(), "source_file", "b.h5")


class TestRestoresDecorator:
    @pytest.mark.parametrize("bad", ["", None, 0, b"_x"])
    def test_a_bad_attribute_name_raises(self, bad):
        """`restore` hands this straight to `setattr`; a bad name must fail here."""
        with pytest.raises(TypeError, match="needs a non-empty attribute name"):
            restores(bad)

    def test_the_getter_is_returned_unchanged(self):
        """The decorator must be invisible to `property`, `mypy` and `inspect`."""

        def getter(self) -> str:
            return "x"

        assert restores("_x")(getter) is getter
