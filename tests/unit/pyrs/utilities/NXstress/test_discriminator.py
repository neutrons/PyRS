"""Unit tests for `pyrs/utilities/NXstress/_discriminator.py`.

Covers the two halves of the discriminator resolver, the configuration it reads,
and the reserved-column guard that keeps a configured field from shadowing an
`NXreflections` column. The multi-workspace I/O these support is exercised in
`test_multi_workspace.py`.
"""

import numpy as np
import yaml
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns
from pyrs.utilities.restorable_property import is_restorable
from pyrs.utilities.NXstress import _discriminator
from pyrs.utilities.NXstress._peaks import _Peaks


def configure(config_override, yaml_text: str, *, validate: bool = True) -> None:
    """Apply an `nxstress` config override that production modules can actually see.

    Goes through `config_override` rather than `default_config.loadEnv(...)`:
    `default_config` *reloads* the singleton, so a module that bound `Config` the
    documented way keeps reading the previous instance and never sees the
    override. `config_override` deep-merges into the live one instead. See that
    fixture's docstring.

    `validate=False` is for the tests whose subject is a configuration
    `ConfigSchema` refuses; see `config_override`'s docstring.
    """
    config_override(yaml.safe_load(yaml_text), validate=validate)


def workspace_with_logs(**logs) -> HidraWorkspace:
    ws = HidraWorkspace("discriminator-test")
    sub_runs = np.array([1, 2, 3])
    for name, values in logs.items():
        ws.set_sample_log(name, sub_runs, np.asarray(values))
    return ws


class TestReservedColumns:
    def test_reserved_columns_matches_peaks_init(self):
        """The guard's reserved set must be exactly what `_Peaks._init` writes.

        Stated as an iteration over the real group rather than as a second
        hand-maintained list: a column added to `_init` without being added to
        `RESERVED_PEAK_COLUMNS` would otherwise silently become assignable as a
        discriminator name, and shadow itself on disk.
        """
        logs = SampleLogs()
        logs.subruns = SubRuns([1, 2, 3])

        written = set(_Peaks._init(logs))

        assert written == set(_discriminator.RESERVED_PEAK_COLUMNS)

    def test_field_colliding_with_reserved_column_raises(self, config_override):
        configure(config_override, "nxstress:\n  discriminator_fields: ['mask']\n")

        with pytest.raises(ValueError, match="collides with the reserved"):
            _discriminator.field_names()

    def test_only_a_reserved_name_encodes_onto_a_reserved_column(self):
        """Why checking the *encoded* name is sufficient, and not merely cautious.

        The guard compares `allowed_identifier(field)` against the reserved set,
        which only catches everything because the encoding is injective: no
        field spelled differently from a reserved column can encode onto one.
        Asserting that here means the guard stays sufficient if the encoding is
        ever changed -- a many-to-one encoding would let an innocuous field name
        silently land on a reserved column, the same class of collision
        Decisions row 23 removed from `allowed_identifier` itself.
        """
        for reserved in _discriminator.RESERVED_PEAK_COLUMNS:
            assert _discriminator.column_name(reserved) == reserved

        candidates = {"direction", "run_number", "HB2B:direction", "2theta", "a.b", "_x_"}
        encoded = {name: _discriminator.column_name(name) for name in candidates}

        assert not set(encoded.values()) & set(_discriminator.RESERVED_PEAK_COLUMNS)
        assert len(set(encoded.values())) == len(candidates), f"encoding is not injective: {encoded}"


class TestFieldNames:
    def test_default_is_empty(self, config_override):
        assert _discriminator.field_names() == ()

    def test_configured_names_are_returned_in_order(self, config_override):
        configure(config_override, "nxstress:\n  discriminator_fields: ['b_field', 'a_field']\n")

        assert _discriminator.field_names() == ("b_field", "a_field")

    def test_blank_name_raises(self, config_override):
        # `validate=False`: `ConfigSchema`'s `NonBlankText` refuses this at import, so
        # `field_names()`'s own blank check cannot be reached by any production path.
        # It is kept as defence-in-depth; whether it should be is DD-1's layering
        # question, not this test's.
        configure(config_override, "nxstress:\n  discriminator_fields: ['  ']\n", validate=False)

        with pytest.raises(ValueError, match="non-blank strings"):
            _discriminator.field_names()

    def test_repeated_name_raises(self, config_override):
        # `validate=False`: `ConfigSchema`'s `_distinct` refuses a repeat at import,
        # so this check too is unreachable in production. See the test above.
        configure(
            config_override,
            "nxstress:\n  discriminator_fields: ['direction', 'direction']\n",
            validate=False,
        )

        with pytest.raises(ValueError, match="more than once"):
            _discriminator.field_names()


class TestResolve:
    def test_property_is_preferred_over_a_log(self):
        ws = workspace_with_logs(name=np.array(["from-the-log"] * 3))

        assert _discriminator.resolve(ws, "name") == "discriminator-test"

    def test_unrelated_method_name_falls_through_to_the_log(self):
        """A bare `hasattr` would return the bound method instead of a value."""
        ws = workspace_with_logs(save_experimental_data=np.array(["11"] * 3))

        assert _discriminator.resolve(ws, "save_experimental_data") == "11"

    def test_constant_log_resolves_to_its_single_value(self):
        ws = workspace_with_logs(direction=np.array(["22"] * 3))

        assert _discriminator.resolve(ws, "direction") == "22"

    @pytest.mark.parametrize("name", ["direction", "sample_dir"], ids=["property", "log"])
    def test_value_varying_across_scan_points_raises(self, name):
        """As in a workspace merged in PyRS from several directions.

        Used to be `get_sample_log_value`'s bare `AssertionError`, "Single log ...
        is a np array with multiple items", which said nothing about why.
        """
        ws = workspace_with_logs(**{name: np.array(["11", "22", "33"])})

        with pytest.raises(ValueError, match=r"discriminator_fields: \[\]', or split it"):
            _discriminator.resolve(ws, name)

    def test_name_in_neither_place_raises(self):
        ws = workspace_with_logs(direction=np.array(["11"] * 3))

        with pytest.raises(ValueError, match=r"matches neither"):
            _discriminator.resolve(ws, "absent_field")


class TestUnresolvableFieldIsReportedAgainstTheConfiguration:
    """A field that resolves to nothing is a configuration error, and must read as one.

    Whether a discriminator field resolves depends on the *workspace*, so
    `field_names` cannot check it when the configuration is read -- this is the
    only place it can be detected. The failure used to surface as
    `get_sample_log_value`'s "Sample log name X is not found in allowed value
    list [...]", which names neither the configuration key nor the fact that a
    property was tried first, so a reader had no reason to look at their config.

    Note what is *not* the problem here. A non-identifier name is perfectly
    legal: `BL7:CS:Sample:Dir` encodes to `BL7__3ACS__3ASample__3ADir` via the
    `__` escape introducer and round-trips through a write and read, which is
    what that encoding exists for. A misspelling fails the same way whether or
    not it is an identifier.
    """

    @staticmethod
    def _message(name):
        ws = workspace_with_logs(direction=np.array(["11"] * 3))
        with pytest.raises(ValueError) as caught:
            _discriminator.resolve(ws, name)
        return str(caught.value)

    def test_it_names_the_configuration_key(self):
        assert "nxstress.discriminator_fields" in self._message("absent_field")

    def test_it_quotes_the_name_so_whitespace_is_visible(self):
        """The whole of the original report: ' direction ' vs 'direction'."""
        # Act
        message = self._message(" direction ")

        # Assert -- `repr`, not bare interpolation, or the spaces are invisible
        assert "' direction '" in message

    def test_it_lists_the_logs_that_are_present(self):
        assert "direction" in self._message("direciton")

    def test_it_says_when_the_name_is_a_method_not_a_property(self):
        """`getattr` would return a bound method, which is why `resolve` tests for `property`."""
        # Act
        message = self._message("save_experimental_data")

        # Assert
        assert "not a property" in message

    def test_a_constancy_failure_is_not_described_as_a_typo(self):
        """The other failure `resolve` can meet must not be re-described as a typo."""
        # Arrange -- a log that varies across scan points
        ws = workspace_with_logs(sample_dir=np.array(["11", "22", "33"]))

        # Act
        with pytest.raises(ValueError) as caught:
            _discriminator.resolve(ws, "sample_dir")

        # Assert
        assert "varies across the scan points" in str(caught.value)
        assert "typo" not in str(caught.value)


class TestApply:
    def test_log_fallback_round_trips_through_resolve(self):
        ws = workspace_with_logs(direction=np.array(["11"] * 3))

        _discriminator.apply(ws, "written_back", "22")

        assert _discriminator.resolve(ws, "written_back") == "22"

    def test_settable_property_is_used_by_both_halves(self):
        """The shape spec 05 will take, with `direction` as a real property.

        `HidraWorkspace` defines no settable property today, so the subject is
        stubbed rather than asserted of a class that has none.
        """

        class _Stubbed(HidraWorkspace):
            @property
            def direction(self):
                return self._stub_direction

            @direction.setter
            def direction(self, value):
                self._stub_direction = value

        ws = _Stubbed("stub")
        ws.set_sample_log("direction", np.array([1, 2, 3]), np.array(["11"] * 3))

        _discriminator.apply(ws, "direction", "33")

        assert _discriminator.resolve(ws, "direction") == "33"

    def test_read_only_unmarked_property_raises(self):
        """An unrestorable read-only property is refused, not written to a log.

        This test previously asserted the opposite -- that `apply` wrote a
        sample log and left the property answering with its constructor value:

            assert ws.name == "discriminator-test"
            assert ws.get_sample_log_value("name") == "written-to-the-log"

        Both assertions passed, and together they pinned the exact silent
        round-trip loss this module exists to prevent: `resolve` reads the
        property, `apply` wrote the log, and the value changed across a round
        trip with nothing raising. `_validate_round_trippable` refuses such a
        field when the configuration is read, but it runs in a different
        function, at a different time, against the class rather than the
        instance -- so `apply` now enforces it where the violation happens.
        """
        ws = workspace_with_logs(direction=np.array(["11"] * 3))

        # `name` is a read-only property on HidraWorkspace, and unmarked.
        with pytest.raises(TypeError, match="read-only property that is not marked"):
            _discriminator.apply(ws, "name", "written-to-the-log")

        assert "name" not in ws.sample_log_names

    def test_direction_is_restored_onto_the_property_not_a_log(self):
        """`direction`'s restorable path, pinned without relying on the sweep.

        An independent mutation pass found that deleting `apply`'s restorable
        branch was caught by exactly one test -- the `vars(HidraWorkspace)`
        sweep -- and only through `hidra_project_file`. `direction` is
        **self-masking**: with the branch gone `apply` writes a `direction`
        sample log, and `workspaces.py`'s legacy log fallback makes `resolve`
        answer correctly anyway, so the feature's headline property constrained
        nothing. Un-marking `hidra_project_file` *and* deleting the branch left
        the whole suite green -- two edits, neither failing anything alone.

        This defeats the mask by giving the workspace **no** `direction` log, so
        the fallback has nothing to answer with and only the property can.
        """
        ws = HidraWorkspace("no-direction-log")
        ws.set_sample_log("dummy", np.array([1, 2, 3]), np.zeros(3))
        assert "direction" not in ws.sample_log_names

        _discriminator.apply(ws, "direction", "33")

        assert ws._direction == "33"
        assert "direction" not in ws.sample_log_names, "must not have gone to a log"
        assert _discriminator.resolve(ws, "direction") == "33"

    def test_apply_normalises_like_resolve(self):
        """The two halves must agree on type, not only on value.

        `resolve` routes everything through `_as_text`; `apply` did not, so a
        `numpy.str_` from the file landed on a property annotated `-> str`.
        """
        ws = workspace_with_logs(direction=np.array(["11"] * 3))

        _discriminator.apply(ws, "direction", np.str_("33"))

        assert type(ws.direction) is str
        assert _discriminator.resolve(ws, "direction") == "33"


class TestKey:
    def test_key_is_name_ordered_regardless_of_configured_order(self):
        ws = workspace_with_logs(
            b_field=np.array(["beta"] * 3),
            a_field=np.array(["alpha"] * 3),
        )

        forward = _discriminator.key(ws, ("a_field", "b_field"))
        reversed_ = _discriminator.key(ws, ("b_field", "a_field"))

        assert forward == reversed_
        assert _discriminator.sort_values(forward) == ("alpha", "beta")

    def test_empty_names_give_an_empty_key(self):
        ws = workspace_with_logs(direction=np.array(["11"] * 3))

        assert _discriminator.key(ws, ()) == ()


class TestEveryPropertyRoundTripsOrIsRefused:
    """No `HidraWorkspace` property may be readable as a discriminator but not restorable.

    `resolve` prefers a property over a sample log, so a field naming a read-only
    property was read from the property and written back to a *log*: the value
    went to the file correctly and came back as the constructor's default, with
    nothing raising. `write(read(f))` then stored something different from `f`.

    This sweeps `vars(HidraWorkspace)` rather than naming properties, so a
    property added later is covered without editing this test -- which is the
    whole point, since the original defect was a property nobody thought about.

    Every property must land in exactly one branch, and
    `test_the_partition_covers_every_property` asserts that rather than leaving
    it to be believed: the refusal branch below has to exclude a name whose
    *encoded* form collides with a reserved `NXreflections` column, because
    `field_names` checks that first and raises a differently-worded error. That
    exclusion is a third bucket, and it is currently empty only by luck.
    """

    @staticmethod
    def _properties():
        return {name: prop for name, prop in vars(HidraWorkspace).items() if isinstance(prop, property)}

    @classmethod
    def _accepted(cls) -> list[str]:
        """Settable or restorable: `apply` can put a value back."""
        return [n for n, p in cls._properties().items() if p.fset is not None or is_restorable(HidraWorkspace, n)]

    @classmethod
    def _refused(cls) -> list[str]:
        """Plain read-only: refused with the read-only message."""
        return [
            n
            for n, p in cls._properties().items()
            if p.fset is None
            and not is_restorable(HidraWorkspace, n)
            and _discriminator.column_name(n) not in _discriminator.RESERVED_PEAK_COLUMNS
        ]

    @classmethod
    def _reserved_named(cls) -> list[str]:
        """Read-only *and* named like a reserved column: refused, worded differently."""
        return [
            n
            for n, p in cls._properties().items()
            if p.fset is None
            and not is_restorable(HidraWorkspace, n)
            and _discriminator.column_name(n) in _discriminator.RESERVED_PEAK_COLUMNS
        ]

    def test_the_sweep_finds_properties(self):
        """A sweep that matched nothing would pass every branch vacuously."""
        assert self._properties(), "no properties found; the sweep's heuristic has drifted"

    def test_the_partition_covers_every_property(self):
        """The three branches must account for every property, with none counted twice."""
        # Act
        buckets = self._accepted() + self._refused() + self._reserved_named()

        # Assert
        assert sorted(buckets) == sorted(self._properties()), (
            f"properties in no branch: {sorted(set(self._properties()) - set(buckets))}; "
            f"in more than one: {sorted({n for n in buckets if buckets.count(n) > 1})}"
        )

    def test_no_property_is_named_like_a_reserved_column(self):
        """The third bucket is empty, and the other two tests assume it stays that way.

        A read-only property whose encoded name collides with a reserved
        `NXreflections` column is still *refused* -- `field_names` checks the
        collision first -- but with the collision message rather than the
        read-only one, so `test_plain_read_only_properties_are_refused_as_fields`
        excludes it. Nothing else would notice such a property appearing.
        """
        assert not self._reserved_named(), (
            f"{self._reserved_named()} are read-only properties named like reserved columns. "
            "They are refused with the collision message, not the read-only one, so add a case "
            "for them rather than widening the exclusion in `_refused`."
        )

    def test_restorable_properties_round_trip(self, minimal_HidraWorkspace):
        # Arrange
        restorable_names = [n for n in self._properties() if is_restorable(HidraWorkspace, n)]
        assert restorable_names, "no restorable properties; expected at least `direction`"

        for name in restorable_names:
            ws = minimal_HidraWorkspace(with_instrument=False)

            # Act
            _discriminator.apply(ws, name, "sentinel-value")

            # Assert
            assert _discriminator.resolve(ws, name) == "sentinel-value", name

    def test_settable_or_restorable_properties_are_accepted_as_fields(self, config_override):
        """Both halves of "settable **or** restorable", not just the one in use.

        `HidraWorkspace` happens to have no settable property, so sweeping its
        real properties exercises only the restorable half -- measured: deleting
        `prop.fset is not None` from `_validate_round_trippable` left the whole
        suite green, so the test's name promised more than it checked. The
        settable half is covered explicitly below, on a stub, the same way
        `TestApply` covers the matching branch in `apply`.
        """
        for name in self._accepted():
            config_override({"nxstress": {"discriminator_fields": [name]}})
            assert _discriminator.field_names() == (name,), name

    def test_a_settable_property_is_accepted_as_a_field(self, config_override, monkeypatch):
        """The `prop.fset is not None` half, which no real property reaches."""

        # Arrange -- a read/write property on the class `_validate_round_trippable`
        # inspects, which is `HidraWorkspace` itself.
        monkeypatch.setattr(
            HidraWorkspace,
            "settable_stub",
            property(lambda self: getattr(self, "_stub", None), lambda self, v: setattr(self, "_stub", v)),
            raising=False,
        )
        config_override({"nxstress": {"discriminator_fields": ["settable_stub"]}})

        # Act / Assert -- accepted because it is settable, not because it is marked
        assert not is_restorable(HidraWorkspace, "settable_stub")
        assert _discriminator.field_names() == ("settable_stub",)

    def test_plain_read_only_properties_are_refused_as_fields(self, config_override):
        """Refused at configuration time, before any file exists to be written wrong."""
        # Arrange
        refusable = self._refused()
        assert refusable, "expected at least `name` to be a plain read-only property"

        for name in refusable:
            # Act / Assert
            config_override({"nxstress": {"discriminator_fields": [name]}})
            with pytest.raises(ValueError, match=r"read-only `HidraWorkspace` property"):
                _discriminator.field_names()
