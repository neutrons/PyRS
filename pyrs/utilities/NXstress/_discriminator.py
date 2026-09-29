"""
pyrs/utilities/NXstress/_discriminator.py

Discriminator-field support for multi-workspace NXstress I/O.

When more than one `HidraWorkspace` is written into a single `NXentry`, the
boundary between them has to be recoverable at read time. It is recovered from
**discriminator columns** on the combined peak index: one extra column per
configured field name, carrying that workspace's value for the field on every
row the workspace contributed.

Which fields those are is deployment policy, not a per-call argument, so it
comes from `pyrs.utilities.config.Config["nxstress.discriminator_fields"]`.
Typical workflow:

1. `field_names()` reads and validates the configured names.
2. `resolve(ws, name)` takes each name's value off each input workspace.
3. `key(ws, names)` packages those as a name-keyed, order-independent tuple,
   which `_peaks.py` writes to disk and sorts on.
4. On read, `apply(ws, name, value)` puts the value back onto each
   reconstructed workspace, so a caller can tell them apart by inspecting the
   same field it originally supplied.

Values are carried **name-keyed** rather than positionally: `discriminator_fields`
could be reordered in config between a write and a later read, and a positional
tuple would silently misattribute values in that case.
"""

import logging
from typing import Any

import numpy as np

from pyrs.core.workspaces import HidraWorkspace

from pyrs.utilities.config import Config

from pyrs.utilities.restorable_property import is_restorable, restore

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import allowed_identifier


_logger = logging.getLogger(__name__)


# Column names `_Peaks._init` writes onto the `peaks` (NXreflections) group.
# A discriminator field may not encode onto any of these: it would either
# shadow the reserved column or be mistaken for one at read time, and both
# corrupt a round trip with no error to signal it.
#
# Pinned against the real group by
# `plans/NXstress-prod/probes/a4_nxstress_extra_columns.py`, and by
# `test_discriminator.py::test_reserved_columns_matches_peaks_init`, so that
# adding a column to `_init` without extending this set fails a test rather
# than silently narrowing the guard.
RESERVED_PEAK_COLUMNS = frozenset(
    {
        "scan_point",
        "h",
        "k",
        "l",
        "phase_name",
        "mask",
        "qx",
        "qy",
        "qz",
        "center",
        "center_errors",
        "center_type",
        "sx",
        "sy",
        "sz",
    }
)


# A discriminator key: `(name, value)` pairs sorted by name, so two keys built
# from the same fields compare equal regardless of the order config listed them.
DiscriminatorKey = tuple[tuple[str, Any], ...]


# Attribute on the `peaks` group naming the discriminator fields the entry was
# written with -- the field names, not their encoded column names. An empty list
# records that the entry holds no workspace boundary. See `names_for_read` for
# how it is used, and `plans/PRs/phase-2.md` for the rules in full.
FIELDS_ATTRIBUTE = "discriminator_fields"


def field_names() -> tuple[str, ...]:
    """Configured discriminator field names, validated.

    Returns:
        The names from `nxstress.discriminator_fields`, in configured order.
        Empty when no discriminator is configured, which is the default.

    Raises:
        ValueError: If a name is blank, if the same name appears twice, or if a
            name encodes onto a reserved `NXreflections` column.
    """
    try:
        configured = Config["nxstress.discriminator_fields"]
    except Exception:  # noqa: BLE001 - a missing or broken key is "none configured"
        _logger.warning(
            "NXstress._discriminator: config key 'nxstress.discriminator_fields' could not be read;\n"
            "  proceeding as if no discriminator fields were configured."
        )
        return ()

    if configured is None:
        return ()
    if not isinstance(configured, (list, tuple)):
        raise ValueError(
            f"Config['nxstress.discriminator_fields'] must be a list of field names, "
            f"got {configured!r} ({type(configured).__name__})."
        )

    names = tuple(configured)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Config['nxstress.discriminator_fields'] entries must be non-blank strings, "
                f"got {name!r} in {list(names)}."
            )
    if len(set(names)) != len(names):
        raise ValueError(
            f"Config['nxstress.discriminator_fields'] names the same field more than once: {list(names)}."
        )

    for name in names:
        collision = column_name(name)
        if collision in RESERVED_PEAK_COLUMNS:
            raise ValueError(
                f"Discriminator field '{name}' collides with the reserved `NXreflections` "
                f"column '{collision}'.\n"
                f"  Reserved: {sorted(RESERVED_PEAK_COLUMNS)}.\n"
                f"  Rename the field in `nxstress.discriminator_fields`."
            )
        _validate_round_trippable(name)
    return names


def _validate_round_trippable(name: str) -> None:
    """Reject a field that `resolve` could read but `apply` could not put back.

    `resolve` prefers a `HidraWorkspace` property over a sample log, so a field
    naming a **read-only** property is readable but not restorable: the value
    would be written to the file correctly and then lost on read, with the
    reconstructed workspace answering its constructor default instead. Nothing
    raises -- `write(read(f))` simply stores something different from `f`.

    Checked here, against configuration, rather than at write time: a field that
    cannot round-trip is a deployment mistake, and the cheapest moment to say so
    is before any file exists.

    A name that is not a property at all is left alone -- it will be resolved as
    a sample log, which round-trips through `set_sample_log`, and whether that
    log exists cannot be known from configuration.

    Args:
        name: A configured discriminator field name.

    Raises:
        ValueError: If `name` is a read-only `HidraWorkspace` property that is
            neither settable nor marked `@restores(...)`.
    """
    if not name.isidentifier():
        return
    prop = getattr(HidraWorkspace, name, None)
    if not isinstance(prop, property):
        return
    if prop.fset is not None or is_restorable(HidraWorkspace, name):
        return
    raise ValueError(
        f"Discriminator field '{name}' is a read-only `HidraWorkspace` property, so its value "
        f"could be written to a file but never restored when the file is read back.\n"
        f"  Mark its getter `@restores('_<attribute>')` in `pyrs/core/workspaces.py` if it should participate in "
        f"NXstress round trips, or name a sample log instead."
    )


def column_name(name: str) -> str:
    """On-disk column name for a discriminator field.

    Uses the same reversible encoding `_sample.py` applies to retained log
    names, so a field name containing characters NeXus disallows still round
    trips exactly.
    """
    return allowed_identifier(name)


_ONE_VALUE_PER_WORKSPACE = (
    "  A discriminator needs one value per workspace. Write a workspace merged across several "
    "values with 'nxstress.discriminator_fields: []', or split it."
)


def resolve(ws: HidraWorkspace, name: str) -> Any:
    """This workspace's value for one discriminator field.

    Prefers a matching `@property` when the class defines one, and otherwise
    falls back to the sample log of the same name.

    The `isinstance(..., property)` test is doing real work, and a bare
    `hasattr` would not do it: `HidraWorkspace` has methods whose names could
    plausibly be configured as fields (`save_experimental_data`, say), and
    `getattr` on one returns a bound method rather than a value.

    Args:
        ws: The workspace to take the value from.
        name: A configured discriminator field name.

    Returns:
        The single value this workspace carries for the field.

    Raises:
        ValueError: If `name` matches neither a property nor a sample log. The
            message names the configuration key, because this is where a typo in
            it first becomes visible and the workspace is the only place it
            *can* be detected: whether a field resolves depends on the data, so
            `field_names` cannot check it when the configuration is read.
            Also if the value is not constant across the workspace's scan
            points -- as in a workspace merged from several directions -- or the
            property raises `ValueError`: a discriminator needs exactly one
            value per workspace.
    """
    attribute = getattr(type(ws), name, None) if name.isidentifier() else None
    if isinstance(attribute, property):
        try:
            return _as_text(getattr(ws, name))
        except ValueError as error:
            raise ValueError(
                f"NXstress: discriminator field {name!r} has no single value for workspace "
                f"'{ws.name}'.\n  {error}\n{_ONE_VALUE_PER_WORKSPACE}"
            ) from error

    if name in ws.sample_log_names:
        # Checked here rather than left to `get_sample_log_value`, whose
        # `AssertionError` for a varying log says nothing about why.
        distinct = sorted({_as_text(value) for value in ws.get_sample_log_values(name)}, key=str)
        if len(distinct) > 1:
            raise ValueError(
                f"NXstress: discriminator field {name!r} varies across the scan points of workspace "
                f"'{ws.name}' ({distinct}), so it cannot tell that workspace apart from others.\n"
                f"{_ONE_VALUE_PER_WORKSPACE}"
            )
        return _as_text(ws.get_sample_log_value(name))

    tried = "a sample log" if attribute is None and not name.isidentifier() else "a property, then a sample log"
    detail = ""
    if attribute is not None:
        # An identifier that names a method rather than a property -- `getattr`
        # would return a bound method, which is why `resolve` tests for
        # `property` rather than using `hasattr`.
        detail = (
            f"\n  '{name}' names a {type(attribute).__name__} on `HidraWorkspace`, not a property, "
            "so it cannot carry a value."
        )
    raise ValueError(
        f"NXstress: discriminator field {name!r} matches neither {tried} of this workspace.\n"
        f"  It comes from the 'nxstress.discriminator_fields' configuration key; check it for a "
        f"typo, including leading or trailing spaces -- the name is quoted above exactly as "
        f"configured.{detail}\n"
        f"  Sample logs present: {sorted(ws.sample_log_names)}"
    )


def _as_text(value: Any) -> Any:
    """Normalise a string value to `str`, leaving everything else alone.

    HDF5 hands string data back as `bytes` even when it was written through the
    variable-length *UTF-8* dtype, so a value read from a file is `b"11"` where
    `"11"` was written. Left alone, a discriminator would therefore stop
    comparing equal to itself across a round trip -- and, because the comparison
    is what groups rows by workspace, would split one workspace into two rather
    than raise. Evidence:
    `plans/NXstress-prod/probes/a4_string_log_dtypes.py`.
    """
    if isinstance(value, (bytes, np.bytes_, np.str_)):
        return to_text(value)
    return value


def apply(ws: HidraWorkspace, name: str, value: Any) -> None:
    """Put a discriminator value back onto a reconstructed workspace.

    Mirrors `resolve`, so a field backed by a property round-trips rather than
    being read from one place and written to another. Three cases, in order: a
    read/write property is assigned; a read-only property marked `@restores(...)`
    has its declared backing attribute written; a non-property becomes a sample
    log; and a property that is neither settable nor restorable **raises**.

    The middle case is the one that is easy to omit, and omitting it is silent:
    `resolve` reads any property, so a read-only one would be read from the
    property and written to a log, and the reconstructed workspace would answer
    with its constructor default instead of the stored value.

    Args:
        ws: A workspace reconstructed by `NXstress.read`, already carrying its
            sample logs -- `set_sample_log` needs the scan points.
        name: A configured discriminator field name.
        value: The value recovered from the file.
    """
    # Normalised exactly as `resolve` normalises what it reads. Without this the
    # two halves disagree on type: `resolve` returns `str` while `apply` would
    # write back whatever the file handed up, so `direction` could come out as
    # `numpy.str_` against a property annotated `-> str`.
    value = _as_text(value)

    prop = getattr(type(ws), name, None) if name.isidentifier() else None
    if isinstance(prop, property) and prop.fset is not None:
        setattr(ws, name, value)
        return

    # A read-only property that declares a backing attribute. Without this branch
    # `resolve` would read the property while `apply` wrote a sample log, so the
    # value silently changed across a round trip -- the property kept answering
    # with whatever the constructor defaulted to.
    if is_restorable(ws, name):
        restore(ws, name, value)
        return

    if isinstance(prop, property):
        # Read-only and unmarked: `resolve` would read this property while the
        # line below wrote a sample log, and the round trip would silently return
        # the constructor default. `_validate_round_trippable` refuses such a
        # field when the configuration is read, but that runs in a different
        # function, at a different time, against the class rather than this
        # instance -- so the invariant is enforced here too, where it is violated.
        raise TypeError(
            f"'{type(ws).__name__}.{name}' is a read-only property that is not marked "
            f"`@restores(...)`, so a value read from the file cannot be put back onto it.\n"
            f"  Writing a sample log instead would leave the property answering with its "
            f"constructor default, and nothing would raise."
        )

    sub_runs = ws.get_sub_runs().raw_copy()
    ws.set_sample_log(name, sub_runs, np.full(len(sub_runs), value))


def key(ws: HidraWorkspace, names: tuple[str, ...]) -> DiscriminatorKey:
    """Name-keyed discriminator key for one input workspace.

    Args:
        ws: The workspace to take values from.
        names: Configured field names, in any order.

    Returns:
        `(name, value)` pairs sorted by name. Empty when `names` is empty,
        which is the single-workspace case.
    """
    return tuple(sorted((name, resolve(ws, name)) for name in names))


def sort_values(disc_key: DiscriminatorKey) -> tuple:
    """The values of a discriminator key, in name order.

    This is the slowest-varying prefix of the peak index's sort key. Name order
    -- not configured order -- is what makes the resulting row order depend only
    on the field *set*, so reordering `discriminator_fields` in config cannot
    change what a write produces.
    """
    return tuple(value for _, value in disc_key)


def columns_on_disk(peaks) -> set[str]:
    """Columns on a `peaks` (NXreflections) group that PyRS does not reserve.

    For a PyRS-written entry these are its discriminator columns. They are not
    *assumed* to be: another NXstress producer may add reflection fields PyRS
    does not know, so which columns discriminate is decided by configuration
    (cross-checked against `FIELDS_ATTRIBUTE` when the entry records it), and any
    other column is ignored on read and refused on append.
    """
    return {str(name) for name in peaks} - RESERVED_PEAK_COLUMNS


def recorded_names(peaks) -> tuple[str, ...] | None:
    """The discriminator fields an entry records it was written with.

    Returns:
        The names in `FIELDS_ATTRIBUTE`, sorted; an empty tuple when the entry
        records no workspace boundary; `None` when the attribute is absent -- a
        file from another producer, or one PyRS wrote before the attribute.
    """
    if FIELDS_ATTRIBUTE not in peaks.attrs:
        return None
    value = peaks.attrs[FIELDS_ATTRIBUTE]
    value = getattr(value, "nxvalue", value)
    # nexusformat reads a one-element string array back as a bare string.
    return tuple(sorted(to_text(name) for name in np.atleast_1d(np.asarray(value, dtype=object))))


def names_for_read(peaks) -> tuple[str, ...]:
    """Discriminator field names to read a `peaks` group with, cross-checked.

    **Configuration is the authority** on which fields discriminate. The entry
    is consulted only to refuse a read that would split it wrongly:

    - When the entry records its fields (`FIELDS_ATTRIBUTE`):
      - an empty record means it holds no workspace boundary -- it was
        written from a single workspace with no field configured -- and it is
        read as one workspace whatever is configured;
      - otherwise the record must equal the configured set, or the read is
        refused. Reading with fewer fields can silently merge workspaces (see
        `_Peaks._validateDiscriminatorNames`), so the refusal names the fields
        to configure.
    - When it does not (a file from another producer, or a PyRS file written
      before the attribute):
      - if none of the configured fields has a column, it is read as one
        workspace;
      - if every one does, they are used;
      - if only some do, the read is refused.

    In every case, any other column the entry carries is ignored: it may be a
    reflection field from another producer, and cannot be told apart from an
    unconfigured discriminator.

    NXstress never merges workspaces: an entry it writes from N > 1
    workspaces always records the boundary between them. Merging happens in
    PyRS before NXstress is called (`HidraWorkspace.append_hidra_project`, used
    by `CombineRunsModel.combine_project_files`, which then passes a length-1
    list). See Decisions Log row 45 in `plans/NXstress-prod/README.md`.

    Args:
        peaks: The `peaks` (NXreflections) group being read.

    Returns:
        The configured field names, or an empty tuple when the entry is read as
        one workspace.

    Raises:
        RuntimeError: If the entry records fields other than the configured
            ones, if it records fields whose columns are missing, or if only some
            configured fields have a column.
    """
    configured = field_names()
    recorded = recorded_names(peaks)
    present = {name for name in configured if column_name(name) in peaks}

    ignored = columns_on_disk(peaks) - {column_name(name) for name in (recorded or configured)}
    if ignored:
        _logger.info(
            f"NXstress._discriminator: ignoring column(s) {sorted(ignored)} on this entry's `peaks` group;\n"
            f"  they are not among the discriminator fields {list(configured)}."
        )

    if recorded is not None:
        if not recorded:
            if configured:
                _logger.info(
                    f"NXstress._discriminator: this entry records no workspace boundary, while "
                    f"'nxstress.discriminator_fields' names {list(configured)};\n"
                    f"  reading it as a single workspace."
                )
            return ()
        if set(recorded) != set(configured):
            raise RuntimeError(
                f"NXstress: this entry was written with discriminator fields {list(recorded)}, "
                f"but 'nxstress.discriminator_fields' is {list(configured)}.\n"
                f"  Set 'nxstress.discriminator_fields' to {list(recorded)} to read it: splitting "
                f"it by any other set can silently merge or mis-split its workspaces."
            )
        missing = [name for name in recorded if column_name(name) not in peaks]
        if missing:
            raise RuntimeError(
                f"NXstress: this entry records discriminator fields {list(recorded)}, but has no "
                f"column for {missing}; the file is damaged."
            )
        return configured

    if not present:
        if configured:
            _logger.info(
                f"NXstress._discriminator: this entry carries no column for any of "
                f"'nxstress.discriminator_fields' = {list(configured)};\n"
                f"  reading it as a single workspace."
            )
        return ()
    if len(present) != len(configured):
        raise RuntimeError(
            f"NXstress: this entry has columns for discriminator fields {sorted(present)} but not "
            f"for {sorted(set(configured) - present)}, all of which 'nxstress.discriminator_fields' "
            f"names.\n"
            "  Configure the fields the file was written with, or none to read it as one workspace."
        )
    return configured
