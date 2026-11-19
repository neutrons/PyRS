"""
pyrs/utilities/NXstress/_workspace_key.py

The workspace key: what tells one input workspace's rows from another's in an entry.

Several `HidraWorkspace`s may share one NXentry. On disk their rows are told apart
by two things:

- the **discriminator values** of the workspace, one column per configured field
  (`_discriminator`);
- its **run**, when the entry records the run as part of the key
  (`_runs.BOUNDARY_ATTRIBUTE`). Two runs sharing every discriminator value -- two
  runs in one strain direction -- are then two workspaces, not one.

`WorkspaceKey` carries the two as separate fields. Write, validation, append and
read all build and compare keys, so the key and the rules for deriving it live
here, owned by none of them.

Typical workflow:

1. `entry_discriminator_names(wss)` and `entry_carries_run(wss)` decide, once
   per entry, which fields and whether the run go into every key.
2. `WorkspaceKey.of(ws, names, carries_run=...)` builds one workspace's key.
3. `WorkspaceKey.sort_values()` gives the key's place in the on-disk row order,
   and `describe()` names it in a message.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.unset import Unset

from ._definitions import logger
from ._discriminator import DiscriminatorKey
from . import _discriminator, _runs

# Every `(name, value)` pair of a workspace key, the run included under its
# column's name: a `DiscriminatorKey` that may also hold `("run_number", run)`.
KeyPairs = tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class WorkspaceKey:
    """The identity of one input workspace within an entry.

    Two keys are equal exactly when their rows belong to the same workspace on
    read, which is what duplicate detection, append's Case A/B classification
    and the read-side split all compare.

    Attributes:
        discriminators: `(name, value)` pairs, sorted by name. Empty when the
            entry records no discriminator field.
        run: The workspace's run, or `Unset` when `of` supplies none because the
            entry's key does not carry the run -- an entry holding one workspace
            merged from several runs, or one written before the run index.
            Keyword-only.

    Example:
        >>> WorkspaceKey((("direction", "11"),), run=1327).as_pairs()
        (('direction', '11'), ('run_number', 1327))
    """

    discriminators: DiscriminatorKey = ()
    run: int | Unset = field(default=Unset, kw_only=True)

    def __post_init__(self) -> None:
        # Equality is what tells workspaces apart, so a key built out of order, or
        # with a numpy integer for a run, must not quietly compare unequal to the
        # same key built by `of`. Refused, never re-sorted: re-sorting would hide
        # the caller's mistake.
        names = [name for name, _ in self.discriminators]
        if names != sorted(set(names)):
            raise ValueError(f"NXstress: discriminator names must be distinct and sorted by name, got {names}.")
        if self.run is not Unset and type(self.run) is not int:
            raise TypeError(f"NXstress: a workspace key's run must be an `int` or `Unset`, got {self.run!r}.")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # Signatures under `validate_call_` reach this type through `IndexedPeaks`.
        # Pydantic would otherwise validate a dataclass field by field, and cannot
        # validate `Unset` -- a sentinel instance, not a type -- so it warns and
        # accepts anything there. A key is only ever built by this module, so an
        # instance check is the whole of what validation can usefully say.
        return core_schema.is_instance_schema(cls)

    @classmethod
    def of(cls, ws: HidraWorkspace, names: tuple[str, ...], *, carries_run: bool) -> "WorkspaceKey":
        """The key of one input workspace.

        Args:
            ws: The workspace.
            names: The discriminator fields this entry is written with
                (`entry_discriminator_names`).
            carries_run: Whether the entry's key carries the run
                (`entry_carries_run`). When it does, the workspace holds exactly
                one run -- write and append both refuse a merged one first.
        """
        discriminators = _discriminator.key(ws, names)
        if not carries_run:
            return cls(discriminators)
        runs = _runs.run_numbers(ws._sample_logs)
        if len(runs) != 1:
            raise ValueError(
                f"NXstress: workspace '{ws.name}' holds {len(runs)} runs ({sorted(runs)}), and its key can carry "
                "only one; it has no scan points, or was merged from several runs."
            )
        (run,) = runs
        return cls(discriminators, run=int(run))

    def as_pairs(self) -> KeyPairs:
        """Every `(name, value)` pair of the key, the run included, sorted by name.

        The run is ordered among the discriminator names, under the name of its
        own column. That merged order is the on-disk row order (`sort_values`).
        """
        if self.run is Unset:
            return self.discriminators
        return tuple(sorted((*self.discriminators, (_runs.RUN_NUMBER, self.run))))

    def sort_values(self) -> tuple:
        """The key's values in name order: the slowest-varying prefix of the peak index's sort key."""
        return _discriminator.sort_values(self.as_pairs())

    def describe(self) -> str:
        """The key as a message names a block: `" for {...}"`, or `""` for an empty key."""
        pairs = self.as_pairs()
        return f" for {dict(pairs)}" if pairs else ""


def entry_discriminator_names(wss: list[HidraWorkspace]) -> tuple[str, ...]:
    """Discriminator fields to write this entry with.

    Whenever any field is configured, its column is written -- `N == 1`
    included. That keeps the read-side cross-check a plain equality test,
    and lets a single-workspace save round-trip its discriminator value
    the same way a multi-workspace one does.

    **Except** for a single workspace that has no single value for a
    configured field: none at all (a PeakFitting save, with
    `discriminator_fields: ["direction"]` shipped), or one varying across its
    scan points (a workspace merged from several directions in PyRS). A
    discriminator records the boundary *between* workspaces, and one
    workspace has none to record, so the entry records no discriminator
    field -- an empty `discriminator_fields` attribute, which every read
    takes as one workspace whatever is configured. Nothing is lost: the
    per-scan-point log, if there is one, is retained with the other logs.
    With more than one workspace the field is the only boundary marker, so
    there `_discriminator.resolve` still refuses.
    """
    names = _discriminator.field_names()
    if len(wss) != 1 or not names:
        return names
    (ws,) = wss
    for name in names:
        try:
            _discriminator.resolve(ws, name)
        except ValueError as error:
            logger.info(
                f"NXstress: workspace '{ws.name}' has no single value for discriminator field "
                f"{name!r}, so this single-workspace entry records no discriminator field.\n"
                f"  ({str(error).splitlines()[0]})"
            )
            return ()
    return names


def entry_carries_run(wss: list[HidraWorkspace]) -> bool:
    """Whether this entry's workspace key includes the run (`_runs.BOUNDARY_ATTRIBUTE`).

    True when every input workspace holds a single run. It is what lets two
    runs sharing every discriminator value -- two runs in one strain
    direction -- come back as two workspaces rather than one. False only for
    a single workspace merged from several runs in PyRS, whose rows must be
    read back together; write-time validation refuses such a workspace
    alongside others.
    """
    return all(_runs.single_run(ws._sample_logs) for ws in wss)
