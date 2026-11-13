"""
pyrs/utilities/restorable_property.py

A read-only property whose backing attribute may be restored by I/O reconstruction.

A reader that rebuilds an object from a file has to put back every value the
object would normally have been given at construction. For a plain attribute
that is easy. For a **read-only property** it is not: the value is public and
readable, but there is no supported way to write it, so a round trip silently
loses it -- the reconstructed object answers with whatever its constructor
defaulted to, and nothing raises.

The usual escapes are both unattractive. Making the property read/write puts a
setter on the public API purely for the benefit of a deserializer, inviting
casual assignment that the class did not intend to allow. Adding a
`set_<name>()` method does the same thing in different spelling. Keeping a
registry of "property -> backing attribute" somewhere else works, but it is a
second place to edit and it drifts silently when someone adds a property and
forgets.

The mark therefore goes on the **getter**, under an ordinary `@property`::

    class Thing:
        @property
        @restores("_colour")
        def colour(self) -> str: return self._colour

A reader restores through `restore()`, and code that needs to know in advance
whether a name can be restored asks `is_restorable()`. Both are total: a sweep
over `vars(cls)` partitions properties into restorable and not, so a test can
assert something about *every* property without naming any of them.

Typical use, from `pyrs/utilities/NXstress/_discriminator.py`::

    if is_restorable(ws, name):
        restore(ws, name, value)

**Why the mark is on the getter and not on a `property` subclass.** The obvious
design is a `restorable_property(property)` subclass carrying the attribute
name. It was implemented that way first, and it is wrong for a reason that only
shows up when you run a type checker: **mypy special-cases the literal
`@property` decorator syntactically**, and does not recognise a subclass used as
a decorator. Measured, on the subclass version:

    t.plain       -> "str"   @property           t.plain       = "x"  -> error: read-only
    t.viafactory  -> "Any"   @restorable()       t.viafactory  = "x"  -> (no error)

So the construct whose entire purpose was to keep a value read-only was the one
option that **removed** the static read-only guarantee, and degraded every
annotated return type to `Any` at the same time. A `TYPE_CHECKING` shim does not
recover it. Marking the getter keeps `@property` literal, so mypy keeps both.

It is also safer by construction. The subclass had to re-implement `getter`,
`setter` and `deleter` to avoid losing the attribute name, and `deleter` was
missed -- CPython's `property_copy` rebuilds through `type(self)(fget, fset,
fdel, doc)` with no way to pass extra state, so the name was silently
re-derived from the getter and `restore()` then wrote to the wrong attribute.
A mark on the function cannot be lost that way: every one of those methods
carries `fget` across unchanged.
"""

from typing import Any, Callable, TypeVar

#: Attribute set on a getter function by `restores`. Private by name because
#: nothing outside this module should read it directly -- `is_restorable` and
#: `restore` are the supported way in, and they are what the sweep tests use.
_RESTORES = "_restores_"

F = TypeVar("F", bound=Callable[..., Any])


def restores(attr: str) -> Callable[[F], F]:
    """Mark a property getter with the backing attribute an I/O reader may write.

    Applied *under* `@property`, so the property itself is an ordinary one and
    stays read-only to both the interpreter and the type checker::

        @property
        @restores("_colour")
        def colour(self) -> str: ...

    Args:
        attr: Name of the instance attribute `restore()` writes. Given
            explicitly rather than derived, so that reading the declaration
            tells you where the value lands.

    Returns:
        The getter, unchanged apart from the mark -- so the decorator is
        invisible to `property`, to `mypy`, and to `inspect`.

    Raises:
        TypeError: If `attr` is not a non-empty string. `restore()` passes it
            to `setattr`, where anything else would fail far from the cause.

    Example:
        >>> class Thing:
        ...     def __init__(self): self._colour = "blue"
        ...     @property
        ...     @restores("_colour")
        ...     def colour(self): return self._colour
        >>> Thing().colour
        'blue'
    """
    if not isinstance(attr, str) or not attr:
        raise TypeError(f"`restores` needs a non-empty attribute name, got {attr!r}.")

    def decorate(fget: F) -> F:
        setattr(fget, _RESTORES, attr)
        return fget

    return decorate


def _restorable_getter(owner: type, name: str) -> Callable[..., Any] | None:
    """The marked getter of `owner.name`, or None if there is not one."""
    prop = getattr(owner, name, None)
    if not isinstance(prop, property) or prop.fget is None:
        return None
    return prop.fget if hasattr(prop.fget, _RESTORES) else None


def is_restorable(obj: Any, name: str) -> bool:
    """Whether `name` is a restorable property of `obj`.

    Args:
        obj: An instance, or a class.
        name: Attribute name to test.

    Returns:
        True when `name` resolves to a property whose getter carries a
        `@restores(...)` mark.
    """
    owner = obj if isinstance(obj, type) else type(obj)
    return _restorable_getter(owner, name) is not None


def restore(obj: Any, name: str, value: Any) -> None:
    """Write `value` to the backing attribute of `obj`'s restorable property `name`.

    Args:
        obj: Instance being reconstructed. A **class** is rejected: restoring
            writes instance state, and `is_restorable` accepting a class while
            this did not was an asymmetry that invited exactly that call.
        name: Name of a restorable property on its type.
        value: Value to restore.

    Raises:
        TypeError: If `obj` is a class, or if `name` is not a restorable
            property of `obj` -- restoring something that merely *looks*
            assignable is how a reconstruction quietly writes to the wrong place.
        AttributeError: If the marked attribute does not exist on `obj`. A
            misspelled `@restores("_colour")` would otherwise create a second,
            unread attribute and lose the value in silence, which is the failure
            this whole module exists to prevent.

    Example:
        >>> class Thing:
        ...     def __init__(self): self._colour = "blue"
        ...     @property
        ...     @restores("_colour")
        ...     def colour(self): return self._colour
        >>> t = Thing(); restore(t, "colour", "red"); t.colour
        'red'
    """
    if isinstance(obj, type):
        raise TypeError(
            f"`restore` needs an instance, not the class '{obj.__name__}' -- it writes instance "
            f"state. (`is_restorable` accepts either, which is why this is worth saying.)"
        )

    fget = _restorable_getter(type(obj), name)
    if fget is None:
        raise TypeError(
            f"'{type(obj).__name__}.{name}' is not a restorable property, so it cannot be restored.\n"
            f"  Mark its getter with `@restores('_<attribute>')` under `@property` if an I/O reader "
            f"is expected to put its value back."
        )

    attr = getattr(fget, _RESTORES)
    if not hasattr(obj, attr):
        raise AttributeError(
            f"'{type(obj).__name__}.{name}' declares `@restores({attr!r})`, but the instance has no "
            f"attribute '{attr}'.\n"
            f"  Setting it anyway would create a second attribute that '{name}' never reads, losing "
            f"the restored value silently. Check the spelling against what `__init__` assigns."
        )
    setattr(obj, attr, value)
