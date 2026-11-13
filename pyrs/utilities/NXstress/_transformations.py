"""
pyrs/utilities/NXstress/_transformations.py

Resolution of `NXtransformations` `depends_on` chains, by the NeXus rule.

An object's `depends_on` field names the chain's entry point; each transformation
field's own `depends_on` attribute names the next link, until the string ".".
`NXtransformations.nxdl.xml`: "If a relative path is given, it is relative to the
group enclosing the `depends_on` specification" -- so a link held by a field inside
`transformations/` names its sibling bare (`translation_x`), while the object's
entry point, one group up, names it as `transformations/translation_x`.

These values are path *strings* interpreted by a reader, not `NXlink`s: HDF5 and
nexusformat never resolve them, so nothing notices a wrong one unless it is
resolved explicitly. The writers call `resolve_depends_on` on every chain right
after building it.
"""

from nexusformat.nexus import NXfield, NXgroup

from pyrs.utilities.convertdatatypes import to_text

#: The `depends_on` value that ends a chain.
CHAIN_END = "."


def resolve_depends_on(obj: NXgroup) -> list[NXfield]:
    """Follow `obj`'s `depends_on` chain, returning its links in traversal order.

    Each value is resolved from the group enclosing it: `obj` itself for the
    entry point, and each link's parent group for that link's attribute. An
    absolute value resolves from the tree's root. The tree may be detached --
    resolution walks group objects rather than `nxpath` strings, whose form
    differs between a detached and an attached tree.

    Args:
        obj: The object whose position the chain defines -- an `NXdetector` or
            `NXsample` holding a `depends_on` field.

    Returns:
        The transformation fields, entry point first, ending with the one whose
        `depends_on` is ".". Empty if `obj`'s own `depends_on` is ".".

    Raises:
        RuntimeError: If `obj` has no `depends_on`, or a link is empty, names
            nothing, names a group rather than a field, has no `depends_on` of
            its own, or revisits an earlier link.
    """
    if "depends_on" not in obj:
        raise RuntimeError(f"'{obj.nxname}' has no `depends_on`.")
    value = to_text(obj["depends_on"].nxvalue)
    enclosing = obj
    holder = f"'{obj.nxname}/depends_on'"

    chain: list[NXfield] = []
    visited: set[int] = set()
    while value != CHAIN_END:
        if not value:
            raise RuntimeError(f"{holder} is empty; a chain must end in {CHAIN_END!r}.")
        link = _resolve(enclosing, value, holder=holder)
        if id(link) in visited:
            raise RuntimeError(f"{holder} = {value!r} revisits a link: the chain is a cycle.")
        visited.add(id(link))
        chain.append(link)

        if "depends_on" not in link.attrs:
            raise RuntimeError(f"transformation '{link.nxname}' has no `depends_on` attribute.")
        value = to_text(link.attrs["depends_on"])
        enclosing = link.nxgroup
        holder = f"'{link.nxname}@depends_on'"
    return chain


def _resolve(enclosing: NXgroup, value: str, *, holder: str) -> NXfield:
    node = enclosing.nxroot if value.startswith("/") else enclosing
    for part in value.strip("/").split("/"):
        if part in ("", "."):
            continue
        if not isinstance(node, NXgroup):
            raise RuntimeError(f"{holder} = {value!r}: '{node.nxname}' is not a group.")
        if part == "..":
            if node.nxgroup is None:
                raise RuntimeError(f"{holder} = {value!r} climbs above the root.")
            node = node.nxgroup
        elif part in node:
            node = node[part]
        else:
            raise RuntimeError(f"{holder} = {value!r} names nothing: '{node.nxname}' has no '{part}'.")
    if not isinstance(node, NXfield):
        raise RuntimeError(f"{holder} = {value!r} names a group, not a transformation field.")
    return node
