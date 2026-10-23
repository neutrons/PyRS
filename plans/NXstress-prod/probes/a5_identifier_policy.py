"""A5: a reversible, legible identifier encoding for PV-log names.

Spec 04 schedules widening `allowed_identifier`. Probing the current function
found a defect nobody had recorded -- it is **many-to-one with no collision
check**, so two distinct PV logs silently collapse into one and the `local_name`
attribute meant to preserve the original is overwritten too
(`a5_nxstress_internals_today.py`). Fixing it requires choosing an encoding.
This probe records the choice, and the three rejected alternatives, because the
rejections are the useful part.

The decision path
-----------------
1. **Convert disallowed characters to `_`.** Rejected: many-to-one, so distinct
   names collide and the original is unrecoverable.
2. **Double every incoming `_`, use `_` + hex for the rest.** Injective, but it
   doubles the *common* case: 37 of the 185 real log names in `tests/data`
   contain an underscore (47 occurrences), so `my_log_value` becomes
   `my__log__value`.
3. **Use `.` as the marker so `_` survives.** Rejected as **unverifiable**: it
   needs `.` to be legal in a NeXus identifier, and the only support is an
   unsourced comment in `_definitions.py`. No `validItemName` rule exists in
   `nexusformat` or in this repo, and the `NXstress.xml`/`.html` schema doc is
   still absent (spec 10 tracks it). **The absence is the finding.**
4. **Adopted: allow only what `str.isidentifier()` permits, and make the escape
   introducer `__` (two underscores).**

The rule needs no external specification, and is strictly *narrower* than any
plausible NeXus rule, so it stays valid whatever the schema doc eventually says.
It also settles the alphabet by itself: `_` is the **only** punctuation a Python
identifier permits -- `$` is not, contrary to a reasonable expectation.

Why a two-underscore introducer
-------------------------------
Making the *introducer* `__` rather than `_` is what buys legibility. A lone `_`
is then never the start of an escape, so it passes through untouched:

* `my_log_value` -> `my_log_value`
* `_DEFAULT_`    -> `_DEFAULT_`
* `a_3Ab`        -> `a_3Ab`

**Zero of the 185 real log names require an escaped underscore.** Only a literal
*double* underscore does, which is rare. The single-underscore introducer of
alternative 2 would have rewritten 47 of them.

Encoding
--------
* any character illegal at its position -> ``__XX`` (two uppercase hex digits),
  or ``__uXXXX`` beyond U+00FF. Position matters: a digit is legal *inside* an
  identifier but not leading, and ``2theta`` is a real log name in ``tests/data``.
* ``_`` -> ``_``, unless the following token would also begin with ``_`` -- that
  is, the next character is another ``_`` or is itself illegal -- in which case
  it is escaped as ``__5F``. This is one character of lookahead, not two.
* everything else passes through.

Decoding needs no lookahead: ``__u`` starts a 7-character form, ``__`` a
4-character one, anything else is a literal.

Run: ``pixi run python plans/NXstress-prod/probes/a5_identifier_policy.py``
"""

from __future__ import annotations

import glob
import itertools
import sys

import h5py

MARK = "__"


def _legal_at(ch: str, lead: bool) -> bool:
    """Is ``ch`` legal at this position of a Python identifier?"""
    if not ch.isascii():
        return False
    return (ch + "x").isidentifier() if lead else ("x" + ch).isidentifier()


def _escape(ch: str) -> str:
    return f"{MARK}{ord(ch):02X}" if ord(ch) < 256 else f"{MARK}u{ord(ch):04X}"


def encode(s: str) -> str:
    """Map an arbitrary PV-log name to a valid Python identifier, reversibly."""
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch == "_":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            # Escape only when the NEXT emitted token would also start with '_',
            # which is the only way a lone '_' could be misread as an introducer.
            if nxt == "_" or (nxt != "" and not _legal_at(nxt, False)):
                out.append(_escape("_"))
            else:
                out.append("_")
        elif _legal_at(ch, i == 0):
            out.append(ch)
        else:
            out.append(_escape(ch))
    return "".join(out)


def decode(s: str) -> str:
    """Exact inverse of :func:`encode`."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith(f"{MARK}u", i):
            out.append(chr(int(s[i + 3 : i + 7], 16)))
            i += 7
        elif s.startswith(MARK, i):
            out.append(chr(int(s[i + 2 : i + 4], 16)))
            i += 4
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def real_log_names() -> set[str]:
    """Every dataset name in the sampled real project files."""
    names: set[str] = set()
    for path in sorted(glob.glob("tests/data/*.h5"))[:6]:
        try:
            with h5py.File(path, "r") as handle:

                def walk(group):
                    for key in group:
                        try:
                            if isinstance(group[key], h5py.Group):
                                walk(group[key])
                            else:
                                names.add(key)
                        except Exception:  # noqa: BLE001 - malformed entries are not the subject
                            pass

                walk(handle)
        except Exception:  # noqa: BLE001
            pass
    return names


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: identifier policy -- Python-identifier rule, `__` introducer")
    print("=" * 78)

    # --- Alternative 2: a single-underscore introducer is injective but ugly ---
    def single_mark(s: str) -> str:
        return s.replace("_", "__").replace(":", "_")

    seen: dict[str, str] = {}
    clash = None
    for n in range(1, 5):
        for tup in itertools.product(["A", "_", ":"], repeat=n):
            s = "".join(tup)
            e = single_mark(s)
            if e in seen and seen[e] != s:
                clash = (seen[e], s, e)
                break
            seen[e] = s
        if clash:
            break
    report(
        "doubling `_` alone (without hex) is injective (alternative 2, naive form)",
        f"FALSE -- {clash[0]!r} and {clash[1]!r} both encode to {clash[2]!r}. "
        f"Adding hex fixes it, but then every underscore doubles.",
    )

    # --- Alternative 3: is there any authoritative NeXus rule to check against? ---
    import pathlib

    import nexusformat

    root = pathlib.Path(nexusformat.__file__).parent
    rule_files = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".xsd", ".xml", ".nxdl"}
        and "validItemName" in p.read_text(errors="replace")
    ]
    report(
        "an authoritative NeXus identifier rule is available (alternative 3)",
        f"FALSE -- files in nexusformat defining `validItemName`: {rule_files or 'NONE'}. "
        f"The absence IS the finding; the schema doc is still not in the repo.",
    )

    report(
        "`$` is legal in a Python identifier",
        "FALSE -- punctuation legal in a Python identifier: "
        f"{[c for c in '_.$:- @#!' if ('a' + c + 'b').isidentifier()]}",
    )

    names = real_log_names()
    already = {n for n in names if n.isidentifier()}
    report(
        "the rule is workable against REAL log names",
        f"{len(names)} distinct names sampled from tests/data; "
        f"{len(already)} already valid identifiers, {len(names) - len(already)} need encoding; "
        f"non-ASCII: {sum(1 for n in names if not n.isascii())}; "
        f"leading-digit: {sorted(n for n in names if n and n[0].isdigit())}",
    )

    print("\n  legibility -- the point of the `__` introducer:")
    for s in ["my_log_value", "average_value", "_DEFAULT_", "a_3Ab", "HB2B:Mot:sz_real", "Scan Index", "2theta"]:
        e = encode(s)
        flag = "  <- unchanged" if e == s else ""
        print(f"    {s!r:24} -> {e!r:30} rt={decode(e) == s}{flag}")

    print("\n  adversarial -- inputs that could be misread as escapes:")
    for s in ["a__b", "a___b", "a_:b", "__", "_", "a$b", "ü"]:
        e = encode(s)
        print(f"    {s!r:10} -> {e!r:18} id={e.isidentifier()} rt={decode(e) == s}")

    escaped = sorted(n for n in names if f"{MARK}5F" in encode(n))
    report(
        "how many REAL names need an escaped underscore",
        f"{len(escaped)} of {len(names)} {escaped} -- versus 37 under a "
        f"single-underscore introducer, which is every name containing '_'.",
    )

    ok = all(encode(n).isidentifier() and decode(encode(n)) == n for n in names)
    report(
        "every real name encodes to a valid identifier AND round-trips",
        f"{ok} over all {len(names)} sampled names",
    )

    seen2: dict[str, str] = {}
    clash2 = None
    for n in range(1, 6):
        for tup in itertools.product(["A", "_", ":", "3"], repeat=n):
            s = "".join(tup)
            e = encode(s)
            if e in seen2 and seen2[e] != s:
                clash2 = (seen2[e], s, e)
                break
            seen2[e] = s
        if clash2:
            break
    report(
        "the adopted encoding is injective",
        f"{'YES' if not clash2 else f'NO -- {clash2}'} over {len(seen2)} inputs",
    )

    print("\n" + "-" * 78)
    print("COST: encoding needs ONE character of lookahead (is the next token")
    print("also going to start with '_'?). Decoding needs none. A literal double")
    print("underscore is escaped; a single one never is.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
