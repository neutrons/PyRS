"""
pyrs/utilities/unset.py

`Unset`: the value of an argument the caller did not supply.

`None` already means something in most of PyRS's domains -- "no mask", "no
calibration file", "no value for this scan point" -- so it cannot also mean "the
caller passed nothing". When a default has to be decided at call time rather
than written into the signature, `Unset` stands in for "not supplied", and
`None` keeps its own meaning.

The usual reason is a default that comes from `pyrs.utilities.config.Config`.
A signature default is evaluated once, when the module is imported, so
`def f(run_number: int = Config["nxstress.default_run_number"])` would ignore any
later override -- including a test's `config_override`. With `Unset` the
signature says "not supplied" and the body looks the default up when called::

    from pyrs.utilities.unset import Unset

    def f(run_number: int | Unset = Unset) -> int:
        if run_number is Unset:
            run_number = Config["nxstress.default_run_number"]
        return run_number

`Unset` works the way `None` does: the *instance* goes directly in an
annotation, `int | Unset`, and `if x is Unset:` narrows `x` to `int` for the
type checker. That is the PEP 661 sentinel, from `typing_extensions`, verified
against the installed mypy: narrowing yields `int`, a wrong argument type is
reported, and identity survives copying and pickling.

Always test with `is Unset`. Like any object, `Unset` is truthy, so a bare
`if not x:` cannot tell it apart from anything else -- which is the point: "not
supplied" must never be confused with a falsey value.
"""

from typing_extensions import Sentinel

Unset = Sentinel("Unset")
