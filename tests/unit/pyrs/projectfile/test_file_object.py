"""
Source-scan guarantees for pyrs/projectfile/file_object.py.

These do not exercise behaviour; they pin a property of the source itself. The
plan series that produced them scheduled an audit of two "legacy log name" FIXMEs
in this module, and that work turned out to have been done upstream already --
the silent fallback to a capitalized "2Theta" key was replaced by an explicit
RuntimeError telling the user to re-reduce. A whole planning cycle was spent
re-discovering that, which a test would have reported on the first commit after
the fix landed.
"""

import ast
import inspect
import re

import pyrs.projectfile.file_object as file_object_module


#: What a reintroduced legacy-log-name FIXME would mention.
LEGACY_LOG_PATTERN = re.compile(r"legacy|2Theta|log[ _]name", re.IGNORECASE)


def test_no_legacy_log_name_fixme():
    """Verify no legacy log-name FIXME remains in file_object.py.

    The legacy path silently accepted a capitalized "2Theta" key written by an
    older reduction. It is now an explicit error, so any reappearance of a FIXME
    naming a legacy log key means that regression is back.

    Scanned by comment *block* rather than by line. A FIXME is a comment, so
    `ast` cannot see it and a text scan is the only option -- but it has to be a
    scan over the block, because the natural way to write the thing this guards
    against spans two lines::

        # FIXME: this fallback is wrong
        #   it accepts the legacy 2Theta key

    where neither line carries both halves and a line-at-a-time scan sees
    neither.
    """
    # Arrange
    source = inspect.getsource(file_object_module)

    # Act -- group consecutive comment lines, then test each group as a whole
    offenders = []
    block: list[str] = []
    for line in source.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("#"):
            block.append(stripped)
            continue
        if block:
            text = " ".join(block)
            if "FIXME" in text and LEGACY_LOG_PATTERN.search(text):
                offenders.append(text)
            block = []

    # Assert
    assert not offenders, f"legacy log-name FIXME(s) reintroduced: {offenders}"


def test_no_capitalized_legacy_two_theta_key_in_code():
    """Verify the capitalized legacy "2Theta" key appears in no executable code.

    Narrower than "reads two-theta through `HidraConstants`", which is what this
    used to claim and does not check. The canonical value is **lowercase** --
    `HidraConstants.TWO_THETA == "2theta"` -- and a literal of it does live in
    this module, at `DiffractionUnit.TwoTheta = "2theta"`. That one is a unit
    label rather than the HDF5 log key, and the key's nine real uses all go
    through the constant, so the enum member is left alone and the claim is
    scoped to the defect actually being guarded: the *capitalized* spelling an
    older reduction wrote.

    Docstrings are excluded deliberately: the RuntimeError that *replaced* the
    legacy fallback names the old "2Theta" key in its own documentation, which is
    a description of the fix, not a reoccurrence of the defect. A plain text scan
    cannot tell those apart -- walking the AST can.
    """
    # Arrange
    source = inspect.getsource(file_object_module)
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    # Act: string constants in executable code, not documentation
    literal_keys = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == "2Theta"
        and id(node) not in docstrings
    ]

    # Assert
    assert not literal_keys, f"capitalized legacy two-theta key present in code: {literal_keys}"
