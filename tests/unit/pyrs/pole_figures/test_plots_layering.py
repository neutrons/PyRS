"""The plots package imports only core, the leaf modules text and io.retry, and its own helper modules."""

import ast
from pathlib import Path

import pytest

from pyrs.pole_figures import plots

#: plots modules that others in plots may import (no figure module imports another figure module)
HELPERS = {"style", "annotate", "polefig_cells", "polefig_interp", "beam_frame", "gif_worker"}
#: package modules outside core that plots may import: leaf modules without file formats or Qt
LEAVES = {"text", "io.retry"}
MODULES = sorted(p.stem for p in Path(plots.__file__).parent.glob("*.py") if p.stem != "__init__")


def _relative_imports(module: str) -> list[tuple[int, str]]:
    """(level, dotted name) of every relative import in a plots module, including ``from . import x``."""
    tree = ast.parse((Path(plots.__file__).parent / f"{module}.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                out.append((node.level, node.module))
            else:
                out += [(node.level, alias.name) for alias in node.names]
    return out


@pytest.mark.parametrize("module", MODULES)
def test_plots_module_imports_core_leaves_and_helpers(module: str) -> None:
    # Arrange
    allowed_here = HELPERS - {module}

    # Act
    imports = _relative_imports(module)

    # Assert
    bad = [
        f"{'.' * level}{name}"
        for level, name in imports
        if not (
            (level == 1 and name.split(".")[0] in allowed_here)
            or (level == 2 and (name.split(".")[0] == "core" or name in LEAVES))
        )
    ]
    assert bad == []
