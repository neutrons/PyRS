"""Re-export shared fixtures needed by tests under tests/ui/, plus fixtures shared by its viewers.

Window fixtures
---------------
Each test module builds its viewer in a **function-scoped** fixture on
pytest-qt's `qtbot`, which closes the window after the test (`qtbot.addWidget`),
so no test sees what an earlier one did to a window.

They used to be session-scoped, sharing one window across every test in a module,
"to avoid the segmentation fault error that occurs when the window is created
with a function scope". **We cannot reproduce that segfault**: per-test windows
pass every GUI test locally under `QT_QPA_PLATFORM=offscreen`, run repeatedly.
CI runs under `xvfb-run`, which is not available locally, so CI is the remaining
check -- but nothing found so far ties that segfault to window scope. Sharing the
window, meanwhile, made the tests order-dependent: a texture-fitting test failed
only when run after another test that had left fit-parameter plot axes selected.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.NXstress.NXstress import NXstress
from tests.unit.pyrs.utilities.NXstress.conftest import minimal_HidraWorkspace, minimal_PeakCollection  # noqa: F401
from tests.util.peak_collection_helpers import createPeakCollection  # noqa: F401


@pytest.fixture
def unloadable_nxstress_files(tmp_path: Path, minimal_HidraWorkspace: Callable[..., HidraWorkspace]) -> dict:  # noqa: F811
    """`.nxs` files a single-workspace viewer must refuse, keyed by why.

    `garbage` is not HDF5 at all (`NeXusError` on open); `two_entries` is a valid
    NXstress file holding two `NXentry` groups (`UnsupportedNXstressFile`).
    """
    garbage = tmp_path / "garbage.nxs"
    garbage.write_text("not an HDF5 file")

    two_entries = tmp_path / "two_entries.nxs"
    ws = minimal_HidraWorkspace(with_instrument=True)
    with NXstress(two_entries, "w") as nx:
        nx.write([ws], [[]])
        nx.write([ws], [[]], entry_number=2)

    return {"garbage": garbage, "two_entries": two_entries}
