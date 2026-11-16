"""
Loading an NXstress file into a viewer that can hold exactly one workspace.

`PeakFittingModel` and `TextureFittingModel` each hold a single
`HidraWorkspace`, while an NXstress file may hold several `NXentry` groups and
each entry several workspaces -- which is what spec 04b exists to produce and
what every append (spec 04c) creates. The two are not reconcilable by the model,
so the mismatch has to be *reported*, not resolved by taking the first of
whatever was found.

That is what these viewers used to do:

    wss, peakss = nx.read()
    ws, peaks = wss[0], peakss[0]

under a comment asserting "a viewer holds one workspace; an entry written by
this viewer holds one". The second half is a statement about files this viewer
wrote, and a user may open any file at all -- so a three-direction entry loaded
"successfully" with two thirds of its content dropped and nothing said.

`UnsupportedNXstressFile` is a `RuntimeError` so that it lands in the viewers'
existing "unable to load" handling: the user gets a dialog naming exactly what
the file holds, and the application stays where it was. Refusing rather than
loading part of it is deliberate -- a partial load can be fitted and saved back,
and the rest of the file would be gone.
"""

from pathlib import Path

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress.NXstress import NXstress


class UnsupportedNXstressFile(RuntimeError):
    """An NXstress file holds more than a single-workspace viewer can represent."""


def read_single_workspace(path: Path) -> tuple[HidraWorkspace, list[PeakCollection]]:
    """The one workspace an NXstress file holds, or a refusal naming what it holds.

    Args:
        path: The `.nxs` file to read.

    Returns:
        `(workspace, peak_collections)` -- the shape a single-workspace viewer
        expects, and what `NXstress.read` returned before spec 04b.

    Raises:
        UnsupportedNXstressFile: If the file holds more than one `NXentry`, or
            the entry holds more than one workspace. Both are legitimate NXstress
            files; neither can be shown by a viewer that holds one workspace, and
            silently showing part of one is how data gets lost on the next save.
    """
    with NXstress(path, "r") as nx:
        entries = nx.entry_numbers()
        if len(entries) > 1:
            raise UnsupportedNXstressFile(
                f"'{path.name}' holds {len(entries)} NXentry groups (numbers {list(entries)}), and this "
                "viewer can show one.\n"
                "  Nothing was loaded, so no data has been lost. Open a file with a single entry, or "
                "split this one."
            )

        workspaces, peak_collections = nx.read()
        if len(workspaces) > 1:
            raise UnsupportedNXstressFile(
                f"'{path.name}' holds {len(workspaces)} workspaces in one NXentry, and this viewer can "
                "show one.\n"
                "  Nothing was loaded, so no data has been lost. Such a file is written when several "
                "workspaces share an entry -- the strain/stress workflow, or an append."
            )

        # Both refusals sit inside the `with`, although only the first needs the
        # open file. Everything `read` returns is detached -- every read path ends
        # in `.nxdata`, which materialises into numpy -- so the second would work
        # outside it. Keeping it inside anyway: the access pattern is the thing to
        # be consistent about, regardless of which individual statement strictly
        # requires the file. The two refusals are also one decision ("can this
        # viewer show what is in here?"), and splitting them across the block
        # boundary reads as though the second were a different kind of check.
        return workspaces[0], peak_collections[0]
