"""
The user guide: open the PDF at the page for a workflow step (Help menu, the ? button on the tab bar, F1).

The guides live in the package (``pyrs.pole_figures/docs``), so they are found the same way on Windows and on
the analysis cluster. There are two: the standalone application, and the tool opened from the PyRS main window (same
steps, a different start section). ``user_guide_pages.json`` (written by ``tools/guide/build_guide.py`` with the PDFs)
maps each guide's section keys to pages. A PDF opened as ``file:///...pdf#page=N`` in a web browser goes to that
page, so a browser is used when one is found (Edge on Windows, Firefox or Chrome on Linux, and the Windows browser
from WSL). Otherwise the PDF opens in the system viewer at page 1. ``open_guide`` reports whether the guide opened, is
missing, or found nothing to open it with, so the GUI can say which.

Example:
    >>> result = open_guide("predict", variant="standalone")  # doctest: +SKIP
    >>> result in set(GuideResult)  # doctest: +SKIP
    True
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Literal, get_args

from qtpy.QtCore import QUrl  # type: ignore[attr-defined]
from qtpy.QtGui import QDesktopServices

GUIDE_DIR = Path(__file__).resolve().parents[1] / "docs"
#: which user guide: the standalone application, or the tool opened from the PyRS main window
GuideVariant = Literal["standalone", "pyrs"]
GUIDE_VARIANTS: tuple[GuideVariant, ...] = get_args(GuideVariant)
#: guide variant -> file stem (PDF and Word file of the same name)
GUIDES: dict[GuideVariant, str] = {
    "standalone": "HIDRA_PathLength_PoleFigs_User_Guide",
    "pyrs": "HIDRA_PathLength_PoleFigs_PyRS_User_Guide",
}
PAGES_JSON = GUIDE_DIR / "user_guide_pages.json"
#: section key of the guide's start (also the page used for a section the page map does not list)
START = "start"


class GuideResult(str, Enum):
    """What :func:`open_guide` did."""

    OPENED = "opened"  # handed to a browser or the system viewer
    MISSING = "missing"  # the PDF is not installed with this copy
    NO_VIEWER = "no viewer"  # nothing was found to open it


def check_variant(variant: str) -> GuideVariant:
    """
    The guide variant, checked.

    Args:
        variant: "standalone" or "pyrs".

    Returns:
        The same variant.

    Raises:
        ValueError: If the variant is not one of GUIDE_VARIANTS.
    """
    for known in GUIDE_VARIANTS:
        if variant == known:
            return known
    raise ValueError(f"guide variant must be one of {GUIDE_VARIANTS}, got {variant!r}")


def guide_pdf(variant: GuideVariant) -> Path:
    """
    The PDF of a guide variant.

    Args:
        variant: "standalone" or "pyrs".

    Returns:
        The PDF path. It may not exist: the guide is not installed with every copy.

    Raises:
        ValueError: If the variant is unknown.
    """
    return GUIDE_DIR / f"{GUIDES[check_variant(variant)]}.pdf"


#: guide section of each tab (keys of user_guide_pages.json)
TAB_SECTIONS = {
    "Sample": "sample",
    "Experiment setup": "setup",
    "Predict": "predict",
    "Table scan": "table_scan",
    "Data and Corrections": "data",
    "MTEX": "mtex",
}
_WINDOWS_BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
_LINUX_BROWSERS = ("firefox", "google-chrome", "chromium", "chromium-browser", "microsoft-edge")


def guide_pages(variant: GuideVariant, *, path: Path = PAGES_JSON) -> dict[str, int]:
    """
    Section key -> first page (1-based) in a guide variant.

    Args:
        variant: Guide variant, "standalone" or "pyrs".
        path: The page map written with the PDFs (default: PAGES_JSON).

    Returns:
        First page by section key. Empty when the map is missing, unreadable or has no entry for the variant.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    pages = data.get(variant, {}) if isinstance(data.get(variant), dict) else {}
    return {str(k): int(v) for k, v in pages.items()}


def page_for(section: str, *, pages: dict[str, int]) -> int:
    """
    The page of a section, 1 for the start or a section the page map does not list.

    Args:
        section: Section key (START for the start).
        pages: Page map of the guide (:func:`guide_pages`).

    Returns:
        The 1-based page.
    """
    return pages.get(section, 1)


def guide_url(section: str, *, variant: GuideVariant) -> str:
    """
    ``file:///.../guide.pdf#page=N`` for the section.

    Args:
        section: Section key (START for the start).
        variant: Guide variant, "standalone" or "pyrs".

    Returns:
        The file URL of the PDF with the page fragment.
    """
    page = page_for(section, pages=guide_pages(variant))
    return f"{guide_pdf(variant).resolve().as_uri()}#page={page}"


def in_wsl() -> bool:
    """True inside Windows Subsystem for Linux, where Windows programs can be started."""
    return sys.platform.startswith("linux") and (
        "WSL_DISTRO_NAME" in os.environ or "microsoft" in platform.release().lower()
    )


def _wsl_path(windows_path: str) -> str:
    """A Windows path such as C:/Program Files/... as seen from WSL (/mnt/c/Program Files/...)."""
    drive, rest = windows_path.split(":", 1)
    return f"/mnt/{drive.lower()}" + rest.replace("\\", "/")


def windows_file_url(windows_path: str, page: int) -> str:
    """
    ``file:///C:/.../guide.pdf#page=N`` (or ``file://server/...`` for a UNC path) for a Windows browser.

    Args:
        windows_path: Absolute Windows path of the PDF.
        page: 1-based page.

    Returns:
        The file URL with the page fragment.

    Example:
        >>> windows_file_url("C:/guides/guide.pdf", 7)
        'file:///C:/guides/guide.pdf#page=7'
    """
    return f"{PureWindowsPath(windows_path).as_uri()}#page={page}"


def browser() -> str | None:
    """A web browser that honors ``#page=N`` on a PDF, or None. Under WSL, the Windows browser."""
    if sys.platform == "win32":
        for p in _WINDOWS_BROWSERS:
            if os.path.isfile(p):
                return p
        return None
    for name in _LINUX_BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    if in_wsl():
        for p in _WINDOWS_BROWSERS:
            if os.path.isfile(_wsl_path(p)):
                return _wsl_path(p)
    return None


def _url_for(exe: str, section: str, variant: GuideVariant) -> str:
    """The guide URL as the browser ``exe`` needs it (a Windows path for a Windows browser started from WSL)."""
    if exe.lower().endswith(".exe") and sys.platform != "win32":
        try:
            win = subprocess.run(["wslpath", "-w", str(guide_pdf(variant).resolve())], capture_output=True,  # noqa: S607
                                 text=True, check=True).stdout.strip()  # fmt: skip
            return windows_file_url(win, page_for(section, pages=guide_pages(variant)))
        except (OSError, subprocess.CalledProcessError):
            pass
    return guide_url(section, variant=variant)


def open_guide(section: str, *, variant: GuideVariant) -> GuideResult:
    """
    Open a guide at the section's page.

    A web browser is tried first, because it honors the page. The system PDF viewer opens the guide at page 1.

    Args:
        section: Section key (START for the start).
        variant: Guide variant, "standalone" or "pyrs".

    Returns:
        OPENED, MISSING (no PDF) or NO_VIEWER (nothing to open it).
    """
    pdf = guide_pdf(variant)
    if not pdf.is_file():
        return GuideResult.MISSING
    exe = browser()
    if exe is not None:
        try:
            subprocess.Popen([exe, _url_for(exe, section, variant)], stdout=subprocess.DEVNULL,  # noqa: S603
                             stderr=subprocess.DEVNULL)  # fmt: skip
            return GuideResult.OPENED
        except OSError:
            pass
    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf)))
    return GuideResult.OPENED if opened else GuideResult.NO_VIEWER
