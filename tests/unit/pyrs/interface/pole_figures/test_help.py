"""The user-guide helpers: variants, page map, file URLs (also from WSL), and what open_guide reports."""

import json
from pathlib import Path

import pytest

from pyrs.interface.pole_figures import help as gui_help
from pyrs.interface.pole_figures.help import GuideResult


def test_check_variant_unknown_raises() -> None:
    # Arrange
    variant = "web"

    # Act / Assert
    with pytest.raises(ValueError, match="'web'"):
        gui_help.check_variant(variant)


def test_guide_pdf_variants_differ() -> None:
    # Arrange
    variants = gui_help.GUIDE_VARIANTS

    # Act
    pdfs = {gui_help.guide_pdf(v) for v in variants}

    # Assert
    assert variants == ("standalone", "pyrs") and len(pdfs) == 2


def test_guide_pages_reads_one_variant(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "pages.json"
    path.write_text(json.dumps({"pyrs": {"start": 1, "data": "7"}, "standalone": {"start": 2}}), encoding="utf-8")

    # Act
    pages = gui_help.guide_pages("pyrs", path=path)

    # Assert
    assert pages == {"start": 1, "data": 7}


def test_guide_pages_missing_file_empty(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "none.json"

    # Act
    pages = gui_help.guide_pages("pyrs", path=path)

    # Assert
    assert pages == {}


def test_page_for_unknown_section_first_page() -> None:
    # Arrange
    pages = {"start": 1, "data": 9}

    # Act
    found = [gui_help.page_for("data", pages=pages), gui_help.page_for("nonsense", pages=pages)]

    # Assert
    assert found == [9, 1]


def test_windows_file_url_drive_path_has_page() -> None:
    # Arrange
    path = r"C:\Users\someone\Shared Folder\guide.pdf"

    # Act
    url = gui_help.windows_file_url(path, 9)

    # Assert
    assert url == "file:///C:/Users/someone/Shared%20Folder/guide.pdf#page=9"


def test_windows_file_url_wsl_share_has_page() -> None:
    # Arrange
    path = r"\\wsl.localhost\Ubuntu-24.04\home\someone\guide.pdf"

    # Act
    url = gui_help.windows_file_url(path, 3)

    # Assert
    assert url.startswith("file://wsl.localhost/Ubuntu-24.04/home/someone/guide.pdf") and url.endswith("#page=3")


def test_wsl_path_windows_browser() -> None:
    # Arrange
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    # Act
    path = gui_help._wsl_path(edge)

    # Assert
    assert path == "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"


def test_open_guide_without_pdf_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    monkeypatch.setattr(gui_help, "guide_pdf", lambda variant: tmp_path / "none.pdf")

    # Act
    result = gui_help.open_guide("data", variant="pyrs")

    # Assert
    assert result == GuideResult.MISSING


def test_open_guide_without_viewer_no_viewer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(gui_help, "guide_pdf", lambda variant: pdf)
    monkeypatch.setattr(gui_help, "browser", lambda: None)
    monkeypatch.setattr(gui_help.QDesktopServices, "openUrl", lambda _url: False)

    # Act
    result = gui_help.open_guide("data", variant="pyrs")

    # Assert
    assert result == GuideResult.NO_VIEWER
