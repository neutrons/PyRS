"""Safe file writes: retry only while the file is locked, and no partial file under the final name."""

import errno
import os
from pathlib import Path

import pytest

from pyrs.pole_figures.io import retry
from pyrs.pole_figures.io.retry import retry_on_locked, write_atomic


class Flaky:
    """A write that fails with the given errors first, then succeeds. Counts its calls."""

    def __init__(self, *errors: OSError) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


def test_retry_on_locked_retries_permission_error() -> None:
    # Arrange: locked twice (a Windows sharing violation), then free
    write = Flaky(PermissionError(errno.EACCES, "locked"), PermissionError(errno.EACCES, "locked"))

    # Act
    retry_on_locked(write, retries=5, wait_s=0)

    # Assert
    assert write.calls == 3


def test_retry_on_locked_still_locked_raises() -> None:
    # Arrange
    write = Flaky(*[PermissionError(errno.EACCES, "locked")] * 4)

    # Act / Assert
    with pytest.raises(PermissionError):
        retry_on_locked(write, retries=3, wait_s=0)
    assert write.calls == 3


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(errno.ENOENT, "no folder"), OSError(errno.ENOSPC, "disk full"),
     OSError(errno.EROFS, "read-only file system")],
)  # fmt: skip
def test_retry_on_locked_other_os_error_raises_at_once(error: OSError) -> None:
    # Arrange
    write = Flaky(error)

    # Act / Assert
    with pytest.raises(OSError, match=str(error.strerror)):
        retry_on_locked(write, retries=5, wait_s=0)
    assert write.calls == 1


def test_retry_on_locked_zero_retries_raises() -> None:
    # Arrange
    write = Flaky()

    # Act / Assert
    with pytest.raises(ValueError, match="retries must be at least 1, got 0"):
        retry_on_locked(write, retries=0, wait_s=0)


def test_write_atomic_writes_whole_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "table.csv"

    # Act
    write_atomic(path, lambda tmp: tmp.write_text("a,b", encoding="utf-8"))

    # Assert
    assert path.read_text(encoding="utf-8") == "a,b"
    assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


def test_write_atomic_failed_write_keeps_old_file(tmp_path: Path) -> None:
    # Arrange: an older file, then a write that stops part way
    path = tmp_path / "table.csv"
    path.write_text("old", encoding="utf-8")

    def broken(tmp: Path) -> None:
        tmp.write_text("new, part", encoding="utf-8")
        raise OSError(errno.ENOSPC, "disk full")

    # Act / Assert
    with pytest.raises(OSError, match="disk full"):
        write_atomic(path, broken, wait_s=0)
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


def test_write_atomic_failed_write_leaves_no_new_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "table.csv"

    def broken(tmp: Path) -> None:
        tmp.write_text("part", encoding="utf-8")
        raise RuntimeError("stopped")

    # Act / Assert
    with pytest.raises(RuntimeError, match="stopped"):
        write_atomic(path, broken, wait_s=0)
    assert list(tmp_path.iterdir()) == []


def test_write_atomic_locked_target_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the final file is locked for the first two moves
    path = tmp_path / "figure.png"
    real_replace = os.replace
    calls = {"n": 0}

    def locked_replace(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(errno.EACCES, "sharing violation")
        real_replace(src, dst)

    monkeypatch.setattr(retry.os, "replace", locked_replace)

    # Act
    write_atomic(path, lambda tmp: tmp.write_bytes(b"png"), retries=5, wait_s=0)

    # Assert
    assert calls["n"] == 3 and path.read_bytes() == b"png"
    assert [p.name for p in tmp_path.iterdir()] == ["figure.png"]


def test_write_atomic_missing_folder_raises_at_once(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "missing" / "table.csv"
    calls = {"n": 0}

    def write(tmp: Path) -> None:
        calls["n"] += 1
        tmp.write_text("a", encoding="utf-8")

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        write_atomic(path, write, retries=5, wait_s=0)
    assert calls["n"] == 1


def test_temp_path_same_folder_and_extension() -> None:
    # Arrange
    path = Path("out") / "figure.png"

    # Act
    tmp = retry.temp_path(path)

    # Assert
    assert tmp.parent == path.parent and tmp.suffix == ".png" and tmp.name.startswith(".figure.")
