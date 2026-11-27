"""Short text: durations, mount matrices as text and back, CSV labels and file-name parts."""

import pytest

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.text import check_label, file_part, format_duration, format_matrix, parse_matrix

#: the cylinder mount: turned about its long axis vz by phi
CYLINDER = MountConventions(up_axis="+vz", q_axis="+vx")


@pytest.mark.parametrize(("seconds", "want"), [(45, "45 s"), (600, "10 min"), (3 * 3600 + 7 * 60, "3 h 07 min")])
def test_format_duration(seconds: float, want: str) -> None:
    # Arrange
    duration = seconds

    # Act
    text = format_duration(duration)

    # Assert
    assert text == want


def test_format_matrix_writes_rows() -> None:
    # Arrange
    M0 = CYLINDER.M0

    # Act
    text = format_matrix(M0)

    # Assert
    assert text == "0 0 1; 0 -1 0; 1 0 0"


def test_parse_matrix_two_rows_raises() -> None:
    # Arrange
    text = "1 0 0; 0 1 0"

    # Act / Assert
    with pytest.raises(ValueError, match="3 rows"):
        parse_matrix(text)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_parse_matrix_non_finite_raises(bad: str) -> None:
    # Arrange
    text = f"1 0 0; 0 1 0; 0 0 {bad}"

    # Act / Assert
    with pytest.raises(ValueError, match=f"finite numbers.*{bad}"):
        parse_matrix(text)


def test_parse_matrix_blank_is_none() -> None:
    # Arrange
    text = "  "

    # Act
    mat = parse_matrix(text)

    # Assert
    assert mat is None


def test_parse_matrix_reads_format_matrix() -> None:
    # Arrange
    text = format_matrix(CYLINDER.M0)

    # Act
    mat = parse_matrix(text)

    # Assert
    assert mat == tuple(float(x) for x in CYLINDER.M0.ravel())


@pytest.mark.parametrize("label", ["1234", "1234+1235", "222", "IN718 200", "run_1.a-b"])
def test_check_label_allowed_unchanged(label: str) -> None:
    # Arrange
    what = "run"

    # Act
    got = check_label(label, what=what)

    # Assert
    assert got == label


@pytest.mark.parametrize("label", ["=1+1", "@SUM(A1)", "-2+3|cmd", "../../../tmp/evil", "", "1234\n"])
def test_check_label_formula_or_path_raises(label: str) -> None:
    # Arrange
    what = "hkl"

    # Act / Assert
    with pytest.raises(ValueError, match="hkl label"):
        check_label(label, what=what)


@pytest.mark.parametrize(("text", "want"), [("IN718 plate / A", "IN718_plate_A"), ("../../../tmp/evil", "tmp_evil"),
                                            ("222", "222"), ("..", "")])  # fmt: skip
def test_file_part_keeps_letters_and_digits(text: str, want: str) -> None:
    # Arrange
    raw = text

    # Act
    part = file_part(raw)

    # Assert
    assert part == want
