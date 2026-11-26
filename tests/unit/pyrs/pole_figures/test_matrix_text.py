"""Mount matrices as text and back."""

import pytest

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.io.matrix_text import format_matrix, parse_matrix

#: the cylinder mount: turned about its long axis vz by phi
CYLINDER = MountConventions(up_axis="+vz", q_axis="+vx")


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
