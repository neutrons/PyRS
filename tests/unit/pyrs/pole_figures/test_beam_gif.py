"""Beam-path GIF settings, frame selection, worker count and the size check before drawing."""

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pyrs.pole_figures.plots.beam_gif import (
    MAX_GIF_PIXELS,
    MIN_FRAMES_PARALLEL,
    GifSettings,
    frame_indices,
    gif_workers,
    measured_beam_gif,
)
from tests.util.pole_figures_helpers import SLAB_CENTER_MM, slab_mesh, synthetic_pole_figure


def test_frame_indices_every_third() -> None:
    # Arrange
    n, every = 7, 3

    # Act
    order = frame_indices(n, every)

    # Assert
    assert order == [0, 3, 6]


def test_frame_indices_step_below_one_counts_as_one() -> None:
    # Arrange
    n, every = 3, 0

    # Act
    order = frame_indices(n, every)

    # Assert
    assert order == [0, 1, 2]


def test_gif_workers_short_gif_is_serial() -> None:
    # Arrange
    n_frames = MIN_FRAMES_PARALLEL - 1

    # Act
    workers = gif_workers(n_frames, "auto")

    # Assert
    assert workers == 1


def test_gif_workers_explicit_count_kept() -> None:
    # Arrange
    n_frames = 10 * MIN_FRAMES_PARALLEL

    # Act
    workers = gif_workers(n_frames, 3)

    # Assert
    assert workers == 3


@pytest.mark.parametrize(
    ("kw", "word"),
    [({"fps": 0.0}, "fps"), ({"fps": 101.0}, "fps"), ({"fps": float("nan")}, "fps"), ({"dpi": 19.0}, "dpi"),
     ({"dpi": 301.0}, "dpi"), ({"every": 0}, "every"), ({"every": 1.5}, "every"), ({"workers": 0}, "workers"),
     ({"workers": "many"}, "workers"), ({"workers": True}, "workers")],
)  # fmt: skip
def test_gif_settings_out_of_range_raises(kw: dict[str, Any], word: str) -> None:
    # Arrange
    value = next(iter(kw.values()))

    # Act / Assert
    with pytest.raises(ValueError, match=f"GIF {word} .*got {re.escape(repr(value))}"):
        GifSettings(**kw)


@pytest.mark.parametrize("kw", [{"fps": 100.0}, {"dpi": 20.0}, {"dpi": 300.0}, {"every": 1}, {"workers": 1}])
def test_gif_settings_limits_accepted(kw: dict[str, Any]) -> None:
    # Arrange
    expected = next(iter(kw.values()))

    # Act
    settings = GifSettings(**kw)

    # Assert
    assert getattr(settings, next(iter(kw))) == expected


def test_gif_settings_frame_pixels() -> None:
    # Arrange
    settings = GifSettings(dpi=80.0)

    # Act
    pixels = settings.frame_pixels()

    # Assert: the 12.5 x 5.8 in frame
    assert pixels == 1000 * 464


def test_measured_beam_gif_too_many_pixels_raises(tmp_path: Path) -> None:
    # Arrange: a 5 x 5 deg pole figure (1297 points) at dpi 300 is over the limit; checked before any drawing
    peak = synthetic_pole_figure(chi_step=5.0, phi_step=5.0)
    settings = GifSettings(dpi=300.0)
    assert peak.n_points * settings.frame_pixels() > MAX_GIF_PIXELS

    # Act / Assert
    with pytest.raises(ValueError, match="Lower the dpi"):
        measured_beam_gif(tmp_path / "big.gif", slab_mesh(), peak=peak, center=SLAB_CENTER_MM, settings=settings)
    assert not (tmp_path / "big.gif").exists()


def test_measured_beam_gif_no_points_raises(tmp_path: Path) -> None:
    # Arrange: a pole figure with no points
    r = synthetic_pole_figure()
    empty = replace(r, chi=r.chi[:0], phi=r.phi[:0], count_s=r.count_s[:0], rate=r.rate[:0], sigma=r.sigma[:0],
                    gauge=r.gauge.take(slice(0, 0)), h=r.h[:0])  # fmt: skip

    # Act / Assert
    with pytest.raises(ValueError, match="no chi, phi orientations"):
        measured_beam_gif(tmp_path / "empty.gif", slab_mesh(), peak=empty, center=SLAB_CENTER_MM)
    assert not (tmp_path / "empty.gif").exists()
