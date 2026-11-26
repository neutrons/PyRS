"""Beam-path GIF frame selection and worker count."""

from pyrs.pole_figures.plots.beam_gif import MIN_FRAMES_PARALLEL, frame_indices, gif_workers


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
    workers = gif_workers(n_frames, 1)

    # Assert
    assert workers == 1
