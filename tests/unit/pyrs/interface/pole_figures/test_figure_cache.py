"""The figure cache of the plot panels: most recently used figures by key, builders called once."""

from matplotlib.figure import Figure

from pyrs.interface.pole_figures.figure_cache import FigureCache


def _counting_builder(built: list[str], name: str) -> Figure:
    built.append(name)
    return Figure()


def test_get_second_time_from_cache() -> None:
    # Arrange
    cache = FigureCache(size=2)
    built: list[str] = []
    first = cache.get("a", lambda: _counting_builder(built, "a"))

    # Act
    again = cache.get("a", lambda: _counting_builder(built, "a"))

    # Assert
    assert again is first and built == ["a"]


def test_get_past_size_drops_least_recent() -> None:
    # Arrange
    cache = FigureCache(size=2)
    built: list[str] = []
    a = cache.get("a", lambda: _counting_builder(built, "a"))
    cache.get("b", lambda: _counting_builder(built, "b"))

    # Act
    cache.get("c", lambda: _counting_builder(built, "c"))  # "a" is the oldest and drops out
    rebuilt = cache.get("a", lambda: _counting_builder(built, "a"))

    # Assert
    assert len(cache) == 2 and rebuilt is not a and built == ["a", "b", "c", "a"]


def test_get_none_from_builder_not_kept() -> None:
    # Arrange
    cache = FigureCache(size=2)
    cache.get("a", Figure)

    # Act
    fig = cache.get("none", lambda: None)

    # Assert
    assert fig is None and len(cache) == 1


def test_clear_drops_every_figure() -> None:
    # Arrange
    cache = FigureCache(size=2)
    cache.get("a", Figure)
    cache.get("b", Figure)

    # Act
    cache.clear()

    # Assert
    assert len(cache) == 0
