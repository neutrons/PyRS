"""
Figures made from core results, shared by the GUI and the headless runs.

Each figure function takes results from ``core`` and returns a matplotlib ``Figure`` (``beam_gif`` writes a GIF
instead). Nothing here imports Qt. Typical workflow:

1. Predict the correction with ``core.prediction.predict``, or load measured pole figures with ``core.polefig``.
2. Draw them with ``predicted.predicted_figure`` or ``polefigures.pole_figures_figure``.
3. Write the figure with ``save.save_figure``.

Example:
    >>> from pyrs.pole_figures.plots.polefigures import pole_figures_figure
    >>> from pyrs.pole_figures.plots.save import save_figure
    >>> fig = pole_figures_figure(results, view="+vz")  # results: list of core.polefig.PolePeak
    >>> save_figure(fig, "pole_figures.png", dpi=150)
"""
