"""
Pydantic models holding each tab's inputs (no Qt).

Each tab has one model. Its fields carry the label (``title``), help text (``description``) and bounds that
:class:`~pyrs.interface.pole_figures.widgets.model_form.ModelForm` turns into input widgets. The presenter reads
the validated model and hands core objects (grids, beams, options) to ``core``.
"""
