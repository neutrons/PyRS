"""
Presenters: connect a tab's widgets to its model and to core/ and plots/.

Each presenter owns its tab model, updates it from the tab's forms, runs long core jobs in a
:class:`~pyrs.interface.pole_figures.worker.Worker`, and shows the results on the tab. Presenters of later steps
take the earlier ones and listen to their signals (sample and material -> predict -> table scan, data -> MTEX).
"""
