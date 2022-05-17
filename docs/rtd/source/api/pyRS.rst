pyRS package
============

.. toctree::

:mod:`Event Nexus Conversion` Module
------------------------------------

.. autoclass:: pyrs.core.nexus_conversion.NeXusConvertingApp
.. autofunction:: pyrs.core.nexus_conversion.NeXusConvertingApp.convert
.. autofunction:: pyrs.core.nexus_conversion.NeXusConvertingApp.save


:mod:`Data Reduction Manager` Module
------------------------------------

.. autoclass:: pyrs.core.reduction_manager.HB2BReductionManager
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`Data Reduction` Module
----------------------------

.. autoclass:: pyrs.core.reduce_hb2b_pyrs.PyHB2BReduction
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`Instrument Definition` Module
-----------------------------------

.. autoclass:: pyrs.core.reduce_hb2b_pyrs.ResidualStressInstrument
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`Instrument Calibration` Module
------------------------------------

.. autoclass:: pyrs.calibration.peakfit_calibration.PeakFitCalibration

    :undoc-members:
    :show-inheritance:

:mod:`Peak Fitting Methods` Module
----------------------------------

.. autoclass:: pyrs.peaks.peak_fit_engine.PeakFitEngine
.. autofunction:: pyrs.peaks.peak_fit_engine.PeakFitEngine.fit_multiple_peaks
