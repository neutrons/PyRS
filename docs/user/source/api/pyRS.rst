:orphan:


::
  It wasn't really clear where this document was intended to go.  It seems a duplicate of `manual_api.rst`.


pyRS package
============

.. toctree::

:mod:`Event Nexus Conversion` Module
------------------------------------

.. autoclass:: pyrs.core.nexus_conversion.NeXusConvertingApp
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pyrs.core.nexus_conversion.NeXusConvertingApp.convert
   :no-index:

.. autofunction:: pyrs.core.nexus_conversion.NeXusConvertingApp.save
   :no-index:



:mod:`Data Reduction Manager` Module
------------------------------------

.. autoclass:: pyrs.core.reduction_manager.HB2BReductionManager
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

:mod:`Data Reduction` Module
----------------------------

.. autoclass:: pyrs.core.reduce_hb2b_pyrs.PyHB2BReduction
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

:mod:`Instrument Definition` Module
-----------------------------------

.. autoclass:: pyrs.core.reduce_hb2b_pyrs.ResidualStressInstrument
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

:mod:`Instrument Calibration` Module
------------------------------------

.. autoclass:: pyrs.calibration.mantid_peakfit_calibration.FitCalibration
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

:mod:`Peak Fitting Methods` Module
----------------------------------

.. autoclass:: pyrs.peaks.peak_fit_engine.PeakFitEngine
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pyrs.peaks.peak_fit_engine.PeakFitEngine.fit_multiple_peaks
   :no-index:
