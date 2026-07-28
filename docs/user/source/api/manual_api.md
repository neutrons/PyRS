<!--
THIS DOCUMENT is partially duplicated at `docs/developer/source/api/modules.rst`, which is automatically generated.
Unfortunately, we can't easily link from this `docs/user/source` tree to the `docs/developer/source` tree, and so this
hand-edited version will be left where it is.
-->

# Python programming API

This chapter describes the programming interface of pyRS, and the implementation of methods for the reduction of
raw neutron event data.

```python
import pyRS
```

The most important class is AzimuthalIntegrator which is an object containing
both the geometry (it inherits from Geometry, another class)
and exposes important methods (functions) like `integrate1d` and `integrate2d`.

```{toctree}
:maxdepth: 3
```

## pyRS package

### Event Nexus Conversion Module

```{autoclass} pyrs.core.nexus_conversion.NeXusConvertingApp
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

```{autofunction} pyrs.core.nexus_conversion.NeXusConvertingApp.convert
:no-index:
```

```{autofunction} pyrs.core.nexus_conversion.NeXusConvertingApp.save
:no-index:
```

### Data Reduction Manager Module

```{autoclass} pyrs.core.reduction_manager.HB2BReductionManager
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

### Data Reduction Module

```{autoclass} pyrs.core.reduce_hb2b_pyrs.PyHB2BReduction
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

### Detector Definition Module

```{autoclass} pyrs.core.instrument_geometry.DENEXDetectorGeometry
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

```{autoclass} pyrs.core.instrument_geometry.DENEXDetectorShift
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

### Instrument Definition Module

```{autoclass} pyrs.core.instrument_geometry.HidraSetup
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

```{autoclass} pyrs.core.reduce_hb2b_pyrs.ResidualStressInstrument
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

### Instrument Calibration Module

```{autoclass} pyrs.calibration.mantid_peakfit_calibration.FitCalibration
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

### Peak Fitting Methods Module

```{autoclass} pyrs.peaks.peak_fit_engine.PeakFitEngine
:no-index:
:members:
:undoc-members:
:show-inheritance:
```

```{autofunction} pyrs.peaks.peak_fit_engine.PeakFitEngine.fit_multiple_peaks
:no-index:
```
