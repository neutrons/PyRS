"""
sample_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `sample` `NXsample` subgroup.
"""

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ sample                                 (NXsample, group)
│   ├─ name                                (dataset)
│   ├─ chemical_formula (optional)         (dataset)
│   ├─ temperature (optional)              (dataset)
│   ├─ stress_field (optional)             (dataset)
│   └─ gauge_volume (optional)             (NXparameters, group)
"""

class Sample_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################
    pass
