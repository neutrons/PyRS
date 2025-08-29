"""
fit_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `fit` `NXprocess` subgroup:
  this subgroup includes the reduced output data as a 'diffraction_data' `NXdata` group.
"""


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ fit                                    (NXprocess, group)
│   ├─ @date                               (attribute: ISO8601 string)
│   ├─ @program                            (attribute: string)
│   ├─ description                         (NXnote, group)
│   ├─ peakparameters                      (NXparameters, group)
│   └─ diffractogram                       (NXdata, group)
│        ├─ diffractogram                  (dataset)
│        ├─ diffractogram_errors           (dataset)
│        ├─ daxis/xaxis                    (dataset)
│        ├─ @axes                          (attribute: string)
│        └─ @signal                        (attribute: string)
"""

class Fit_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################
    pass
