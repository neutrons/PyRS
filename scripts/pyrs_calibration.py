#!/usr/bin/python
"""``pyrs-calibration``: FitCalibration of the HB2B detector from a JSON input.

A command-line wrapper only; the workflow and its options are :mod:`pyrs.calibration.peakfit_driver` (#1101).
"""

from pyrs.calibration.peakfit_driver import main

if __name__ == "__main__":
    main()
