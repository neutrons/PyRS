"""``pyrs-pixel-calibration``: the ring-based HB2B detector calibration from one calibration run.

A command-line wrapper only; the workflow and its options are :mod:`pyrs.calibration.ring_driver` (#1101).

Usage (through pixi, from the repository)::

    pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output HB2B_CAL.json
"""

import sys

import numpy as np

from pyrs.calibration.ring_driver import main

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    sys.exit(main())
