"""
tests/scripts/cis_tests/plot_sample_points.py

Smoke-test / "by hand" script for visually inspecting the sample-point
coordinates stored in HiDRA project files.

Loads one ``StrainField`` per project file from ``tests/data`` and plots each
file's (x, y, z) sample positions as a 3-D scatter, so that overlapping and
disjoint scans can be compared at a glance.

Usage
-----
    python tests/scripts/cis_tests/plot_sample_points.py

Edit the file list in the ``__main__`` block below to plot a different set.
"""

import os
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

from pyrs.dataobjects.fields import StrainField

# tests/scripts/cis_tests/ -> tests/data
test_data_dir = str(Path(__file__).resolve().parents[2] / "data")


def plot_sample_points(*files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for file in files:
        strain = StrainField(filename=os.path.join(test_data_dir, file), peak_tag="peak0")
        ax.scatter(strain.x, strain.y, strain.z)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.legend(files)
    plt.show()


if __name__ == "__main__":
    plot_sample_points("HB2B_1327.h5", "HB2B_1328.h5", "HB2B_1331.h5", "HB2B_1332.h5")
