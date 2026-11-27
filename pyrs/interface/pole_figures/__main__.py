"""
Run the GUI with ``python -m pyrs.interface.pole_figures``.

Example:
    From a shell::

        python -m pyrs.interface.pole_figures
"""

import sys

from .app import main

if __name__ == "__main__":  # worker processes (GIF frames) import this module again: they must not start the GUI
    sys.exit(main())
