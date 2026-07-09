"""
Run the pytest suite and force an immediate process exit once it's done.

Mantid/mantidqt's native cleanup code can segfault during Python interpreter
shutdown under a real X server (e.g. CI's xvfb+xcb), well after pytest has
already reported every test result and pytest-cov has already written its
coverage report -- observed only there, not under the offscreen Qt platform
used for local runs. Calling pytest.main() directly (rather than the `pytest`
CLI entry point) guarantees every bit of pytest's own reporting has finished
before we flush and exit, so os._exit() only skips the unrelated, crashing
native teardown that follows -- see docs/ground_truths.md.
"""

import os
import sys

import pytest

if __name__ == "__main__":
    exit_code = pytest.main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
