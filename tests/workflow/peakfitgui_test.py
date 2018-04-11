# In order to test the peak fit window (GUI)
import pyrs.interface
import pyrs.core


def test_main():
    """
    test main
    """
    fit_window = pyrs.interface.fitpeakwindow(None)
    pyrs_core = pyrs.core.pyrscore()
