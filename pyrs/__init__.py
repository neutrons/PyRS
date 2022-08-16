# type: ignore
__description__ = "Reduction software for HB2B"
__url__ = 'https://github.com/neutrons/PyRS'

__author__ = 'W.Zhou'
__email__ = 'zhouw@ornl.gov'

__license__ = 'GNU GENERAL PUBLIC LICENSE'

try:
    from pyrs._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"