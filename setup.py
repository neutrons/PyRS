import codecs
import sys
import os
import re
import versioneer  # https://github.com/warner/python-versioneer
from shutil import copyfile
from setuptools import setup, find_packages

NAME = "pyrs"
META_PATH = os.path.join("pyrs", "__init__.py")
KEYWORDS = ["class", "attribute", "boilerplate"]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
INSTALL_REQUIRES = []

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    # print (r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta))
    # print (META_FILE)
    # print (re.M)
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    """
    main
    """
    scripts = ['scripts/pyrsplot',
               'tests/guitest/peakfitgui_test.py',
               'tests/guitest/manualreduction_test.py',
               'scripts/convert_nexus_to_hidra.py',
               'scripts/pyrscalibration.py',
               'scripts/reduce_HB2B.py',
               'scripts/create_mask.py',
               'scripts/convert_raw_data.py']


    test_scripts = ['tests/unittest/pyrs_core_test.py',
                    'tests/unittest/reduction_test.py',  # beta version
                    'tests/unittest/fit_peaks_test.py',  # beta version
                    'tests/unittest/utilities_test.py',
                    'tests/unittest/polefigurecal_test.py',
                    'tests/unittest/straincalculationtest.py',
                    'tests/guitest/texturegui_test.py',
                    'tests/guitest/strainstressgui_test.py',
                    'tests/guitest/calibration_gui_test.py',
                    #'tests/unittest/test_reduced_hb2b.py',
                    'tests/unittest/reduction_study.py',
                    #'tests/unittest/instrument_geometry_test.py',
                    'tests/unittest/reduction_study.py',
                    'tests/unittest/compare_reduction_engines_test.py']
    print(test_scripts)
    scripts.extend(test_scripts)
    print(scripts)
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        url=find_meta("url"),
        version=versioneer.get_version(),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        long_description=read("README.rst"),
        packages=find_packages(),
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        # from ours
        package_dir={},  # {"": "src"},
        package_data={'': ['*.ui']},
        scripts=scripts,
        #scripts=["scripts/pyrsplot", "tests/unittest/pyrs_core_test.py", "tests/guitest/peakfitgui_test.py"],
        cmdclass=versioneer.get_cmdclass(),
    )

    print ('Scripts compiled: {0}'.format(scripts))
