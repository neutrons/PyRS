PyRS follows all the suggestions of https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/.


The PyRS documentation is written in [reStructuredText](http://docutils.sourceforge.net/rst.html)
and processed using [Sphinx](http://sphinx.pocoo.org/). It uses a custom
bootstrap theme for Sphinx and both are required to build the documentation.

To install Sphinx and the bootstrap theme use `easy_install`:

    easy_install -U Sphinx
    easy_install -U sphinx-bootstrap-theme

or `pip`:

    pip install Sphinx
    pip install sphinx-bootstrap-theme

Installing Sphinx may require admin privileges on some environments.
