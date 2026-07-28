PyRS follows all the suggestions of https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/.

The PyRS documentation is written in [Markdown](https://commonmark.org/) using the
[MyST](https://myst-parser.readthedocs.io/) syntax extensions and processed using
[Sphinx](https://www.sphinx-doc.org/) with the `sphinx_rtd_theme` theme.

There are two documentation trees:

- `docs/user/source` — the user's guide, built for [Read the Docs](https://readthedocs.org/) (see
  [.readthedocs.yml](../.readthedocs.yml)).
- `docs/developer/source` — the developer's guide, including the auto-generated API reference.

Build either locally with `pixi`:

    pixi run build-docs        # user docs -> docs/_build/user
    pixi run build-dev-docs    # developer docs -> docs/_build/developer
    pixi run docs-serve        # serve the built user docs at http://localhost:8000
    pixi run docs-autobuild    # auto-rebuild and serve the user docs on changes

`api/modules.rst` under `docs/developer/source` is generated automatically by `sphinx-apidoc`
at build time (see `conf.py`) and should not be edited by hand; it remains reStructuredText because
`sphinx-apidoc` does not emit Markdown. Every other page is Markdown (MyST).
