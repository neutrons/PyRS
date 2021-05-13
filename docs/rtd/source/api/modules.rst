Python programming API
======================

This chapter describes the programming interface of pyRS, and the implementation of methods for the reduction of
raw neutron event data.

.. code-block:: python

	import pyRS

The most important class is AzimuthalIntegrator which is an object containing
both the geometry (it inherits from Geometry, another class)
and exposes important methods (functions) like `integrate1d` and `integrate2d.

.. toctree::
   :maxdepth: 2

   pyRS
