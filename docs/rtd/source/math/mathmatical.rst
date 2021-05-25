Mathematical Methods
####################

Strain/Stress Analysis
======================

PyRS models Strain and Stress tensor :math:`_{ij}` fields using the following formulation:

*Strain*, aka  unconstrained strain, :math:`\epsilon_{ij}` is measured as the fraction change from a reference state :math:`d_0`.

.. math::

   \epsilon_{ij} = \frac{d_{ij} - d_0}{d_0}


*Residual stress* is determined by measuring stress along 3 orthogonal directions

.. math::

   \sigma_{ij} = \frac{E}{(1 + \nu)}\left[\epsilon_{ij} + \frac{\nu}{1-2\nu}(\epsilon_{11} + \epsilon_{22} + \epsilon_{33})\right]

where:

:math:`\nu`
   Poisson's ratio

:math:`E`
   Young's modulus


Notice that :math:`\epsilon_{ij}` with :math:`i = j` are principal strains.
But not all all three orthogonal strains are equivalent to principal strains.
The off-diagonal strain component, i.e., :math:`\epsilon_{ij}` with :math:`i \neq j` are all set to *zero*.
It is very hard to measure these values in HB2B's setup.

As a result, the stress in PyRS is calculated as follows:

.. math::

   \sigma_{ii} = \frac{E}{(1 + \nu)}\left[\epsilon_{ii} + \frac{\nu}{1-2\nu}(\epsilon_{11} + \epsilon_{22} + \epsilon_{33})\right]

where the second term in the sum is common to all 3 principle directions, :math:`\sigma_{11}`, :math:`\sigma_{22}`, and :math:`\sigma_{33}`.

**in-plane strain**: this is a special case in which the stress in one of the principal directions is zero, :math:`\sigma_{33}=0`. Therefore, the corresponding strain can be determined from the other principal components using Poisson's ratio:

.. math::

   \epsilon_{33} = \frac{\nu}{\nu-1}(\epsilon_{11} + \epsilon_{22})


Resulting in a simplified set of equations for the *in-plane stress* case:

.. math::

   \sigma_{11} &=& \frac{E}{(1 + \nu)}\left[\epsilon_{11} + \frac{\nu (\epsilon_{11} + \epsilon_{22})}{1-\nu}\right] \\[1cm]
   \sigma_{22} &=& \frac{E}{(1 + \nu)}\left[\epsilon_{22} + \frac{\nu (\epsilon_{11} + \epsilon_{22})}{1-\nu}\right] \\[1cm]
   \sigma_{33} &=& 0
