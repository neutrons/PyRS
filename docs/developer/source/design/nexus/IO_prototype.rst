.. _IO_prototype:

==================
NeXus IO prototype
==================

.. contents
  :local:

Overview
--------

The objective of the NeXus IO-prototype work was to provide a first-pass implementation of NeXus-compliant output using the existing ``NXstress`` schema.  Both *output*, and the corresponding *input* methods are now implemented.
Primarily, the classes ``HidraProjectFile``, ``HidraWorkspace``, and ``PeakCollection`` are used to obtain information about the reduced data.  Supporting classes such as ``SampleLogs``, and ``InstrumentSetup`` are also used, where information about the instrument and the experiment specifics is required.

A sub-objective was to provide an output format that could include *all* of the data associated with an experiment, such as input-data and any additional normalization spectra.  It should be noted however, that including this information is *optional* with respect to the output format.  (Also, As an alternative, it is quite common to simply specify input-data by noting the *file-names* in appropriate fields.)

The next sections provide a correspondance between the python classes, and sections within the ``NXstress`` schema.  Any place where there's still confusion, or there is simply not enough information to meet the requirements of the schema, will be indicated using bold text.

For purposes of the prototype, the ``nexusformat`` python package is used in the implementation, and that working group's validator has been used for validation of compliance.  With respect to validation, its important to use a validator that allows *overriding* NeXus base-class definitions, which ``NXstress`` does extensively.  In this regard, NeXus International Advisory Committee's (NIAC) C-language validator is an *incomplete* implementation, and gives misleading results.  Also noted were several *bugs* in the implementation of the ``nexusformat`` validator: during validation of role-specified groups (noted as ``UPPERCASE`` in the schema), which allow any desired name to be used for the group.  Unfortunately, in this case the validator actually requires ``UPPERCASE``, and won't allow *custom* names.

Primary ``NXentry`` group
-------------------------

Issues found:

#. **start_time** and **end_time**:  These are specified as lists by scan-point (aka *subrun* number in PyRS).
   We could alternatively use the minimum and maximum over all of the sub-run times to obtain these values.
   The validator has trouble with the placement of *lists* for these fields, but *technically* our use of lists
   should be compliant with `NXstress`, and this is a _defect_ with the validator itself

Single ``PEAKS`` group
----------------------

This group is intended to contain the canonical (or *reference*) peak values.

Issues found:

#. ``PEAKS`` group: only a single PEAKS group is allowed by the ``NXstress`` schema. This meant that in order to include *multiple* ``PeakCollection`` we needed to use a *flattened*-indexing scheme.  Such a scheme is allowed by the ``NXstress``-schema, but it is highly unlikely that any *generic* NeXus application will be able to decode these indices automatically.  **Further, since any data reduction (and associated peak fitting) would normally depend on the mask used, a *mask* field has been added to ``PeakCollection``.**  In order to not overly modify the existing code, this ``mask`` field is *optional*, with a *default* value corresponding to the *default* mask.

#. **Converting from ``PyRS`` <peak tag> format to <phase name> and ``(h, k, l)`` (Miller indices) tags**.  At present we make this conversion automatically using a regular-expression based parser, however this is not an ideal solution.  Here it would be better if these values were specified *explicitly* by PyRS as *separate* fields in the ``PeakCollection``.

#. **It's assumed that ``PeakCollection.d_reference`` provides the required values to include in this section**.

#. **``(sx, sy, sz)`` are included from the logs**, but mostly just because the logs had the same variable names -- **this is probably incorrect**!

#. **``(qx, qy, qz)`` are required by ``NXstress``** (, components of the normalized scattering vector Q in the sample reference frame)**.   These seem to have no correspondance in the current PyRS codebase -- these values are initialized to ``NaN``.

``FIT`` (NXprocess) group
-------------------------

This group contains the fitting results from the selected *peak-profile* / *background-function* combination.  In order to include results from multiple ``PeakCollection``, this group uses the *identical* flattened-indexing scheme as the ``PEAKS`` group.  Note that the ``PEAKS`` group includes all of the field-values which define this index, and those values are not repeated in the ``FIT`` group.

#. The splitting of the ``PeakCollection`` fields between ``FIT`` and ``PEAKS`` subgroups from ``NXstress`` was a bit confusing.  This needs to be examined carefully to determine if it is correct.

#. **Not yet in PyRS but required by the ``NXstress`` schema**: ``FIT/DIFFRACTOGRAM/fit``, ``fit_errors``: these datasets should contain the reconstructed spectrum from the fitted model.  We don't seem to have methods to do this yet, so these are initialized to NaN.


``SAMPLE`` (NXsample) group
---------------------------
This was complicated!  Again the main issue is the *naming* of things in ``NXstress`` vs. the naming in the PyRS codebase

Issues found:

#. **Using ``PointList.(vx, vy, vz)`` as the sample positions**?  Is this correct?

#. **Possible mis-match between per-scan-point logs, and logs which have a single value for the entire experiment**.  This still needs to be checked log-by-log!

#. Where at all possible, *all* of the available logs have been included in an additional ``logs`` (``NXcollection``) subgroup.



``INSTRUMENT`` (NXinstrument) group
-----------------------------------

Issues found:
-------------

#. Mask I/O should be fully implemented, including both detector and solid-angle masks.  By necessity, this treatment assumes that a ``<default>`` mask will always exist, and this mask is *created* at output when necessary.  Detector and solid-angle masks are distinguished by their array *shape*.  **Note that under the current mask-naming scheme used by PyRS, masks must have *distinct* names, regardless of type.**

#. **Calibrated** vs. **uncalibrated** instrument is only partially treated, and this will require some additional work in order to make sure that the treatment is correct.  **In PyRS itself, several bugs were found relating to how calibration is applied:** specifically, there's nothing preventing it from being applied *multiple* times.

#. **Monochromator information is only partially available**.

#. **There is a whole lot of room for adjustments and *corrections* in this section!**


Possible extensions
-------------------

#. A better and more complete treatment of instrument calibration, and the monochromator information.

#. Probable *rework* of the flattened indexing scheme used by the ``PEAKS`` and ``FIT`` groups.  The current scheme works, but seems *messy* to me.  (The main issue seems to be that the ``NXstress`` schema itself may be *implicitly* assuming that only *one* peak has been fitted.)

#. Treatment of masks by *type*.  This should almost certainly be changed so that it is *explicit*.  At present, distinguishing between a *detector* or *solid-angle* mask depends only on array shape.
