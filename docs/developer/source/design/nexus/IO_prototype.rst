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

Schema reference
----------------

The authoritative ``NXstress`` application definition is vendored into this
directory as :download:`NXstress.nxdl.xml <NXstress.nxdl.xml>`, copied verbatim
from the NeXus definitions repository:

* source: ``nexusformat/definitions``, ``applications/NXstress.nxdl.xml``
* definitions release: ``v2026.01`` (the repository's release tag -- not the
  ``version="1.0"`` attribute on the file's own root element, which is the NXDL
  schema-language version and a different thing)
* commit: ``004da96ef29bc6b7529e1f8e5d9415cd000bccb8`` ("Update copytright dates to 2026")
* sha256: ``bee46c0723720bc440570cc8f74be37cea6145b15b7e6f4c3ec00c421144b905``

It is a *snapshot*, not a submodule.  Refresh it by re-copying from an updated
checkout and updating the three provenance values above.

Note that the schema is **not** the validator.  The NeXus-org ``NXstress``
validator lives in its own repository and is still not present here; the
installed ``nexusformat`` 1.0.8 package has no validator capability at all.

Identifier names are constrained by ``validItemName``, defined in ``nxdl.xsd``
of the same repository rather than in the application definition itself.  That
rule is reproduced, with its provenance, beside the code that implements it in
``pyrs/utilities/NXstress/_definitions.py``.

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

#. ``(sx, sy, sz)``: *resolved*.  The doubt recorded here was justified -- the fields were filled from the identically-named logs, and that was indeed incorrect.  The schema defines ``peaks/sx``, ``sy``, ``sz`` as the *sample position in the sample reference frame* (required, dimension ``n_Peaks``), which in PyRS terms is ``PointList.(vx, vy, vz)`` and **not** the stage-position logs that happen to share the spelling.  These fields are now written from the sample coordinates, subrun-subset indexed per ``PeakCollection``, and each carries a ``local_name`` attribute naming the log it actually came from.

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

#. Using ``PointList.(vx, vy, vz)`` as the sample positions: *yes*.  The schema's sample-position fields are ``peaks/(sx, sy, sz)``, documented as the sample position in the sample reference frame, and ``PointList.(vx, vy, vz)`` is what PyRS records for that quantity.  Note the two differ in *indexing*: ``peaks/*`` is per peak-row (``n_Peaks``) while the ``vx``/``vy``/``vz`` fields written here are per scan-point (``nP``).  The latter are a PyRS extension -- the schema does not define them -- retained because the reader round-trips ``SampleLogs`` through them.

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


Multiple workspaces in one ``NXentry``
--------------------------------------

The prototype wrote one ``HidraWorkspace`` per entry.  It now writes *N* of
them into a single entry, which is what a stress/strain measurement needs: the
three strain directions belong to one experiment, not three.

.. code-block:: python

   with NXstress(path, "w") as nxs:
       nxs.write(wss, peakss)      # wss: list[HidraWorkspace]
                                   # peakss: list[list[PeakCollection]]

``peakss`` carries **one list per workspace**, positionally parallel to ``wss``.
A workspace that has no peak fits contributes an empty *inner* list, so a single
workspace with no fits is ``write([ws], [[]])`` -- not ``write([ws], [])``,
which is a different (and wrong) shape.

Reading is the inverse, and returns the same pair:

.. code-block:: python

   with NXstress(path, "r") as nxs:
       wss, peakss = nxs.read(entry_number=1)

How the *N* workspaces are told apart on read is a configuration decision, since
nothing in the schema distinguishes them:

* ``nxstress.discriminator_fields`` names sample logs (or ``HidraWorkspace``
  properties) whose values differ between the workspaces -- ``direction``, for
  the stress/strain case.  Each becomes an extra column on ``peaks``, and
  ``read`` groups the rows by the tuple of their values.  A PV-style log name
  such as ``BL7:CS:Sample:Dir`` is encoded reversibly for the on-disk column and
  recovered exactly on read, by the same ``validItemName`` rule described above.
* ``nxstress.merge_workspaces`` is consulted **only** when
  ``discriminator_fields`` is empty.  ``false`` rejects a multi-workspace write
  outright rather than writing a file that cannot be split again; ``true`` says
  the merge is intentional and indistinguishable, and ``read`` then returns one
  merged workspace rather than *N*.

Two data *families* behave differently under this, and the distinction runs
through the whole implementation:

* the **scan-point family** -- ``input_data/detector_counts``, the
  ``SAMPLE_DESCRIPTION`` per-scan-point fields, ``monochromator/wavelength``,
  the diffractograms and the entry-level times -- is plain concatenation along
  ``scan_point``, in workspace order;
* the **peak-index family** -- ``peaks`` and the ``FIT`` parameter groups -- is
  ordered by the compound key ``(discriminators, phase, h, k, l, mask)``, with
  the discriminator values as its most slowly varying coordinates.  Note the key
  does *not* include ``scan_point``: one key identifies one ``PeakCollection``,
  and ``scan_point`` orders the rows *within* that key's run.  Putting the
  discriminators first is what makes each workspace's rows one contiguous block,
  which is what makes the read-side split a grouping rather than new indexing
  machinery.

Issues found:

#. The format guarantees **two** things about row order, and *not* a global
   lexicographic sort: each compound key occupies one contiguous run, and
   ``scan_point`` increases within a run.  A file may therefore be "locally
   sorted, globally segmented".  This was relaxed deliberately, so that append
   could be a plain tail-append; the reader never required more.

#. **A discriminator field that resolves to nothing is a configuration error,
   not a missing value**, and is reported as one.  A field named in the
   configuration but absent from the workspace cannot be defaulted -- the whole
   point of the column is to tell workspaces apart.

Multiple ``NXentry`` groups in one file
----------------------------------------

A ``.nxs`` file may hold several entries, and this is the *other* way to put
more data into one file -- the alternative to appending, not a part of it.  The
two answer different questions:

* **a new ``NXentry``** for a distinct data-reduction or sample *condition*.
  The entries are independent: nothing is shared or aligned between them.
* **appending to an existing entry** (next section) for more data belonging to
  the *same* condition.
* and, within one entry, the compound peak index for finer variation --
  masks, phases, directions.  Multi-direction stress data is **not** multiple
  entries; it is one entry with ``direction`` as a discriminator field.

.. code-block:: python

   with NXstress(path, "w") as nxs:            # new file
       nxs.write(wss, peakss)                  # -> entry 1

   with NXstress(path, "a", entry_number=2) as nxs:   # existing file
       nxs.write(other_wss, other_peakss)             # -> a second entry

   with NXstress(path, "r") as nxs:
       nxs.entry_numbers()                     # -> (1, 2)
       wss, peakss = nxs.read(entry_number=2)

On disk the first entry is named ``entry`` and later ones ``entry_2``,
``entry_3``, ... -- the number is a suffix, and entry 1 carries none.
``entry_numbers()`` derives its result from the names actually present rather
than from a count, so a file with a gap in its numbering reads correctly.

Issues found:

#. **Omitting ``entry_number`` on an existing file appends; it does not add an
   entry.**  Before append existed, every write created a new auto-numbered
   entry, and that is no longer what an unqualified ``write`` does.  Adding an
   entry to a file that already has one is now an explicit request --
   ``entry_number = max + 1``.  This is the one behavioural change most likely
   to surprise code written against the earlier implementation.

#. Overwriting the *contents* of an existing entry is not implemented, and
   raises rather than doing something partial.

Appending to an existing ``NXentry``
-------------------------------------

An entry already on disk can be *grown* with more data, rather than every write
creating a fresh entry.  Use this for more of the *same* condition's data; for
a distinct condition, add an entry instead (previous section).

.. code-block:: python

   with NXstress(path, "a", entry_number=2) as nxs:   # or per call:
       nxs.write(wss, peakss, entry_number=2)

   with NXstress(path, "r") as nxs:
       nxs.entry_numbers()        # -> (1, 2, ...), 1-based, ascending

``entry_numbers()`` is a method rather than a property, and must be called
inside the ``with`` block: it reads the open file, and a property that silently
returned nothing outside the context manager would be worse than one that
raises.

**Whether a call appends or writes afresh is decided by the targeted entry, not
by the file mode:**

.. list-table::
   :header-rows: 1

   * - ``entry_number``
     - targets
     - effect
   * - omitted
     - the highest-numbered existing entry
     - **append**
   * - an existing entry
     - that entry
     - **append**
   * - no existing entry, ``<= max + 1``
     - a new entry there
     - fresh write
   * - omitted, file has no entries
     - entry 1
     - fresh write

An ``entry_number`` beyond ``max + 1`` raises ``ValueError`` naming the next
free number, rather than leaving a gap.

Append is **tail-append only**.  The incoming batch is sorted among itself and
added after each dataset's current end; nothing already on disk is read back,
reordered or rewritten.  The three outcomes:

* a compound key **not** already in the entry -- appended;
* a compound key already present, with new scan points -- ``NotImplementedError``.
  The entry is untouched and the instance stays usable;
* a compound key already present, sharing a scan point -- ``RuntimeError``.  The
  entry is untouched and the instance is **invalidated**: a later ``write`` on
  it raises.

Issues found:

#. **Every check runs before the first resize.**  This is not tidiness: an
   append that failed part-way would leave some datasets grown and others not,
   which no reader could detect.  A refused append leaves the file
   byte-for-byte unchanged, and that is asserted by digest rather than by
   ``pytest.raises`` -- an exception reaching the caller says nothing about what
   was written before it was raised.

#. **A file written by a PyRS predating this work cannot be appended to**, and
   says so rather than failing obscurely.  Per-scan-point datasets used to be
   created at their final size; HDF5 cannot extend a contiguous dataset by any
   mechanism, so they are now written with ``maxshape``/``chunks``.  The
   pre-flight names the first dataset it finds that cannot grow.

#. **Extending a key already in the file remains unimplemented**, as does
   overwriting an entry's contents.  Both are real gaps rather than oversights:
   each would mean inserting into the middle of a position-aligned family, which
   is the thing tail-append exists to avoid.


Possible extensions
-------------------

#. A better and more complete treatment of instrument calibration, and the monochromator information.

#. Probable *rework* of the flattened indexing scheme used by the ``PEAKS`` and ``FIT`` groups.  The current scheme works, but seems *messy* to me.  (The main issue seems to be that the ``NXstress`` schema itself may be *implicitly* assuming that only *one* peak has been fitted.)

#. Treatment of masks by *type*.  This should almost certainly be changed so that it is *explicit*.  At present, distinguishing between a *detector* or *solid-angle* mask depends only on array shape.
