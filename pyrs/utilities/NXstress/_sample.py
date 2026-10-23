"""
pyrs/utilities/NXstress/_sample.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `sample` `NXsample` subgroup.
"""

import numpy as np
from nexusformat.nexus import NXcollection, NXsample, NXfield, NXtransformations

from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    allowed_identifier,
    appendable_dtype,
    FIELD_DTYPE,
    GROUP_NAME,
    growable,
    row_aligned_fields,
    tail_append,
)


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ sample                                 (NXsample, group)
│   ├─ name                                (dataset)
│   ├─ chemical_formula (optional)         (dataset)
│   ├─ temperature (optional)              (dataset)
│   ├─ stress_field (optional)             (dataset)
│   └─ gauge_volume (optional)             (NXparameters, group)
"""


class _Sample:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    #: The sample-frame transformation chain, written under
    #: `SAMPLE_DESCRIPTION/transformations` and entered through the group's
    #: `depends_on`, as `(nexus name, source log, axis vector, type, units)` in
    #: **write order**. Traversal starts at the last entry and follows each
    #: field's own `depends_on` back to ".", so write order read left to right is
    #: the composition order.
    #:
    #: **The rotation axes and their order were recovered by decomposition, not
    #: chosen.** `PoleFigureCalculator.rotate_project_q` composes omega/chi/phi
    #: as the closed-form pole-figure projection of Bob He, *Two-Dimensional
    #: X-ray Diffraction*, ch. 8.3 -- sines and cosines, no matrix -- so the
    #: convention appears nowhere in the source. It is recoverable all the same:
    #: driving that function through its public signature with the probe vector
    #: aimed along each basis vector reads off the matrix it implements, and
    #: exactly one composition reproduces it. See
    #: `plans/NXstress-prod/probes/a5_sample_rotation_axes.py`, whose output is
    #: pasted into `open-questions/04-nxstress-internal-cleanup.md` at Q5:
    #:
    #:     sample -> lab  ==  M0 @ R_omega(0,-1,0) @ R_chi(1,0,0) @ R_phi(0,0,-1)
    #:     worst elementwise error over 200 random triples: 5.818e-14
    #:
    #: `M0` is `frame_offset` below: at `omega = chi = phi = 0` the projection
    #: does **not** reduce to the identity but to `diag(1, -1, -1)`, a constant
    #: 180-degree rotation about `+x` relating the two frames. It is part of the
    #: definition, not an artefact -- omitting it would assert the sample frame
    #: is the lab frame at zero stage angles, which is false. It is also why a
    #: first attempt at this decomposition found no match at all and concluded,
    #: wrongly, that the convention was unrecoverable: a search over stage
    #: rotations alone cannot match a composition carrying a constant factor.
    #:
    #: NXtransformations describes **sample -> lab**, so what is written is the
    #: transpose of the lab -> sample matrix the projection applies. The detector
    #: side composes `Rx @ Ry @ Rz` about `+x/+y/+z`
    #: (`reduce_hb2b_pyrs.py::generate_rotation_matrix`) -- see
    #: `docs/ground_truths.md`, "PyRS's sample stage convention is recoverable
    #: from `rotate_project_q`, offset included", for the general statement and
    #: the warning that an underdetermined decomposition may mean the candidate
    #: space is too small rather than the referent ambiguous -- confirming the lab
    #: frame is right-handed with conventional axes -- so the negative sample
    #: axes here are a real feature of the stage, not a sign error.
    TRANSFORMATION_CHAIN = (
        ("translation_x", "vx", (1.0, 0.0, 0.0), "translation", "mm"),
        ("translation_y", "vy", (0.0, 1.0, 0.0), "translation", "mm"),
        ("translation_z", "vz", (0.0, 0.0, 1.0), "translation", "mm"),
        ("frame_offset", None, (1.0, 0.0, 0.0), "rotation", "degrees"),
        ("omega", "omega", (0.0, -1.0, 0.0), "rotation", "degrees"),
        ("chi", "chi", (1.0, 0.0, 0.0), "rotation", "degrees"),
        ("phi", "phi", (0.0, 0.0, -1.0), "rotation", "degrees"),
    )

    #: `frame_offset`'s constant value -- `M0`, above.
    FRAME_OFFSET_DEGREES = 180.0

    #: Logs the rotation sub-chain is built from. Every real HB2B file carries
    #: them; the synthetic fixtures need not. Written all together or not at
    #: all, on the same rule as `OPTIONAL_SCAN_POINT_FIELDS`, because a rotation
    #: present for some scan points and not others would describe a frame that
    #: changes meaning part-way through the entry.
    #:
    #: `frame_offset` is gated with them rather than written unconditionally.
    #: It is constant and so always *computable*, but it only describes the
    #: sample frame in conjunction with the stage angles: writing it alone would
    #: assert the stage was at zero, which an entry without the angle logs gives
    #: no grounds for.
    ROTATION_SOURCE_LOGS = ("omega", "chi", "phi")

    TRANSFORMATIONS_GROUP = "transformations"

    # Log keys included in the NXstress schema.
    NXstress_logs = {
        HidraConstants.SAMPLE_NAME,
        *HidraConstants.SAMPLE_COORDINATE_NAMES,
        HidraConstants.CHEMICAL_FORMULA,
        HidraConstants.TEMPERATURE,
        HidraConstants.STRESS_FIELD,
        HidraConstants.STRESS_FIELD_DIRECTION,
    }

    @classmethod
    def _scalar(cls, logss: list[SampleLogs], key: str, fallback: str) -> str:
        """One entry-wide value drawn from N inputs, which must agree.

        `name`, `chemical_formula` and the `stress_field` direction describe the
        sample, and the NXstress schema allows exactly one of each per entry.
        Writing one input's value while silently discarding a different one
        would misdescribe the file, so a disagreement raises.

        **Every** value is compared, not the first of each input. These are held
        as per-scan-point logs but describe the entry, so a log that varies
        *within* one input is as wrong as two inputs disagreeing -- and taking
        `[0]` would quietly keep one of them.

        Args:
            logss: Sample logs of every input workspace.
            key: Log key holding the entry-wide value.
            fallback: Value to use when no input carries the log.

        Returns:
            The agreed value, as `str`. Normalised rather than returned raw: the
            comparison above already goes through `to_text`, so returning the
            unnormalised value would write the one spelling the check did not
            look at.

        Raises:
            RuntimeError: If any two values disagree.
        """
        values = [v for logs in logss for v in np.atleast_1d(logs.get(key, (fallback,)))]
        distinct = {to_text(v) for v in values}
        if len(distinct) > 1:
            raise RuntimeError(
                f"NXstress._sample: input workspaces disagree on '{key}': {sorted(distinct)}.\n"
                "  A single NXentry describes one sample; write them to separate entries."
            )
        return distinct.pop() if distinct else fallback

    @classmethod
    def _concatenated_pointlist(cls, logss: list[SampleLogs], counts: list[int]) -> tuple:
        """Sample positions for every input, concatenated in workspace order.

        Taken per input rather than from a merged `SampleLogs`, because the
        concatenated scan-point axis need not be strictly increasing and so
        cannot be held by a `SubRuns` at all -- see
        `plans/NXstress-prod/probes/a5_subruns_nonmonotonic.py`.
        """
        per_axis: list[list[np.ndarray]] = [[], [], []]
        for logs, n_scan in zip(logss, counts):
            if cls._coordinates_are_finite(logs):
                pl = logs.get_pointlist()
                vv = (pl.vx, pl.vy, pl.vz)
            else:
                vv = (np.full((n_scan,), np.nan),) * 3
            for axis, values in zip(per_axis, vv):
                axis.append(np.asarray(values))
        return tuple(np.concatenate(axis) for axis in per_axis)

    @classmethod
    def _coordinates_are_finite(cls, logs: SampleLogs) -> bool:
        """Whether every sample coordinate this `SampleLogs` carries is finite.

        Asked up front rather than discovered by catching the failure. `PointList`
        enforces finiteness with a bare `assert`, so the only way to detect it
        through `get_pointlist` was to catch `AssertionError` and **match its
        message text** -- which stops matching the moment that message is
        reworded, turning a handled case into an unhandled failure mid-write.
        `PointList`'s other two assertions still propagate, which is correct:
        they report a malformed coordinate set, not an absent one.

        A *missing* coordinate log is deliberately not covered here. That raises
        `ValueError` from `get_pointlist`, naming which log is absent, and must
        keep doing so -- NaN-filling it would hide a workspace with no positions
        at all.

        Args:
            logs: One input workspace's sample logs.

        Returns:
            True when every coordinate log present holds only finite values.
        """
        return all(
            bool(np.all(np.isfinite(np.asarray(logs[name], dtype=float))))
            for name in HidraConstants.SAMPLE_COORDINATE_NAMES
            if name in logs
        )

    @classmethod
    def init_group(cls, logss: list[SampleLogs], data: NXsample | None = None) -> NXsample:
        """
        Create SAMPLE_DESCRIPTION (NXsample) group following NXstress schema:
          - subrun[nP]: link to the scanpoint axis
          - vx[nP], vy[nP], vz[nP]: sample positions in mm (from SampleLogs, converted via PointList)
          - name: sample descriptive name if present in logs; otherwise 'unknown'
          - chemical_formula: sample formula if present in logs; otherwise 'unknown'
          - [optional fields, only if present in the logs]: 'temperature', 'stress_field'

        Accepts the sample logs of all N input workspaces and concatenates the
        per-scan-point fields in workspace order. Everything here is raw-array
        concatenation: the merged scan-point axis is not required to be
        monotonic, so it is never routed through a `SampleLogs` or `SubRuns`.
        """
        if data is not None:
            return cls._append_group(logss, data)

        # Create SAMPLE_DESCRIPTION as an NXsample
        sd = NXsample()

        # Name of sample (required): try the expected log key; fall back to 'unknown'.
        sd["name"] = NXfield(cls._scalar(logss, HidraConstants.SAMPLE_NAME, "unknown"))

        # Link scanpoints to subruns: subrun[nP] (unitless)
        # SampleLogs.subruns is a SubRuns object; use .raw_copy() to get a NumPy array
        per_workspace = [logs.subruns.raw_copy() for logs in logss]
        counts = [len(points) for points in per_workspace]
        scan_points = np.concatenate(per_workspace) if per_workspace else np.empty((0,), dtype=int)
        sd["scan_point"] = NXfield(scan_points.astype(FIELD_DTYPE.INT_DATA.value), units="", **growable(1))
        N_scan = len(scan_points)

        # 3) Sample positions per scanpoint (mm). Use SampleLogs.get_pointlist().
        # PointList returns vx, vy, vz arrays in millimeters.
        vv = cls._concatenated_pointlist(logss, counts)
        for axis_name, axis_values in zip(HidraConstants.SAMPLE_COORDINATE_NAMES, vv):
            vs = np.asarray(axis_values, dtype=FIELD_DTYPE.FLOAT_DATA.value)
            if vs.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress required log '{axis_name}' has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {vs.shape[0]}"
                )
            f = NXfield(vs, name=axis_name, units="mm", **growable(vs.ndim))
            sd[axis_name] = f

        cls._init_transformations(sd, cls._chain_columns(logss, counts, N_scan))

        # Optionally, add other NXstress SAMPLE_DESCRIPTION fields if available in logs:
        #   - `HidraConstants.CHEMICAL_FORMULA` (NXCHAR)
        #   - `HidraConstants.TEMPERATURE`[nTemp] (NXTEMPERATURE)
        #   - `HidraConstants.STRESS_FIELD`[nsField] (with `@direction` attr = 'x'|'y'|'z')
        # The lines below are safe no-ops if the corresponding logs are not present.
        sd["chemical_formula"] = NXfield(cls._scalar(logss, HidraConstants.CHEMICAL_FORMULA, "unknown"))

        # Example of temperature if present (stored as numeric array and units carried separately)
        # `OPTIONAL_SCAN_POINT_FIELDS` names this field and `stress_field` below; the
        # append path iterates that constant, so a field added to one must be added
        # to the other or it will be written and never grown.
        if cls._present_in_all(logss, HidraConstants.TEMPERATURE):
            tkey = HidraConstants.TEMPERATURE
            tvals = np.concatenate([np.asarray(logs[tkey], dtype=FIELD_DTYPE.FLOAT_DATA.value) for logs in logss])
            tf = NXfield(tvals, name="temperature", **growable(tvals.ndim))
            tf.attrs["units"] = logss[0].units(tkey) or "K"
            sd["temperature"] = tf

        # Example of stress_field if present (values + direction attribute)
        if cls._present_in_all(logss, HidraConstants.STRESS_FIELD):
            # TODO: we don't have an example of these entries, so the dimensions may not be correct!
            # -- Assuming:
            #      <stress field> :: (<scan points>, ...)
            #      <stress field direction > :: {'x', 'y', 'z'}: scalar
            #
            sf = np.concatenate(
                [np.asarray(logs[HidraConstants.STRESS_FIELD], dtype=FIELD_DTYPE.FLOAT_DATA.value) for logs in logss]
            )
            if sf.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress required log '{HidraConstants.STRESS_FIELD}' has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {sf.shape[0]}"
                )
            # Rank comes from the data, not an assumption: `stress_field` is
            # (<scan point>, ...) and the trailing axes are unknown -- see the TODO above.
            sff = NXfield(sf, name="stress_field", **growable(sf.ndim))
            # The direction is entry-wide, so it goes through `_scalar` like `name`
            # and `chemical_formula`: a bare `logss[0][key]` wrote the whole
            # per-scan-point ARRAY into what the schema defines as a scalar
            # attribute, and the read side then broadcast it again -- a 3-point
            # direction round-tripped from shape (3,) to (3, 3) without raising,
            # because `SampleLogs` only checks the first axis. It also let input
            # workspaces disagree silently, where `name` raises.
            direction = cls._scalar(logss, HidraConstants.STRESS_FIELD_DIRECTION, "x")
            sff.attrs["direction"] = direction
            sd["stress_field"] = sff

        # Retain any additional logs that happen to be present.
        sd["logs"] = NXcollection()
        for key in cls._retained_log_keys(logss):
            # convert ':' to '_':
            name = allowed_identifier(key)
            values = cls._writable(np.concatenate([np.asarray(logs[key]) for logs in logss]))
            sd["logs"][name] = NXfield(
                values,
                # source PV-log name as attribute
                local_name=key,
                # 'units' as attribute
                units=logss[0].units(key),
                **growable(values.ndim),
            )

        return sd

    # Per-scan-point fields written only when every input carries the log, as
    # `(log key, on-disk field name, dtype)`. An entry either has one for all its
    # scan points or for none of them. The fresh-write path in `init_group` still
    # spells each one out, because each carries its own attribute rule (`units`
    # for temperature, `direction` for stress_field) that does not fit this
    # tuple; **a field added there must be added here too**, or append will write
    # it on a fresh write and silently never grow it. Pinned by
    # `test_append.py::TestWriterEmitsResizableDatasets`, which would not catch
    # the omission -- only a round trip carrying the log would.
    OPTIONAL_SCAN_POINT_FIELDS = (
        (HidraConstants.TEMPERATURE, "temperature", FIELD_DTYPE.FLOAT_DATA.value),
        (HidraConstants.STRESS_FIELD, "stress_field", FIELD_DTYPE.FLOAT_DATA.value),
    )

    @classmethod
    def _retained_log_columns(cls, logss: list[SampleLogs]) -> dict:
        """On-disk column name -> source log key, for the logs this batch would write.

        Keyed by the encoded name rather than the raw PV key, because that is what
        the file indexes by and what a comparison against an existing group has to
        match.
        """
        return {allowed_identifier(key): key for key in cls._retained_log_keys(logss)}

    @classmethod
    def _chain_columns(cls, logss: list[SampleLogs], counts: list[int], N_scan: int) -> list[tuple]:
        """The chain links this batch supplies, with their per-scan-point values.

        One source for both the fresh-write and the append path, so the two can
        never disagree about which links exist or what they carry. Translations
        are read through the same `_concatenated_pointlist(logss, counts)` call
        the coordinate loop makes, with the same arguments, so `translation_x`
        and `vx` are the same array by construction rather than by inspection.

        An optional rotation absent from every input is skipped; present in some
        and not others raises, via `_present_in_all`.

        Args:
            logss: Sample logs of the workspaces being written, in workspace order.
            counts: Scan points contributed by each workspace, in the same order.
            N_scan: Total scan points, against which each column is checked.

        Returns:
            `(nexus name, source log, values, axis vector, transformation type,
            units)` in write order, holding only the links this batch supplies.

        Raises:
            RuntimeError: If a column's first axis is not the scan-point axis.
        """
        translations = dict(zip(HidraConstants.SAMPLE_COORDINATE_NAMES, cls._concatenated_pointlist(logss, counts)))
        rotations = cls._rotations_available(logss)

        columns = []
        for name, source, vector, transformation_type, units in cls.TRANSFORMATION_CHAIN:
            if transformation_type == "rotation" and not rotations:
                continue

            if source is None:
                values = np.full(N_scan, cls.FRAME_OFFSET_DEGREES, dtype=FIELD_DTYPE.FLOAT_DATA.value)
            elif source in translations:
                values = np.asarray(translations[source], dtype=FIELD_DTYPE.FLOAT_DATA.value)
            else:
                values = np.concatenate(
                    [np.asarray(logs[source], dtype=FIELD_DTYPE.FLOAT_DATA.value) for logs in logss]
                )

            if values.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress transformation '{name}' (from log '{source}') has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {values.shape[0]}"
                )
            columns.append((name, source, values, vector, transformation_type, units))
        return columns

    @classmethod
    def _rotations_available(cls, logss: list[SampleLogs]) -> bool:
        """Whether every input carries every stage angle the rotation sub-chain needs.

        All three or none: the three compose into one frame, so two of them
        describe nothing. `_present_in_all` raises if a single angle is present
        in some inputs and not others, which is the misalignment case; this adds
        the rule that the three must agree with each other too.
        """
        present = [cls._present_in_all(logss, key) for key in cls.ROTATION_SOURCE_LOGS]
        if any(present) and not all(present):
            supplied = [key for key, ok in zip(cls.ROTATION_SOURCE_LOGS, present) if ok]
            raise RuntimeError(
                f"NXstress._sample: the sample rotation chain needs all of "
                f"{list(cls.ROTATION_SOURCE_LOGS)}, but only {supplied} are present.\n"
                "  Two of the three angles describe no frame; supply all of them, or none."
            )
        return all(present)

    @classmethod
    def _init_transformations(cls, sd: NXsample, columns: list[tuple]) -> None:
        """Write the sample-frame transformation chain and the group's entry point.

        NXstress defines `peaks/sx,sy,sz` as the sample position *in the sample
        reference frame*, and defines that frame by the sample transformations.
        Without this group the positions are numerically present and semantically
        unanchored -- correct for a PyRS-to-PyRS round trip, which shares the
        convention implicitly, and uninterpretable for the external consumer the
        format exists to serve.

        Each link carries the per-scan-point values of its axis, so the chain is
        one transformation per scan point rather than a static placement. The
        values duplicate logs that stay where readers already expect them --
        `vx`/`vy`/`vz` and, for the rotations, `logs/{omega,chi,phi}`. The
        duplication is a few floats per scan point and buys a group a NeXus
        reader can traverse without knowing PyRS; keeping the raw logs untouched
        means the interpretation this group applies is additive, and a consumer
        who disputes the axes can still read the angles.

        `depends_on` names the **last** link, because NeXus enters a chain there
        and follows each field's own `depends_on` back to ".". Naming the first --
        whose `depends_on` is "." -- terminates traversal immediately and leaves
        the rest written but unreachable. That is exactly the defect Follow-up 7
        F7.5 found on the detector side, and the reason this is written as a loop
        ending in `sd["depends_on"] = depends`: the entry point cannot be set to
        anything but the last link without editing the loop itself.

        Args:
            sd: The `NXsample` group being built; its coordinate fields must
                already be written.
            columns: `_chain_columns` output for this batch.
        """
        transformations = NXtransformations()

        depends = "."
        for name, source, values, vector, transformation_type, units in columns:
            field = NXfield(values, units=units, **growable(1))
            field.attrs["transformation_type"] = transformation_type
            field.attrs["vector"] = np.asarray(vector, dtype=np.float64)
            field.attrs["depends_on"] = depends
            # The log these values come from, as `_sample.py` records provenance
            # everywhere else: the NeXus name is `translation_x`, the PyRS one `vx`.
            if source is not None:
                field.attrs["local_name"] = source
            transformations[name] = field
            depends = f"./{cls.TRANSFORMATIONS_GROUP}/{name}"

        sd[cls.TRANSFORMATIONS_GROUP] = transformations
        sd["depends_on"] = depends

    @classmethod
    def _append_transformations(cls, sd: NXsample, columns: list[tuple]) -> None:
        """Grow each chain link by this batch's rows.

        `depends_on` and every `vector` are left exactly as the first write set
        them: an append adds scan points to an existing frame, it does not
        redefine one. `validateAppend` has already established that the links on
        disk are the links this batch supplies.
        """
        if cls.TRANSFORMATIONS_GROUP not in sd:
            # An entry written before this group existed. Appending to it would
            # leave a chain shorter than the axes it describes, which is worse
            # than the absence -- `NXstress._validateAppendableShapes` refuses
            # such an entry up front, so this is a backstop, not a path.
            raise RuntimeError(
                "NXstress._sample: the target entry has no `transformations` group, so its "
                "sample frame is undefined and cannot be extended.\n"
                "  It was written before the sample transformation chain existed; write a new entry."
            )

        transformations = sd[cls.TRANSFORMATIONS_GROUP]
        for name, _source, values, *_ in columns:
            tail_append(transformations[name], values)

    @classmethod
    def appendableDatasets(cls, sd: NXsample) -> dict:
        """Every dataset in ``SAMPLE_DESCRIPTION`` that an append grows.

        Declared by the module that writes the group, rather than discovered by
        array length from outside it -- see `_definitions.row_aligned_fields` for
        why the length sweep this replaced could reject a legal append.

        Args:
            sd: The target entry's existing `SAMPLE_DESCRIPTION` group.

        Returns:
            Path -> `NXfield`, for `NXstress._validateAppendableShapes`.
        """
        found = row_aligned_fields(sd, GROUP_NAME.SAMPLE_DESCRIPTION)
        if "logs" in sd:
            found.update(row_aligned_fields(sd["logs"], f"{GROUP_NAME.SAMPLE_DESCRIPTION}/logs"))
        if cls.TRANSFORMATIONS_GROUP in sd:
            found.update(
                row_aligned_fields(
                    sd[cls.TRANSFORMATIONS_GROUP],
                    f"{GROUP_NAME.SAMPLE_DESCRIPTION}/{cls.TRANSFORMATIONS_GROUP}",
                )
            )
        return found

    @classmethod
    def validateAppend(cls, logss: list[SampleLogs], sd: NXsample) -> None:
        """Check a batch of sample logs against an existing group, without mutating it.

        Every scan point in an entry carries the same set of logs and the same
        optional fields; appending a different set would leave the rows already on
        disk with no value for a new one, and no error to signal it. Separate from
        `_append_group` so that `NXstress._classifyAppend` can run it before
        anything anywhere has been resized.

        Args:
            logss: Sample logs of the workspaces being appended.
            sd: The target entry's existing `NXsample` group.

        Raises:
            RuntimeError: If the sample name or chemical formula differs from the
                entry's, or if the optional-field or retained-log sets differ.
        """
        # `name` and `chemical_formula` describe the whole entry and are never
        # rewritten by an append, so an incoming batch that disagrees would be
        # recorded under the first write's sample -- the disagreement `_scalar`
        # refuses for a single write.
        for key, field_name in (
            (HidraConstants.SAMPLE_NAME, "name"),
            (HidraConstants.CHEMICAL_FORMULA, "chemical_formula"),
        ):
            incoming = cls._scalar(logss, key, "unknown")
            on_disk = to_text(sd[field_name].nxdata) if field_name in sd else "unknown"
            if incoming != on_disk:
                raise RuntimeError(
                    f"NXstress: cannot append -- the incoming workspaces' sample {field_name} "
                    f"'{incoming}' differs from the target entry's '{on_disk}'.\n"
                    "  An NXentry describes one sample; write a new entry for a different one."
                )

        for key, field_name, _ in cls.OPTIONAL_SCAN_POINT_FIELDS:
            present_on_disk = field_name in sd
            present_in_input = cls._present_in_all(logss, key)
            if present_on_disk != present_in_input:
                raise RuntimeError(
                    f"NXstress: cannot append -- the target entry "
                    f"{'has' if present_on_disk else 'does not have'} a '{field_name}' field, "
                    f"while the incoming workspaces "
                    f"{'do' if present_in_input else 'do not'} supply '{key}'.\n"
                    "  Appended data must carry the same optional sample fields as the entry it joins."
                )

        rotations_on_disk = cls.TRANSFORMATIONS_GROUP in sd and any(
            name in sd[cls.TRANSFORMATIONS_GROUP]
            for name, _s, _v, kind, _u in cls.TRANSFORMATION_CHAIN
            if kind == "rotation"
        )
        if rotations_on_disk != cls._rotations_available(logss):
            raise RuntimeError(
                f"NXstress: cannot append -- the target entry's transformation chain "
                f"{'has' if rotations_on_disk else 'has no'} rotation links, while the incoming "
                f"workspaces {'do' if not rotations_on_disk else 'do not'} supply "
                f"{list(cls.ROTATION_SOURCE_LOGS)}.\n"
                "  A chain that gained or lost its rotations part-way through an entry would mean "
                "the sample frame changes definition between scan points."
            )

        incoming = set(cls._retained_log_columns(logss))
        on_disk = set(sd["logs"]) if "logs" in sd else set()
        if incoming != on_disk:
            raise RuntimeError(
                f"NXstress: cannot append -- the incoming workspaces' retained sample logs do not "
                f"match the target entry's.\n"
                f"  On disk:   {sorted(on_disk)}\n"
                f"  Incoming:  {sorted(incoming)}\n"
                "  Every scan point in an entry must carry the same set of logs; appending a "
                "different set would leave the existing rows with no value for a new log."
            )

        mismatched = []
        for name, key in cls._retained_log_columns(logss).items():
            incoming_dtype = cls._writable(np.concatenate([np.asarray(logs[key]) for logs in logss])).dtype
            on_disk_dtype = sd["logs"][name].dtype
            if not appendable_dtype(incoming_dtype, on_disk_dtype):
                mismatched.append(f"    {key}: on disk {on_disk_dtype}, incoming {incoming_dtype}")
        if mismatched:
            raise RuntimeError(
                "NXstress: cannot append -- incoming sample log(s) cannot be stored in the target "
                "entry's columns:\n" + "\n".join(mismatched) + "\n"
                "  A log keeps the type the entry was first written with."
            )

    @classmethod
    def _append_group(cls, logss: list[SampleLogs], sd: NXsample) -> NXsample:
        """Tail-append one batch of sample logs to an existing SAMPLE_DESCRIPTION group.

        Every per-scan-point field grows by the same row count, in lockstep, so the
        group is never left with one field longer than another. The scalars (`name`,
        `chemical_formula`) describe the entry and are left as the first write set
        them.

        Args:
            logss: Sample logs of the workspaces being appended, in workspace order.
            sd: The existing on-disk `NXsample` group.

        Returns:
            The same group, grown.

        Raises:
            RuntimeError: If the retained-log or optional-field set differs from
                the incoming one, if a per-scan-point array has the wrong first
                axis, or if any field was written non-resizably.
            nexusformat.nexus.NeXusError: If a field the append expects is absent
                from the existing group -- a group this writer did not produce.
        """
        # FIRST, before anything is resized. Re-checked here as well as in
        # `NXstress._classifyAppend` because this method is reachable on its own --
        # and a mismatch found after the first `tail_append` would already have
        # grown part of the group, which is the defect this ordering exists to
        # prevent, not merely a tidier place to put the call.
        cls.validateAppend(logss, sd)

        per_workspace = [logs.subruns.raw_copy() for logs in logss]
        counts = [len(points) for points in per_workspace]
        scan_points = np.concatenate(per_workspace) if per_workspace else np.empty((0,), dtype=int)
        N_scan = len(scan_points)

        vv = cls._concatenated_pointlist(logss, counts)
        coordinates = []
        for axis_name, axis_values in zip(HidraConstants.SAMPLE_COORDINATE_NAMES, vv):
            vs = np.asarray(axis_values, dtype=FIELD_DTYPE.FLOAT_DATA.value)
            if vs.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress required log '{axis_name}' has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {vs.shape[0]}"
                )
            coordinates.append((axis_name, vs))

        tail_append(sd["scan_point"], scan_points.astype(FIELD_DTYPE.INT_DATA.value))
        for axis_name, vs in coordinates:
            tail_append(sd[axis_name], vs)

        for key, field_name, dtype in cls.OPTIONAL_SCAN_POINT_FIELDS:
            if cls._present_in_all(logss, key):
                values = np.concatenate([np.asarray(logs[key], dtype=dtype) for logs in logss])
                if values.shape[0] != N_scan:
                    raise RuntimeError(
                        f"NXstress required log '{key}' has unexpected shape.\n"
                        f"  First axis should be <scan point> (== {N_scan}), not {values.shape[0]}"
                    )
                tail_append(sd[field_name], values)

        for name, key in cls._retained_log_columns(logss).items():
            tail_append(
                sd["logs"][name],
                cls._writable(np.concatenate([np.asarray(logs[key]) for logs in logss])),
            )

        cls._append_transformations(sd, cls._chain_columns(logss, counts, N_scan))

        return sd

    @classmethod
    def _writable(cls, values: np.ndarray) -> np.ndarray:
        """Coerce a log array to something HDF5 can actually store.

        Fixed-width *bytes* (`|S`) is coerced for a second, independent reason:
        such a column is sized by the longest value present at write time, and a
        later tail-append of a longer value is **silently truncated** -- h5py
        raises nothing, and `b"a_considerably_longer_filename.h5"` lands as
        `b"a_consid"`. Since append (spec 04c) grows these columns after the
        width is fixed, every string log must be variable-length. Evidence:
        `plans/NXstress-prod/probes/a4_growable_string_fields.py`, claim 5.

        NumPy's fixed-width unicode dtype (`<U`) has no h5py conversion path --
        writing one raises `TypeError: No conversion path for dtype`, which is an
        h5py limitation rather than a NeXus rule. The variable-length UTF-8 dtype
        already used for `phase_name` and `mask` holds the same values, so a
        unicode log is converted rather than rejected.

        Bytes logs (`start_time`, `end_time`, `Filename`) are writable as they
        stand, so the coercion is about width rather than storability for them;
        either way the values read back as `bytes`, which is unchanged. Evidence:
        `plans/NXstress-prod/probes/a4_string_log_dtypes.py`.
        """
        return values.astype(FIELD_DTYPE.STRING.value) if values.dtype.kind in ("U", "S") else values

    @classmethod
    def _present_in_all(cls, logss: list[SampleLogs], key: str) -> bool:
        """Whether an optional per-scan-point log is present in every input.

        Present in some inputs and not others is rejected rather than
        part-filled: the field is written as one array spanning every scan
        point, so a partial log would silently misalign with the scan-point
        axis from the first gap onwards.
        """
        present = [key in logs for logs in logss]
        if any(present) and not all(present):
            raise RuntimeError(
                f"NXstress._sample: log '{key}' is present in some input workspaces and not others "
                f"(present={present}).\n"
                "  Every input must carry it, or none of them."
            )
        return all(present) and bool(present)

    @classmethod
    def _retained_log_keys(cls, logss: list[SampleLogs]) -> list[str]:
        """Non-schema log keys to retain, which every input must share.

        Same reasoning as `_present_in_all`: each retained log is written as a
        single array over the concatenated scan-point axis, so a key missing
        from one input has no values to contribute and would shift every
        later input's values onto the wrong scan points.
        """
        key_sets = [{k for k in logs if k not in cls.NXstress_logs} for logs in logss]
        shared = set.intersection(*key_sets) if key_sets else set()
        divergent = set.union(*key_sets) - shared if key_sets else set()
        if divergent:
            raise RuntimeError(
                f"NXstress._sample: input workspaces carry different sample logs; "
                f"{sorted(divergent)} is absent from at least one.\n"
                "  Every input must carry the same set of logs."
            )
        return sorted(shared)

    @classmethod
    @validate_call_
    def sampleLogsFromNexus(cls, sample, rows=None) -> SampleLogs:
        """Read SampleLogs from an NXsample group.

        Parameters
        ----------
        sample : NXsample
            The NXsample group from the HDF5 file
        rows : np.ndarray, optional
            Boolean mask selecting one input workspace's rows of the
            concatenated scan-point axis. Required when the entry holds more
            than one workspace, and applied *before* any `SubRuns` is built:
            the concatenated axis is not required to be monotonic and
            `SubRuns` rejects a non-monotonic array outright, so splitting
            afterwards is not merely wasteful but impossible. See
            `plans/NXstress-prod/probes/a5_subruns_nonmonotonic.py`.

        Returns
        -------
        SampleLogs
            Populated SampleLogs object
        """

        # Read scan_point array
        scan_point = sample["scan_point"].nxdata
        if rows is not None:
            scan_point = scan_point[rows]

        def selected(values):
            return values[rows] if rows is not None else values

        # Initialize SampleLogs and set subruns
        logs = SampleLogs()
        logs.subruns = SubRuns(scan_point)

        # Read vx, vy, vz coordinates (stored at top level of NXsample)
        for coord_name in HidraConstants.SAMPLE_COORDINATE_NAMES:
            if coord_name in sample:
                coord_field = sample[coord_name]
                values = selected(coord_field.nxdata)
                units = coord_field.attrs.get("units", "mm")
                logs[coord_name, units] = values

        # Read extra logs from the 'logs' NXcollection (if present)
        if "logs" in sample:
            logs_collection = sample["logs"]
            for field_name in logs_collection:
                field = logs_collection[field_name]
                # Get the original PV-log key from local_name attribute
                original_key = field.attrs.get("local_name", field_name)
                units = field.attrs.get("units", "")
                values = selected(field.nxdata)
                logs[original_key, units] = values

        # Read optional scalar fields
        if "name" in sample:
            sample_name = to_text(sample["name"].nxdata)
            logs[HidraConstants.SAMPLE_NAME, ""] = np.array([sample_name] * len(scan_point))

        if "chemical_formula" in sample:
            chem_formula = to_text(sample["chemical_formula"].nxdata)
            logs[HidraConstants.CHEMICAL_FORMULA, ""] = np.array([chem_formula] * len(scan_point))

        if "temperature" in sample:
            temp_field = sample["temperature"]
            temp_values = selected(temp_field.nxdata)
            temp_units = temp_field.attrs.get("units", "K")
            logs[HidraConstants.TEMPERATURE, temp_units] = temp_values

        if "stress_field" in sample:
            stress_field = sample["stress_field"]
            stress_values = selected(stress_field.nxdata)
            logs[HidraConstants.STRESS_FIELD, ""] = stress_values
            # Read direction attribute if present
            if "direction" in stress_field.attrs:
                # Scalar by construction on the write side; `to_text` because an
                # HDF5 attribute comes back as `bytes`, and because a file written
                # before that fix holds the whole array here -- which would
                # broadcast into a 2-D log rather than failing.
                direction = to_text(np.atleast_1d(stress_field.attrs["direction"])[0])
                logs[HidraConstants.STRESS_FIELD_DIRECTION, ""] = np.array([direction] * len(scan_point))

        return logs
