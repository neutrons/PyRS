"""
pyrs/utilities/NXstress/_sample.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `sample` `NXsample` subgroup.
"""

import numpy as np
from nexusformat.nexus import NXcollection, NXsample, NXfield

from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import allowed_identifier, CHUNK_SHAPE, FIELD_DTYPE


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
    def init_group(cls, sampleLogs: SampleLogs) -> NXsample:
        """
        Create SAMPLE_DESCRIPTION (NXsample) group following NXstress schema:
          - subrun[nP]: link to the scanpoint axis
          - vx[nP], vy[nP], vz[nP]: sample positions in mm (from SampleLogs, converted via PointList)
          - name: sample descriptive name if present in logs; otherwise 'unknown'
          - chemical_formula: sample formula if present in logs; otherwise 'unknown'
          - [optional fields, only if present in the logs]: 'temperature', 'stress_field'
        """
        # Create SAMPLE_DESCRIPTION as an NXsample
        sd = NXsample()

        # Name of sample (required): try the expected log key; fall back to 'unknown'.
        sd["name"] = NXfield(sampleLogs.get(HidraConstants.SAMPLE_NAME, ("unknown",))[0])

        # Link scanpoints to subruns: subrun[nP] (unitless)
        # SampleLogs.subruns is a SubRuns object; use .raw_copy() to get a NumPy array
        scan_points = sampleLogs.subruns.raw_copy()
        sd["scan_point"] = NXfield(
            scan_points.astype(FIELD_DTYPE.INT_DATA.value), chunks=CHUNK_SHAPE(1), maxshape=(None,), units=""
        )
        N_scan = len(scan_points)

        # 3) Sample positions per scanpoint (mm). Use SampleLogs.get_pointlist().
        # PointList returns vx, vy, vz arrays in millimeters.
        try:
            pl = sampleLogs.get_pointlist()
            vv = (pl.vx, pl.vy, pl.vz)
        except AssertionError as e:
            if "some coordinates do not have finite values" in str(e):
                vv = (np.full((N_scan,), np.nan),) * 3
            else:
                raise
        for axis_name, axis_values in zip(HidraConstants.SAMPLE_COORDINATE_NAMES, vv):
            vs = np.asarray(axis_values, dtype=FIELD_DTYPE.FLOAT_DATA.value)
            if vs.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress required log '{axis_name}' has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {vs.shape[0]}"
                )
            f = NXfield(vs, name=axis_name, units="mm")
            sd[axis_name] = f

        # Optionally, add other NXstress SAMPLE_DESCRIPTION fields if available in logs:
        #   - `HidraConstants.CHEMICAL_FORMULA` (NXCHAR)
        #   - `HidraConstants.TEMPERATURE`[nTemp] (NXTEMPERATURE)
        #   - `HidraConstants.STRESS_FIELD`[nsField] (with `@direction` attr = 'x'|'y'|'z')
        # The lines below are safe no-ops if the corresponding logs are not present.
        sd["chemical_formula"] = NXfield(sampleLogs.get(HidraConstants.CHEMICAL_FORMULA, ("unknown",))[0])

        # Example of temperature if present (stored as numeric array and units carried separately)
        if HidraConstants.TEMPERATURE in sampleLogs:
            tkey = HidraConstants.TEMPERATURE
            tvals = np.asarray(sampleLogs[tkey], dtype=FIELD_DTYPE.FLOAT_DATA.value)
            tf = NXfield(tvals, name="temperature")
            tf.attrs["units"] = sampleLogs.units(tkey) or "K"
            sd["temperature"] = tf

        # Example of stress_field if present (values + direction attribute)
        if HidraConstants.STRESS_FIELD in sampleLogs:
            # TODO: we don't have an example of these entries, so the dimensions may not be correct!
            # -- Assuming:
            #      <stress field> :: (<scan points>, ...)
            #      <stress field direction > :: {'x', 'y', 'z'}: scalar
            #
            sf = np.asarray(sampleLogs[HidraConstants.STRESS_FIELD], dtype=FIELD_DTYPE.FLOAT_DATA.value)
            if sf.shape[0] != N_scan:
                raise RuntimeError(
                    f"NXstress required log '{HidraConstants.STRESS_FIELD}' has unexpected shape.\n"
                    f"  First axis should be <scan point> (== {N_scan}), not {sf.shape[0]}"
                )
            sff = NXfield(sf, name="stress_field")
            # If a direction log exists, attach it; otherwise default to 'x'
            direction_key = HidraConstants.STRESS_FIELD_DIRECTION
            direction = sampleLogs[direction_key] if direction_key in sampleLogs else "x"
            sff.attrs["direction"] = direction
            sd["stress_field"] = sff

        # Retain any additional logs that happen to be present.
        sd["logs"] = NXcollection()
        for key in sampleLogs:
            # convert ':' to '_':
            name = allowed_identifier(key)
            if key not in cls.NXstress_logs:
                sd["logs"][name] = NXfield(
                    sampleLogs[key],
                    # source PV-log name as attribute
                    local_name=key,
                    # 'units' as attribute
                    units=sampleLogs.units(key),
                )

        return sd

    @classmethod
    @validate_call_
    def sampleLogsFromNexus(cls, sample) -> SampleLogs:
        """Read SampleLogs from an NXsample group.

        Parameters
        ----------
        sample : NXsample
            The NXsample group from the HDF5 file

        Returns
        -------
        SampleLogs
            Populated SampleLogs object
        """

        # Read scan_point array
        scan_point = sample["scan_point"].nxdata

        # Initialize SampleLogs and set subruns
        logs = SampleLogs()
        logs.subruns = SubRuns(scan_point)

        # Read vx, vy, vz coordinates (stored at top level of NXsample)
        for coord_name in HidraConstants.SAMPLE_COORDINATE_NAMES:
            if coord_name in sample:
                coord_field = sample[coord_name]
                values = coord_field.nxdata
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
                values = field.nxdata
                logs[original_key, units] = values

        # Read optional scalar fields
        if "name" in sample:
            sample_name = sample["name"].nxdata
            if isinstance(sample_name, (bytes, np.bytes_)):
                sample_name = sample_name.decode("utf-8")
            logs[HidraConstants.SAMPLE_NAME, ""] = np.array([sample_name] * len(scan_point))

        if "chemical_formula" in sample:
            chem_formula = sample["chemical_formula"].nxdata
            if isinstance(chem_formula, (bytes, np.bytes_)):
                chem_formula = chem_formula.decode("utf-8")
            logs[HidraConstants.CHEMICAL_FORMULA, ""] = np.array([chem_formula] * len(scan_point))

        if "temperature" in sample:
            temp_field = sample["temperature"]
            temp_values = temp_field.nxdata
            temp_units = temp_field.attrs.get("units", "K")
            logs[HidraConstants.TEMPERATURE, temp_units] = temp_values

        if "stress_field" in sample:
            stress_field = sample["stress_field"]
            stress_values = stress_field.nxdata
            logs[HidraConstants.STRESS_FIELD, ""] = stress_values
            # Read direction attribute if present
            if "direction" in stress_field.attrs:
                direction = stress_field.attrs["direction"]
                logs[HidraConstants.STRESS_FIELD_DIRECTION, ""] = np.array([direction] * len(scan_point))

        return logs
