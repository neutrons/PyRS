"""
Sample tab inputs: STL, gauge center, mount conventions and a preview orientation.

The sample presenter loads the STL, and the predict and data presenters read :meth:`SampleModel.conventions` and
:meth:`SampleModel.gauge_center`. The validator rejects a sample-to-stage matrix that is not a proper rotation.

Example:
    >>> model = SampleModel(gauge_vx=29.4, gauge_vy=19.75)
    >>> model.gauge_center()
    array([29.4 , 19.75,  0.  ])
    >>> model.conventions().incident_from
    '+vy'
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

from pyrs.pole_figures.core.geometry import MountConventions
from pyrs.pole_figures.core.rotation import check_rotation
from ..mount_matrix import IDENTITY9

GAUGE_FIELDS = ["gauge_vx", "gauge_vy", "gauge_vz"]
MOUNT_FIELDS = ["incident_from", "phi_sense", "chi_tilts_up_axis_toward_Q", "chi_offset_deg"]  # stage conventions
VIEW_FIELDS = ["view_from", "vz_name"]
PREVIEW_FIELDS = ["preview_chi", "preview_phi", "preview_omega", "preview_tth"]


class SampleModel(BaseModel):
    """
    Inputs of the Sample tab.

    The mount defaults are the HB-2B conventions confirmed on the reference measurement: incident beam from +vy,
    +phi clockwise seen from above, +vx up at chi 0, and +chi tilting the upward phi axis toward Q.

    Attributes:
        stl_path: The sample STL file. Its frame is the sample frame (vx, vy, vz, mm).
        gauge_vx: vx of the gauge center = rotation point (mm, STL frame).
        gauge_vy: vy of the gauge center (mm).
        gauge_vz: vz of the gauge center (mm).
        mount_matrix: Sample-to-stage rotation, 9 values row by row. Rows: the phi axis (up), the chi axis (across)
            and Q at chi 0, phi 0. Columns: vx, vy, vz. Identity is PF's mount.
        incident_from: Side the incident beam comes from at chi 0, phi 0: "+vy" or "-vy".
        phi_sense: Sense of +phi seen from above.
        chi_tilts_up_axis_toward_Q: True when +chi tilts the upward phi axis toward Q.
        chi_offset_deg: Cradle chi = logged chi + this offset (deg).
        view_from: Direction the pole figures are seen from: "+vz", "-vz" or "+vx".
        vz_name: Label of +vz in the figures, for example BD.
        preview_chi: Logged chi of the preview orientation (deg).
        preview_phi: phi of the preview orientation (deg).
        preview_omega: omega of the preview orientation (deg).
        preview_tth: 2theta of the preview orientation (deg).
    """

    stl_path: str = Field(default="", title="STL file")

    gauge_vx: float = Field(
        default=0.0, ge=-1000, le=1000, title="vx (mm)", description="Gauge center = rotation point"
    )
    gauge_vy: float = Field(
        default=0.0, ge=-1000, le=1000, title="vy (mm)", description="Gauge center = rotation point"
    )
    gauge_vz: float = Field(
        default=0.0, ge=-1000, le=1000, title="vz (mm)", description="Gauge center = rotation point"
    )

    mount_matrix: tuple[float, ...] = Field(
        default=IDENTITY9, title="Sample-to-stage matrix",
        description="Rows: the phi axis (up), the chi axis (across) and Q at chi 0, phi 0. Columns: vx, vy, vz. "
                    "Each column is where that sample axis points on the stage. Identity = PF's mount",
    )  # fmt: skip
    incident_from: Literal["+vy", "-vy"] = Field(
        default="+vy", title="Incident beam from",
        description="Side of the stage the incident beam comes from at chi 0, phi 0: +vy = the side the chi axis "
                    "(across) points to, which is +vy for the identity mount",
    )  # fmt: skip
    phi_sense: Literal["clockwise_from_above", "counterclockwise_from_above"] = Field(
        default="clockwise_from_above", title="+phi sense", description="Check with the top camera"
    )
    chi_tilts_up_axis_toward_Q: bool = Field(  # noqa: N815 (PF config key)
        default=True, title="+chi tilts phi axis toward Q", description="Sense of chi"
    )
    chi_offset_deg: float = Field(
        default=0.0, ge=-180, le=180, title="chi offset (deg)", description="Cradle chi = logged chi + offset"
    )

    view_from: Literal["+vz", "-vz", "+vx"] = Field(
        default="+vz", title="Pole figures seen from", description="+vz: +vx right, +vy up. -vz: as the 2D maps"
    )
    vz_name: str = Field(default="vz", title="Name of +vz", description="Label in the figures, e.g. BD")

    preview_chi: float = Field(default=0.0, ge=-180, le=180, title="chi (deg, logged)")
    preview_phi: float = Field(default=0.0, ge=-360, le=360, title="phi (deg)")
    preview_omega: float = Field(default=45.0, gt=0, lt=180, title="omega (deg)")
    preview_tth: float = Field(default=90.0, gt=0, lt=180, title="2theta (deg)")

    @model_validator(mode="after")
    def _rotation(self) -> "SampleModel":
        if len(self.mount_matrix) != 9:
            raise ValueError("the sample-to-stage matrix needs 9 values")
        check_rotation([self.mount_matrix[0:3], self.mount_matrix[3:6], self.mount_matrix[6:9]])
        return self

    def conventions(self) -> MountConventions:
        """The mount conventions as core :class:`MountConventions`."""
        return MountConventions(
            phi_sense=self.phi_sense,
            sample_to_instrument=self.mount_matrix,
            chi_tilts_up_axis_toward_Q=self.chi_tilts_up_axis_toward_Q,
            incident_from=self.incident_from,
            chi_offset_deg=self.chi_offset_deg,
        )

    def gauge_center(self) -> npt.NDArray[np.float64]:
        """The gauge center (mm, STL frame) used by the prediction."""
        return np.array([self.gauge_vx, self.gauge_vy, self.gauge_vz])
