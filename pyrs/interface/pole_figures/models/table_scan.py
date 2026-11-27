"""
Table scan tab inputs: run title, cutoff, count-time rule and measurement order.

The table scan presenter reads these with the last prediction to write the measurement table.

Example:
    >>> model = TableScanModel(cap_on=True, max_s=60.0)
    >>> model.scaled, model.cap
    (True, 60.0)
"""

import math
from typing import Final, Literal

from pydantic import BaseModel, Field

TimeMode = Literal["Scale with correction", "Fixed per point"]
SCALED: Final = "Scale with correction"
FIXED: Final = "Fixed per point"
SETTING_FIELDS = ["run_title", "cutoff", "time_mode", "cap_on", "max_s", "serpentine"]


class TableScanModel(BaseModel):
    """
    Inputs of the Table scan tab. The cutoff is copied from the Predict tab after each prediction.

    Attributes:
        run_title: Start of each run title and file name.
        cutoff: Points with a predicted correction above this are not measured.
        time_mode: Scale the count time with the correction, or a fixed time per point.
        cap_on: Limit the longest count to ``max_s``.
        max_s: Longest count time per point when capped (s).
        serpentine: Alternate the phi direction from one chi ring to the next.
    """

    run_title: str = Field(default="", title="Run title", description="Start of each run title and file name")
    cutoff: float = Field(
        default=3.0, gt=1, le=1000, title="Leave out correction >",
        description="Points with a predicted correction above this are not measured",
    )  # fmt: skip
    time_mode: TimeMode = Field(
        default=SCALED, title="Count time",
        description="Scale: time = base x correction, so corrected statistics are about even. Fixed: base for all.",
    )  # fmt: skip
    cap_on: bool = Field(default=False, title="Cap time per point", description="Limit the longest count")
    max_s: float = Field(default=120.0, ge=1, le=36000, title="Cap (s)")
    serpentine: bool = Field(
        default=True, title="Alternate phi by chi ring", description="phi up on one ring, down on the next (no swing)"
    )

    @property
    def scaled(self) -> bool:
        """True when the count time scales with the correction."""
        return self.time_mode == SCALED

    @property
    def cap(self) -> float:
        """Cap on the count time per point (s), inf when the cap is off."""
        return self.max_s if self.cap_on else math.inf
