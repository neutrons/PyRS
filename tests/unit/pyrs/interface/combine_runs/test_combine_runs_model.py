"""A new combine forgets the previous combination, even when it fails (#931)

Otherwise a refused or failed combine left the earlier combination in place, ready to be exported
under the new name. No project file is needed: the combine fails on files that do not exist.
"""

import pytest

from pyrs.interface.combine_runs.combine_runs_model import CombineRunsModel


def test_failed_combine_clears_previous_combination(tmp_path):
    """The previous workspace, record and warnings are gone after a combine that raises."""
    # Arrange
    model = CombineRunsModel()
    model._hidra_ws = object()
    model._record = {"source": "previous"}
    model.warnings = ["previous warning"]

    # Act
    # a missing file raises OSError from the file-record lookup
    with pytest.raises(OSError):
        model.combine_project_files([str(tmp_path / "missing_1.h5"), str(tmp_path / "missing_2.h5")])

    # Assert
    assert model._hidra_ws is None
    assert model._record is None
    assert model.warnings == []
