import numpy as np
import pytest
from pyrs.dataobjects.sample_logs import DirectionExtents


class TestDirectionExtents:

    def test_init(self):
        d = DirectionExtents(range(2))
        assert list(d) == pytest.approx([0, 1, 1])
        d = DirectionExtents(range(42))
        assert list(d) == pytest.approx([0, 41, 1])

        precision = DirectionExtents.precision
        d = DirectionExtents(np.arange(0, 1, precision))
        assert d.numpoints == int(1 / precision)
        assert list(d) == [0, 1 - precision, precision]

        too_close_together = 0.1 * precision
        d = DirectionExtents(np.arange(0, 1, too_close_together))
        assert d.numpoints == int(1 / precision)  # we cannot resolve more coords than those limited by the precision
        assert list(d) == [0, 1 - too_close_together, (1 - too_close_together) / (d.numpoints - 1)]

    def test_to_createmd(self):
        d = DirectionExtents(range(42))
        assert d.to_createmd == '-0.5,41.5'

    def test_to_binmd(self):
        d = DirectionExtents(range(42))
        assert d.to_binmd('label') == 'label,-0.5,41.5,42'


if __name__ == '__main__':
    pytest.main()
