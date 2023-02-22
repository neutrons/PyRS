import numpy as np

from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField

from pyrs.interface.strainstressviewer.model import Model

d0_default = 1.0828
d0e_default = 0


def get_test_stress(test_data_dir: str):
    sample11 = StrainField(test_data_dir + '/3393_PWHT-TD.h5')
    sample22 = StrainField(test_data_dir + '/3394_PWHT-ND.h5')
    sample33 = StrainField(test_data_dir + '/3395_PWHT-LD.h5')
    return StressField(sample11, sample22, sample33, 200, 0.3)


class TestD0Grid:

    def test_validation_full_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-all.csv", delimiter=',', unpack=True)

        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[0] == 1

    def test_validation_partial_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-none.csv", delimiter=',', unpack=True)

        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[0] == -1

    def test_validation_no_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-some.csv", delimiter=',', unpack=True)

        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[0] == 0

    def test_correction_full_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-all.csv", delimiter=',', unpack=True)

        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[1], x)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[2], y)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[3], z)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[4], d0)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[5], d0e)

    def test_correction_partial_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-some.csv", delimiter=',', unpack=True)
        x_clean, y_clean, z_clean, d0_clean, d0e_clean = np.loadtxt(test_data_dir + "/do-grid-some-cleaned.csv",
                                                                    delimiter=',', unpack=True)

        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[1], x_clean)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[2], y_clean)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[3], z_clean)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[4], d0_clean)
        assert np.array_equal(model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[5], d0e_clean)

    def test_correction_no_match(self, test_data_dir: str):
        model = Model()
        model.stress = get_test_stress(test_data_dir)
        x, y, z, d0, d0e = np.loadtxt(test_data_dir + "/do-grid-none.csv", delimiter=',', unpack=True)

        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[1] is None
        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[2] is None
        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[3] is None
        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[4] is None
        assert model.validate_d0_grid_data(x, y, z, d0, d0e, d0_default, d0e_default)[5] is None
