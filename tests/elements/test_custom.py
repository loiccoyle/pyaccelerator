from unittest import TestCase

import numpy as np

from accelerator.elements.custom import CustomThin
from accelerator.lattice import Lattice


class TestCustomThin(TestCase):
    def test_init(self):
        custom = CustomThin(transfer_matrix_h=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert custom.length == 0
        assert custom.name.startswith("custom_thin")
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]], name="custom_element")
        assert custom.name == "custom_element"
        custom = CustomThin(transfer_matrix_h=np.identity(3))
        assert np.allclose(custom.transfer_matrix_v, np.identity(3))
        custom = CustomThin(transfer_matrix_v=np.identity(3))
        assert np.allclose(custom.transfer_matrix_h, np.identity(3))
        custom = CustomThin(
            transfer_matrix_h=np.identity(3), transfer_matrix_v=np.identity(3) * 2
        )
        assert np.allclose(custom.transfer_matrix_h, np.identity(3))
        assert np.allclose(custom.transfer_matrix_v, np.identity(3) * 2)
        with self.assertRaises(ValueError):
            custom = CustomThin()

    def test_transfer_matrix(self):
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # the linear part is the identity matrix
        expected_transfer_matrix_h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected_transfer_matrix_v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m_h, m_v = (
            custom._get_transfer_matrix_h(),
            custom._get_transfer_matrix_v(),
        )
        assert np.allclose(m_h, expected_transfer_matrix_h)
        assert np.allclose(m_v, expected_transfer_matrix_v)
        assert np.allclose(custom.m_h, m_h)
        assert np.allclose(custom.m_v, m_v)

    def test_repr(self):
        repr(CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_serialize(self):
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        dic = custom._serialize()
        assert dic["element"] == "CustomThin"
        assert np.allclose(dic["transfer_matrix_h"], custom.transfer_matrix_h)
        assert np.allclose(dic["transfer_matrix_v"], custom.transfer_matrix_v)
        assert dic["name"] == custom.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        custom.transfer_matrix_h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        custom.transfer_matrix_v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dic = custom._serialize()
        assert dic["element"] == "CustomThin"
        assert np.allclose(dic["transfer_matrix_h"], custom.transfer_matrix_h)
        assert np.allclose(dic["transfer_matrix_v"], custom.transfer_matrix_v)
        assert dic["name"] == custom.name

    def test_plot(self):
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        custom.plot()

    def test_copy(self):
        custom = CustomThin([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        copy = custom.copy()
        assert np.allclose(copy.transfer_matrix_h, custom.transfer_matrix_h)
        assert np.allclose(copy.transfer_matrix_v, custom.transfer_matrix_v)
        assert copy.name == custom.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        custom = CustomThin(1)
        custom.transfer_matrix_h = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
        copy = custom.copy()
        assert np.allclose(copy.transfer_matrix_h, custom.transfer_matrix_h)
        assert np.allclose(copy.transfer_matrix_v, custom.transfer_matrix_v)
        assert copy.name == custom.name
