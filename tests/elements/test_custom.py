from unittest import TestCase

import numpy as np

from accelerator.elements.custom import CustomThin
from accelerator.lattice import Lattice


class TestCustomThin(TestCase):
    def test_init(self):
        custom = CustomThin(np.identity(5))
        assert custom.length == 0
        assert custom.name.startswith("custom_thin")
        custom = CustomThin(np.identity(5), name="custom_element")
        assert custom.name == "custom_element"
        custom = CustomThin(np.identity(5))
        assert np.allclose(custom.transfer_matrix, np.identity(5))
        with self.assertRaises(TypeError):
            custom = CustomThin()

    def test_transfer_matrix(self):
        custom = CustomThin(np.identity(5))
        # the linear part is the identity matrix
        expected_transfer_matrix = np.identity(5)
        m = custom._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(custom.m, m)

    def test_repr(self):
        repr(CustomThin(np.identity(5)))

    def test_serialize(self):
        custom = CustomThin(np.identity(5))
        dic = custom._serialize()
        assert dic["element"] == "CustomThin"
        assert np.allclose(dic["transfer_matrix"], custom.transfer_matrix)
        assert dic["name"] == custom.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        custom = CustomThin(np.identity(5))
        custom.transfer_matrix = np.identity(5)
        dic = custom._serialize()
        assert dic["element"] == "CustomThin"
        assert np.allclose(dic["transfer_matrix"], custom.transfer_matrix)
        assert dic["name"] == custom.name

    def test_copy(self):
        custom = CustomThin(np.identity(5))
        copy = custom.copy()
        assert np.allclose(copy.transfer_matrix, custom.transfer_matrix)
        assert copy.name == custom.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        custom = CustomThin(1)
        custom.transfer_matrix = np.identity(5)
        copy = custom.copy()
        assert np.allclose(copy.transfer_matrix, custom.transfer_matrix)
        assert copy.name == custom.name
