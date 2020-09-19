from unittest import TestCase

import numpy as np

from accelerator.elements.quadrupole import Quadrupole


class TestQuadrupole(TestCase):
    def test_init(self):
        quadrupole = Quadrupole(1)
        assert quadrupole.length == 0
        assert quadrupole.name.startswith("quadrupole_")
        quadrupole = Quadrupole(1, name="quad_f")
        assert quadrupole.name == "quad_f"

    def test_transfer_matrix(self):
        quadrupole = Quadrupole(1)
        expected_transfer_matrix_h = np.array([[1, 0], [-1, 1]])
        expected_transfer_matrix_v = np.array([[1, 0], [1, 1]])
        m_h, m_v = (
            quadrupole._get_transfer_matrix_h(),
            quadrupole._get_transfer_matrix_v(),
        )
        assert np.allclose(m_h, expected_transfer_matrix_h)
        assert np.allclose(m_v, expected_transfer_matrix_v)
        assert np.allclose(quadrupole.m_h, m_h)
        assert np.allclose(quadrupole.m_v, m_v)

    def test_repr(self):
        repr(Quadrupole(1))

    def test_serialize(self):
        quad = Quadrupole(0.6)
        dic = quad._serialize()
        assert dic["element"] == "Quadrupole"
        assert dic["f"] == quad.f
        assert dic["name"] == quad.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        quad = Quadrupole(0.6)
        quad.f = 0.8
        dic = quad._serialize()
        assert dic["element"] == "Quadrupole"
        assert dic["f"] == quad.f
        assert dic["name"] == quad.name

    def test_plot(self):
        quad = Quadrupole(0.8)
        quad.plot()

    def test_copy(self):
        quad = Quadrupole(1)
        copy = quad.copy()
        assert copy.f == quad.f
        assert copy.name == quad.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        quad = Quadrupole(1)
        quad.f = 2
        copy = quad.copy()
        assert copy.f == quad.f
        assert copy.name == quad.name
