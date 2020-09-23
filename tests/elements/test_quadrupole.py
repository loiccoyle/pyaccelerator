from unittest import TestCase

import numpy as np

from accelerator.lattice import Lattice
from accelerator.elements.quadrupole import QuadrupoleThin, Quadrupole


class TestQuadrupole(TestCase):
    def test_init(self):
        quadrupole = Quadrupole(k=1 / 2, l=5)
        assert quadrupole.length == 5
        assert quadrupole.name.startswith("quadrupole_")
        quadrupole = Quadrupole(k=1 / 2, l=5, name="quad_f")
        assert quadrupole.name == "quad_f"

    def test_transfer_matrix(self):
        quadrupole = Quadrupole(k=1 / 100, l=1)
        m_h, m_v = (
            quadrupole._get_transfer_matrix_h(),
            quadrupole._get_transfer_matrix_v(),
        )
        expected_transfer_matrix_h = np.array(
            [[0.99500417, 0.99833417], [-0.00998334, 0.99500417]]
        )
        expected_transfer_matrix_v = np.array(
            [[1.00500417, 1.0016675], [0.01001668, 1.00500417]]
        )
        assert np.allclose(m_h, expected_transfer_matrix_h)
        assert np.allclose(m_v, expected_transfer_matrix_v)
        assert np.allclose(quadrupole.m_h, m_h)
        assert np.allclose(quadrupole.m_v, m_v)

        quadrupole = Quadrupole(k=-1 / 100, l=1)
        m_h, m_v = (
            quadrupole._get_transfer_matrix_h(),
            quadrupole._get_transfer_matrix_v(),
        )
        expected_transfer_matrix_h = np.array(
            [[1.00500417, 1.0016675], [0.01001668, 1.00500417]]
        )
        expected_transfer_matrix_v = np.array(
                [[0.99500417, 0.99833417], [-0.00998334, 0.99500417]]
                )
        assert np.allclose(m_h, expected_transfer_matrix_h)
        assert np.allclose(m_v, expected_transfer_matrix_v)
        assert np.allclose(quadrupole.m_h, m_h)
        assert np.allclose(quadrupole.m_v, m_v)

    def test_slice(self):
        quadrupole = Quadrupole(k=1 / 2, l=5)
        assert len(quadrupole.slice(10)) == 10
        assert isinstance(quadrupole.slice(10), Lattice)
        assert np.allclose(quadrupole.slice(10).m_h, quadrupole.m_h)
        assert np.allclose(quadrupole.slice(10).m_v, quadrupole.m_v)
        assert quadrupole.slice(10)[0].name == quadrupole.name + "_slice_0"

    def test_repr(self):
        repr(Quadrupole(1 / 100, 1))

    def test_serialize(self):
        quad = Quadrupole(1 / 100, 1)
        dic = quad._serialize()
        assert dic["element"] == "Quadrupole"
        assert dic["k"] == quad.k
        assert dic["l"] == quad.l
        assert dic["name"] == quad.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        quad = Quadrupole(1 / 100, 1)
        quad.l = 2
        dic = quad._serialize()
        assert dic["element"] == "Quadrupole"
        assert dic["k"] == quad.k
        assert dic["l"] == quad.l
        assert dic["name"] == quad.name

    def test_plot(self):
        quad = Quadrupole(1 / 100, 1)
        quad.plot()

    def test_copy(self):
        quad = Quadrupole(1 / 100, 1)
        copy = quad.copy()
        assert copy.k == quad.k
        assert copy.l == quad.l
        assert copy.name == quad.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        quad = Quadrupole(1 / 100, 1)
        quad.l = 2
        copy = quad.copy()
        assert copy.k == quad.k
        assert copy.l == quad.l
        assert copy.name == quad.name


class TestQuadrupoleThin(TestCase):
    def test_init(self):
        quadrupole = QuadrupoleThin(1)
        assert quadrupole.length == 0
        assert quadrupole.name.startswith("quadrupole_thin")
        quadrupole = QuadrupoleThin(1, name="quad_f")
        assert quadrupole.name == "quad_f"

    def test_transfer_matrix(self):
        quadrupole = QuadrupoleThin(1)
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
        repr(QuadrupoleThin(1))

    def test_serialize(self):
        quad = QuadrupoleThin(0.6)
        dic = quad._serialize()
        assert dic["element"] == "QuadrupoleThin"
        assert dic["f"] == quad.f
        assert dic["name"] == quad.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        quad = QuadrupoleThin(0.6)
        quad.f = 0.8
        dic = quad._serialize()
        assert dic["element"] == "QuadrupoleThin"
        assert dic["f"] == quad.f
        assert dic["name"] == quad.name

    def test_plot(self):
        quad = QuadrupoleThin(0.8)
        quad.plot()

    def test_copy(self):
        quad = QuadrupoleThin(1)
        copy = quad.copy()
        assert copy.f == quad.f
        assert copy.name == quad.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        quad = QuadrupoleThin(1)
        quad.f = 2
        copy = quad.copy()
        assert copy.f == quad.f
        assert copy.name == quad.name
