from unittest import TestCase

import numpy as np

from accelerator.elements.quadrupole import Quadrupole, QuadrupoleThin
from accelerator.lattice import Lattice


class TestQuadrupole(TestCase):
    def test_init(self):
        quadrupole = Quadrupole(k=1 / 2, l=5)
        assert quadrupole.length == 5
        assert quadrupole.name.startswith("quadrupole_")
        quadrupole = Quadrupole(k=1 / 2, l=5, name="quad_f")
        assert quadrupole.name == "quad_f"

    def test_transfer_matrix(self):
        quadrupole = Quadrupole(k=1 / 100, l=1)
        m = quadrupole._get_transfer_matrix()
        expected_transfer_matrix = np.array(
            [
                [0.99500417, 0.99833417, 0, 0, 0],
                [-0.00998334, 0.99500417, 0, 0, 0],
                [0, 0, 1.00500417, 1.0016675, 0],
                [0, 0, 0.01001668, 1.00500417, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(quadrupole.m, m)

        quadrupole = Quadrupole(k=-1 / 100, l=1)
        m = quadrupole._get_transfer_matrix()
        expected_transfer_matrix = np.array(
            [
                [1.00500417, 1.0016675, 0, 0, 0],
                [0.01001668, 1.00500417, 0, 0, 0],
                [0, 0, 0.99500417, 0.99833417, 0],
                [0, 0, -0.00998334, 0.99500417, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(quadrupole.m, m)

    def test_transport(self):
        quadrupole = Quadrupole(k=1, l=1)
        out = quadrupole._transport(np.array([0, 0, 0, 0, 0]))
        assert np.allclose(out, [0, 0, 0, 0, 0])
        # out = quadrupole._transport(np.array([1, 0, 0, 0, 0]))
        # assert np.allclose(out, [1, 0, 0, 0, 0])
        # out = quadrupole._transport(np.array([1, 0, 0, 0, 1]))
        # assert np.allclose(out, [1, 0, 0, 0, 1])

    def test_slice(self):
        quadrupole = Quadrupole(k=1 / 2, l=5)
        assert len(quadrupole.slice(10)) == 10
        assert isinstance(quadrupole.slice(10), Lattice)
        assert np.allclose(quadrupole.slice(10).m, quadrupole.m)
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
        expected_transfer_matrix = np.array(
            [
                [1, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        m = quadrupole._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(quadrupole.m, m)

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
