from unittest import TestCase

import numpy as np

from accelerator.elements.quadrupole import Quadrupole


class TestQuadrupole(TestCase):
    def test_init(self):
        quadrupole = Quadrupole(1)
        assert quadrupole.length == 0
        assert quadrupole._m_h == None
        assert quadrupole._m_v == None

    def test_transfer_matrix(self):
        quadrupole = Quadrupole(1)
        expected_transfer_matrix_h = np.array([[1, 0], [-1, 1]])
        expected_transfer_matrix_v = np.array([[1, 0], [1, 1]])
        out = quadrupole.transfer_matrix()
        assert np.allclose(out[0], expected_transfer_matrix_h)
        assert np.allclose(out[1], expected_transfer_matrix_v)
        assert np.allclose(quadrupole.m_h, out[0])
        assert np.allclose(quadrupole.m_v, out[1])

    def test_repr(self):
        repr(Quadrupole(1))

    def test_serialize(self):
        quad = Quadrupole(0.6)
        dic = quad._serialize()
        assert dic["element"] == "Quadrupole"
        assert dic["f"] == 0.6

    def test_plot(self):
        quad = Quadrupole(0.8)
        quad.plot()
