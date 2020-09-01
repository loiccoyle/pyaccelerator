from unittest import TestCase

import numpy as np

from accelerator.elements.drift import Drift
from accelerator.lattice import Lattice


class TestDrift(TestCase):
    def test_init(self):
        drift = Drift(1)
        assert drift.length == 1
        assert drift._m_h == None
        assert drift._m_v == None

    def test_transfer_matrix(self):
        drift = Drift(1)
        expected_transfer_matrix = np.array([[1, 1], [0, 1]])
        out = drift.transfer_matrix()
        assert np.allclose(out[0], expected_transfer_matrix)
        assert np.allclose(out[1], expected_transfer_matrix)
        assert np.allclose(drift.m_h, out[0])
        assert np.allclose(drift.m_v, out[1])

    def test_slice(self):
        drift = Drift(1)
        assert len(drift.slice(10)) == 10
        assert isinstance(drift.slice(10), Lattice)
        assert np.allclose(drift.slice(10).m_h, drift.m_h)
        assert np.allclose(drift.slice(10).m_v, drift.m_v)

    def test_repr(self):
        repr(Drift(1))

    def test_dxztheta_ds(self):
        drift = Drift(1)
        assert np.allclose(drift._dxztheta_ds(0, 1), [1, 0, 0])

    def test_plot(self):
        drift = Drift(1)
        drift.plot()
