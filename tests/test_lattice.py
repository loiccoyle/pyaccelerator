from unittest import TestCase

import numpy as np

from accelerator.lattice import Lattice
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import Quadrupole
from accelerator.elements.dipole import Dipole


class TestLattice(TestCase):
    def test_init(self):
        lat = Lattice()
        assert len(lat) == 0
        lat = Lattice([Drift(1)])
        assert len(lat) == 1

    def test_transfer_matrixes(self):
        lat = Lattice([Drift(1), Drift(1)])
        assert np.allclose(lat.m_h, Drift(2).m_h)
        assert np.allclose(lat.m_v, Drift(2).m_v)

        lat = Lattice([Drift(1), Quadrupole(0.8)])
        assert np.allclose(lat.m_h, Quadrupole(0.8).m_h @ Drift(1).m_h)
        assert np.allclose(lat.m_v, Quadrupole(0.8).m_v @ Drift(1).m_v)

    def test_slice(self):
        lat = Lattice([Drift(1), Quadrupole(0.8)])
        lat_sliced = lat.slice(Drift, 10)
        assert len(lat_sliced) == 11
        assert all([isinstance(element, Drift) for element in lat_sliced[:10]])
        assert all([element.length == 0.1 for element in lat_sliced[:10]])
        assert np.allclose(lat.m_h, lat_sliced.m_h)
        assert np.allclose(lat.m_v, lat_sliced.m_v)

        lat = Lattice([Drift(1), Dipole(np.pi/4, 100)])
        lat = lat.slice(Drift, 10)
        assert len(lat) == 11
        lat = lat.slice(Dipole, 10)
        assert len(lat) == 20

    def test_some_list_methods(self):
        lat = Lattice()
        assert len(lat) == 0
        with self.assertRaises(TypeError):
            lat.m_h
        lat.append(Drift(1))
        assert len(lat) == 1
        assert np.allclose(lat.m_h, Drift(1).m_h)
        lat.append(Drift(1))
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(2).m_h)

        drift = Drift(1)
        quad = Quadrupole(0.8)
        lat = Lattice([drift, quad])
        assert lat[0] == drift
        assert lat[1] == quad
        lat.reverse()
        assert lat[0] == quad
        assert lat[1] == drift

        lat = Lattice([Drift(1)])
        assert len(lat) == 1
        assert np.allclose(lat.m_h, Drift(1).m_h)
        assert np.allclose(lat.m_v, Drift(1).m_v)
        lat = lat * 2
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(2).m_h)
        assert np.allclose(lat.m_v, Drift(2).m_v)

        lat = Lattice([Drift(1)])
        lat = lat + lat
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(2).m_h)
        assert np.allclose(lat.m_v, Drift(2).m_v)

        lat = Lattice([Drift(1)])
        lat.extend([Drift(1)])
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(2).m_h)
        assert np.allclose(lat.m_v, Drift(2).m_v)

        lat = Lattice([Drift(1)])
        lat.insert(0, Quadrupole(0.8))
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(1).m_h @ Quadrupole(0.8).m_h)
        assert np.allclose(lat.m_v, Drift(1).m_v @ Quadrupole(0.8).m_v)

        lat = Lattice([Drift(1)])
        popped = lat.pop(0)
        assert len(lat) == 0
        assert isinstance(popped, Drift)

    def test_plot(self):
        lat = Lattice([Drift(1)])
        lat.plot()
