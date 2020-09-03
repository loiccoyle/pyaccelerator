from unittest import TestCase

import numpy as np

from accelerator.beam import Beam
from accelerator.elements.dipole import Dipole
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import Quadrupole
from accelerator.lattice import Lattice


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

        lat = Lattice([Drift(1), Dipole(np.pi / 4, 100)])
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

        drift = Drift(1)
        lat = Lattice([drift])
        lat.remove(drift)
        assert len(lat) == 0

        lat = Lattice([Drift(1)])
        lat.clear()
        assert len(lat) == 0

    def test_tranport_phasespace(self):
        # transporting phase space coords:
        lat = Lattice([Drift(1)])
        u, u_prime, s = lat.transport([1, 0])
        assert len(u) == 2
        assert len(u_prime) == 2
        assert len(s) == 2
        assert np.allclose(u, np.array([1, 1]))
        assert np.allclose(u_prime, np.array([0, 0]))
        assert np.allclose(s, np.array([0, 1]))

        u, u_prime, s = lat.transport([1, 1])
        assert len(u) == 2
        assert len(u_prime) == 2
        assert len(s) == 2
        assert np.allclose(u, np.array([1, 2]))
        assert np.allclose(u_prime, np.array([1, 1]))
        assert np.allclose(s, np.array([0, 1]))

        # make sure the transport is consistent:
        lat = Lattice([Drift(2)])
        u_1, u_prime_1, s_1 = lat.transport([1, 1])
        lat = Lattice([Drift(1), Drift(1)])
        u_2, u_prime_2, s_2 = lat.transport([1, 1])
        assert len(u_2) == 3
        assert len(u_prime_2) == 3
        assert len(s_2) == 3
        assert u_1[-1] == u_2[-1]
        assert u_prime_1[-1] == u_prime_2[-1]
        assert s_1[-1] == s_2[-1]

    def test_tranport_twiss(self):
        # transporting phase space coords:
        f = 0.8
        l = 1
        FODO = Lattice(
            [
                Quadrupole(2 * f),
                Drift(l),
                # expected beta minimum
                Quadrupole(-f),
                Drift(l),
                Quadrupole(2 * f),
            ]
        )
        beta, alpha, gamma, s = FODO.transport(FODO.m_h.twiss.invariant)
        assert len(beta) == len(FODO) + 1
        assert len(alpha) == len(FODO) + 1
        assert len(gamma) == len(FODO) + 1
        assert len(s) == len(FODO) + 1
        # make sure the periodic solution is infact periodic
        self.assertAlmostEqual(beta[0], beta[-1])
        self.assertAlmostEqual(alpha[0], alpha[-1])
        self.assertAlmostEqual(gamma[0], gamma[-1])

        # make sure the beta is minimum in the right place
        assert np.argmin(beta) == 2

        # now in the v plane
        beta, alpha, gamma, s = FODO.transport(FODO.m_v.twiss.invariant, plane="v")
        assert len(beta) == len(FODO) + 1
        assert len(alpha) == len(FODO) + 1
        assert len(gamma) == len(FODO) + 1
        assert len(s) == len(FODO) + 1
        # make sure the periodic solution is infact periodic
        self.assertAlmostEqual(beta[0], beta[-1])
        self.assertAlmostEqual(alpha[0], alpha[-1])
        self.assertAlmostEqual(gamma[0], gamma[-1])

        # make sure the beta is maximum in the right place
        assert np.argmax(beta) == 2

    def test_transport_ellipse(self):
        # transporting phase space coords:
        f = 0.8
        l = 1
        n_angles = 100
        FODO = Lattice(
            [
                Quadrupole(2 * f),
                Drift(l),
                Quadrupole(-f),
                Drift(l),
                Quadrupole(2 * f),
            ]
        )
        beam = Beam()
        u, u_prime, s = FODO.transport(
            beam.phasespace(FODO.m_h.twiss.invariant, n_angles=n_angles)
        )
        assert u.shape[-1] == len(FODO) + 1
        assert u_prime.shape[-1] == len(FODO) + 1
        assert len(s) == len(FODO) + 1
        assert u.shape[0] == n_angles
        assert u_prime.shape[0] == n_angles

    def test_transport_distribution(self):
        f = 0.8
        l = 1
        FODO = Lattice(
            [
                Quadrupole(2 * f),
                Drift(l),
                Quadrupole(f),
                Drift(l),
                Quadrupole(2 * f),
            ]
        )
        n_particles = 10
        beam = Beam(n_particles=n_particles)
        u, u_prime, s = FODO.transport(
            beam.matched_particle_distribution(FODO.m_h.twiss.invariant)
        )
        assert u.shape[-1] == len(FODO) + 1
        assert u_prime.shape[-1] == len(FODO) + 1
        assert len(s) == len(FODO) + 1
        assert u.shape[0] == n_particles
        assert u_prime.shape[0] == n_particles

    def test_transport_input(self):
        # just checking it doesn't raise ValueError
        lat = Lattice([Drift(1)])
        # make sure arrays also work
        lat.transport(np.array([1, 2]))
        lat.transport(np.array([1, 2, 3]))
        lat.transport(np.array([[1, 2], [1, 2]]))
        # make sure lists work
        lat.transport([1, 2])
        lat.transport([1, 2, 3])
        lat.transport([[1, 2], [1, 2]])
        # tuples ?
        lat.transport((1, 2))
        lat.transport((1, 2, 3))
        lat.transport(((1, 2), (1, 2)))

    def test_transport_error(self):
        lat = Lattice([Drift(1)])
        with self.assertRaises(ValueError):
            lat.transport([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            lat.transport([1])
        with self.assertRaises(ValueError):
            lat.transport(np.ones((2,2,2)))

    def test_plot(self):
        lat = Lattice([Drift(1), Quadrupole(0.8)])
        lat.plot()