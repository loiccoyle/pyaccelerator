from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from accelerator.beam import Beam
from accelerator.elements.dipole import Dipole, DipoleThin
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import Quadrupole, QuadrupoleThin
from accelerator.lattice import Lattice, Plotter


class TestLattice(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_folder = Path("test_lattice")
        cls.test_folder.mkdir()

    def test_init(self):
        lat = Lattice()
        assert len(lat) == 0
        lat = Lattice([Drift(1)])
        assert len(lat) == 1

    def test_transfer_matrixes(self):
        lat = Lattice([Drift(1), Drift(1)])
        assert np.allclose(lat.m_h, Drift(2).m_h)
        assert np.allclose(lat.m_v, Drift(2).m_v)

        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        assert np.allclose(lat.m_h, QuadrupoleThin(0.8).m_h @ Drift(1).m_h)
        assert np.allclose(lat.m_v, QuadrupoleThin(0.8).m_v @ Drift(1).m_v)

    def test_slice(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
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
        quad = QuadrupoleThin(0.8)
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
        lat.insert(0, QuadrupoleThin(0.8))
        assert len(lat) == 2
        assert np.allclose(lat.m_h, Drift(1).m_h @ QuadrupoleThin(0.8).m_h)
        assert np.allclose(lat.m_v, Drift(1).m_v @ QuadrupoleThin(0.8).m_v)

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
        s, u, u_prime, dp = lat.transport(phasespace=[1, 0, 0])
        assert len(u) == 2
        assert len(u_prime) == 2
        assert len(s) == 2
        assert np.allclose(u, np.array([1, 1]))
        assert np.allclose(u_prime, np.array([0, 0]))
        assert np.allclose(s, np.array([0, 1]))
        assert dp.shape == u.shape

        s, u, u_prime, dp = lat.transport([1, 1, 0])
        assert len(u) == 2
        assert len(u_prime) == 2
        assert len(s) == 2
        assert np.allclose(u, np.array([1, 2]))
        assert np.allclose(u_prime, np.array([1, 1]))
        assert np.allclose(s, np.array([0, 1]))
        assert dp.shape == u.shape

        # make sure the transport is consistent:
        lat = Lattice([Drift(2)])
        s_1, u_1, u_prime_1, dp = lat.transport([1, 1, 0])
        lat = Lattice([Drift(1), Drift(1)])
        s_2, u_2, u_prime_2, dp = lat.transport([1, 1, 0])
        assert len(u_2) == 3
        assert len(u_prime_2) == 3
        assert len(s_2) == 3
        assert u_1[-1] == u_2[-1]
        assert u_prime_1[-1] == u_prime_2[-1]
        assert s_1[-1] == s_2[-1]
        assert dp.shape == u_2.shape

    def test_tranport_twiss(self):
        # transporting phase space coords:
        f = 0.8
        l = 1
        FODO = Lattice(
            [
                QuadrupoleThin(2 * f),
                Drift(l),
                # expected beta minimum
                QuadrupoleThin(-f),
                Drift(l),
                QuadrupoleThin(2 * f),
            ]
        )
        s, beta, alpha, gamma = FODO.transport(twiss=FODO.m_h.twiss_solution)
        assert len(beta) == len(FODO) + 1
        assert len(alpha) == len(FODO) + 1
        assert len(gamma) == len(FODO) + 1
        assert len(s) == len(FODO) + 1
        # make sure the default is correct
        s_2, beta_2, alpha_2, gamma_2 = FODO.transport()
        assert np.allclose(beta, beta_2)
        assert np.allclose(alpha, alpha_2)
        assert np.allclose(gamma, gamma_2)
        assert np.allclose(s, s_2)
        # make sure the periodic solution is infact periodic
        self.assertAlmostEqual(beta[0], beta[-1])
        self.assertAlmostEqual(alpha[0], alpha[-1])
        self.assertAlmostEqual(gamma[0], gamma[-1])

        # make sure the beta is minimum in the right place
        assert np.argmin(beta) == 2

        # now in the v plane
        s, beta, alpha, gamma = FODO.transport(twiss=FODO.m_v.twiss_solution, plane="v")
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
                QuadrupoleThin(2 * f),
                Drift(l),
                QuadrupoleThin(-f),
                Drift(l),
                QuadrupoleThin(2 * f),
            ]
        )
        beam = Beam()
        s, u, u_prime, dp = FODO.transport(
            beam.ellipse(FODO.m_h.twiss_solution, n_angles=n_angles)
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
                QuadrupoleThin(2 * f),
                Drift(l),
                QuadrupoleThin(f),
                Drift(l),
                QuadrupoleThin(2 * f),
            ]
        )
        n_particles = 10
        beam = Beam(n_particles=n_particles)
        s, u, u_prime, dp = FODO.transport(
            phasespace=beam.match(FODO.m_h.twiss_solution)
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
        lat.transport(phasespace=np.array([1, 2, 1]))
        lat.transport(phasespace=np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        # make sure lists work
        lat.transport(phasespace=[1, 2, 3])
        lat.transport(phasespace=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        # tuples ?
        lat.transport(phasespace=(1, 2, 3))
        lat.transport(phasespace=((1, 2, 3), (1, 2, 3), (1, 2, 3)))

    def test_transport_error(self):
        lat = Lattice([Drift(1)])
        with self.assertRaises(ValueError):
            lat.transport(phasespace=[1, 0, 1], twiss=[1, 1, 1])
        with self.assertRaises(ValueError):
            lat.transport(twiss="solution")

    def test_save_load(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.5)])
        save_file = self.test_folder / "lattice.json"
        lat.save(save_file)
        assert save_file.is_file()
        lat_loaded = Lattice.load(save_file)
        assert isinstance(lat_loaded[0], Drift)
        assert isinstance(lat_loaded[1], QuadrupoleThin)
        assert lat_loaded[0].length, 1
        assert lat_loaded[1].f, 0.5

    def test_plot(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        lat.plot()

    def test_copy(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8), Dipole(1, 1)])
        lat_copy = lat.copy()
        assert len(lat) == len(lat_copy)
        assert lat[0].l == lat_copy[0].l
        assert lat[0].name == lat_copy[0].name
        assert lat[1].f == lat_copy[1].f
        assert lat[1].name == lat_copy[1].name
        assert lat[2].rho == lat_copy[2].rho
        assert lat[2].theta == lat_copy[2].theta
        assert lat[2].name == lat_copy[2].name
        # new element instances are created
        assert not any([id(orig) == id(copy) for orig, copy in zip(lat, lat_copy)])

        lat_shallow_copy = lat.copy(deep=False)
        # the same instances are in both lattices
        assert all([id(orig) == id(copy) for orig, copy in zip(lat, lat_shallow_copy)])

    def test_repr_(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8), Dipole(1, 1)])
        repr(lat)

    def test_search(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8), Dipole(1, 1)])
        assert lat.search("drift") == [0]
        assert lat.search("quadrupole") == [1]
        assert lat.search("dipole") == [2]

        lat = Lattice(
            [QuadrupoleThin(0.8, name="quad_f"), QuadrupoleThin(-0.8, name="quad_d")]
        )
        assert lat.search("quad_f") == [0]
        assert lat.search("quad_d") == [1]
        assert lat.search("quad_[fd]") == [0, 1]

        with self.assertRaises(ValueError):
            lat.search("drift")

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.test_folder)


# I don't really know how to test the correctness of plots...
class TestPlotter(TestCase):
    def test_plotter(self):
        lat = Lattice(
            [
                Drift(1),
                Quadrupole(1 / 100, 1),
                Quadrupole(-1 / 100, 1),
                Quadrupole(0, 1),
                QuadrupoleThin(0.6),
                QuadrupoleThin(-0.6),
                QuadrupoleThin(0),
                Dipole(1, np.pi / 2),
                DipoleThin(np.pi / 16),
            ]
        )
        plotter = Plotter(lat)

        fig, axes = plotter.top_down()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)
        fig, axes = plotter.lattice()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)
        fig, axes = plotter()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

        # testing the margins
        lat = Lattice(
            [
                Drift(0.01),
            ]
        )
        plotter = Plotter(lat)
        fig, axes = plotter.top_down()
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, plt.Axes)

    def test_repr(self):
        lat = Lattice([Drift(1)])
        plotter = Plotter(lat)
        repr(plotter)
