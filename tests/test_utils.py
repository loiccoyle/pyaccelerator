from unittest import TestCase

import numpy as np

from pyaccelerator import utils
from pyaccelerator.elements.drift import Drift
from pyaccelerator.elements.quadrupole import QuadrupoleThin
from pyaccelerator.lattice import Lattice


class TestUtils(TestCase):
    def test_to_v_vec(self):
        assert utils.to_v_vec([1, 2]).shape == (2, 1)
        assert utils.to_v_vec([1, 2, 3]).shape == (3, 1)
        assert np.allclose(utils.to_v_vec([1, 2]), np.array([1, 2]).reshape(2, 1))
        with self.assertRaises(ValueError):
            utils.to_v_vec([[1, 2], [2, 3]])

    def test_to_twiss(self):
        assert np.allclose(utils.to_twiss([1, 0, 1]), np.array([1, 0, 1]).reshape(3, 1))
        assert np.allclose(
            utils.to_twiss([1, 0, None]), np.array([1, 0, 1]).reshape(3, 1)
        )
        with self.assertRaises(ValueError):
            utils.to_twiss([1, None, None])
        with self.assertRaises(ValueError):
            utils.to_twiss([1, 2, 3, 4])

    def test_to_phase_coord(self):
        assert np.allclose(
            utils.to_phase_coord([1, 2, 0]), np.array([1, 2, 0]).reshape(3, 1)
        )
        with self.assertRaises(ValueError):
            utils.to_phase_coord([1, 2])

    def test_complete_twiss(self):
        assert utils.complete_twiss(beta=1, alpha=0, gamma=1) == (1, 0, 1)
        assert utils.complete_twiss(beta=1, alpha=0, gamma=None) == (1, 0, 1)
        assert utils.complete_twiss(beta=1, alpha=None, gamma=1) == (1, 0, 1)
        assert utils.complete_twiss(beta=None, alpha=0, gamma=1) == (1, 0, 1)
        with self.assertRaises(ValueError):
            assert utils.complete_twiss(beta=1, alpha=None, gamma=None)

    def test_compute_one_turn(self):
        list_of_m = [np.diag([1, 1])] * 2
        assert np.allclose(utils.compute_one_turn(list_of_m), np.diag([1, 1]))
        list_of_m = [np.array([[1, 1], [0, 1]]), np.array([[1, 0], [1, 1]])]
        assert np.allclose(
            utils.compute_one_turn(list_of_m), np.array([[1, 1], [1, 2]])
        )
        list_of_m.reverse()
        assert np.allclose(
            utils.compute_one_turn(list_of_m), np.array([[2, 1], [1, 1]])
        )

    def test_compute_twiss_clojure(self):
        assert utils.compute_twiss_clojure([1, 0, 1]) == 1
        assert utils.compute_twiss_clojure([1, 0, 0]) == 0

    def test_compute_m_twiss(self):
        transfer_matrix = np.array([[1, 1], [0, 1]])
        twiss = np.array([[1, -2, 1], [0, 1, -1], [0, 0, 1]])
        assert np.allclose(utils.compute_m_twiss(transfer_matrix), twiss)

    def test_compute_twiss_solution(self):
        transfer_matrix = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            utils.compute_twiss_solution(transfer_matrix)

        # TODO: make sure this is correct
        transfer_matrix_fodo = Lattice(
            [
                QuadrupoleThin(2 * 0.8),
                Drift(1),
                QuadrupoleThin(-0.8),
                Drift(1),
                QuadrupoleThin(2 * 0.8),
            ]
        ).m.h
        expected = np.array([[3.33066560e00], [1.11528141e-16], [3.00240288e-01]])
        assert np.allclose(utils.compute_twiss_solution(transfer_matrix_fodo), expected)

    def test_namedtuples(self):
        # just testing the plotting, these values are nonsense
        distribution = utils.PhasespaceDistribution(
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
        )
        distribution.plot()
        distribution.plot("h")
        distribution.plot("v")

        distribution = utils.PhasespaceDistribution(
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([0, 0.1, 0.2]),
        )
        distribution.plot()
        distribution.plot("h")
        distribution.plot("v")
        with self.assertRaises(ValueError):
            distribution.plot("asdada")

        transported_phasespace = utils.TransportedPhasespace(
            np.array([0, 1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
        )
        transported_phasespace.plot()
        transported_phasespace.plot(add_legend=True)
        transported_phasespace.plot("h")
        transported_phasespace.plot("v")
        with self.assertRaises(ValueError):
            transported_phasespace.plot("asdada")

        transported_distribution = utils.TransportedPhasespace(
            np.array([0, 1, 2]),
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([[0, 0, 0], [0, 0, 0]]),
        )
        transported_distribution.plot()
        transported_distribution.plot(add_legend=True)
        transported_distribution.plot("h")
        transported_distribution.plot("v")

        transported_twiss = utils.TransportedTwiss(
            np.array([0, 1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
        )
        transported_twiss.plot()
