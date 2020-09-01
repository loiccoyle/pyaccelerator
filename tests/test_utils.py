from unittest import TestCase

import numpy as np

from accelerator import utils
from accelerator.lattice import Lattice
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import Quadrupole


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
            utils.to_twiss([1, 2, 3, 4])

    def test_to_phase_coord(self):
        assert np.allclose(utils.to_phase_coord([1, 2]), np.array([1, 2]).reshape(2, 1))
        with self.assertRaises(ValueError):
            utils.to_phase_coord([1, 2, 3])

    def test_complete_twiss(self):
        assert utils.complete_twiss(beta=1, alpha=0, gamma=1) == (1, 0, 1)
        assert utils.complete_twiss(beta=1, alpha=0, gamma=None) == (1, 0, 1)
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

    def test_compute_invariant(self):
        transfer_matrix = np.array([[1, 1], [0, 1]])
        np.testing.assert_almost_equal(
            utils.compute_invariant(transfer_matrix), [[1, -1], [0, 0]]
        )
        # no invariants
        with self.assertRaises(ValueError):
            utils.compute_invariant(np.array([[1, 1], [1, 1]]))

    def test_compute_twiss_invariant(self):
        with self.assertRaises(ValueError):
            utils.compute_twiss_invariant(np.array([[1, 2], [3, 4]]))
        transfer_matrix = utils.compute_m_twiss(np.array([[1, 1], [0, 1]]))
        with self.assertRaises(ValueError):
            utils.compute_twiss_invariant(transfer_matrix)

        # TODO: make sure this is correct
        transfer_matrix_fodo = Lattice(
            [
                Quadrupole(2 * 0.8),
                Drift(1),
                Quadrupole(-0.8),
                Drift(1),
                Quadrupole(2 * 0.8),
            ]
        ).m_h.twiss
        expected = np.array([[3.33066560e00], [1.11528141e-16], [3.00240288e-01]])
        assert np.allclose(
            utils.compute_twiss_invariant(transfer_matrix_fodo), expected
        )
