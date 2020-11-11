from unittest import TestCase

import numpy as np

from accelerator.transfer_matrix import TransferMatrix


class TestTransferMatrix(TestCase):
    def test_init(self):
        t_m = TransferMatrix(np.identity(5))
        assert t_m.shape == (5, 5)
        assert hasattr(t_m, "twiss")
        assert hasattr(t_m, "h")
        assert hasattr(t_m, "v")
        with self.assertRaises(ValueError):
            TransferMatrix([1, 1, 1])
        with self.assertRaises(ValueError):
            TransferMatrix([[1, 1], [1, 2], [3, 3]])
        with self.assertRaises(ValueError):
            TransferMatrix([[1, 1], [1, 2]])

    def test_twiss(self):
        t_m = TransferMatrix(np.identity(5))
        assert t_m.twiss("h").shape == (3, 3)
        assert t_m.twiss("h").shape == (3, 3)
        assert np.allclose(t_m.twiss("h"), np.diag([1, 1, 1]))
        assert np.allclose(t_m.twiss("v"), np.diag([1, 1, 1]))

    # def test_twiss_solution(self):
    #     # twiss invariant in a drift:
    #     t_m = TransferMatrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    #     assert t_m.twiss_solution is None

    #     # twiss invariant in a FODO:
    #     L = 1
    #     f = 0.8
    #     FODO = [
    #         QuadrupoleThin(f * 2),
    #         Drift(L),
    #         QuadrupoleThin(-f),
    #         Drift(L),
    #         QuadrupoleThin(f * 2),
    #     ]
    #     FODO = Lattice(FODO)

    #     # From the course exmple notebook
    #     # FODO phase advance
    #     psi_cell = np.arccos(1.0 - L ** 2 / (2.0 * f ** 2))

    #     # Calc. periodic Twiss solutions
    #     beta_x0 = 2.0 * L / np.sin(psi_cell) * (1.0 + np.sin(psi_cell / 2.0))
    #     gamma_x0 = 1.0 / beta_x0
    #     beta_y0 = 2.0 * L / np.sin(psi_cell) * (1.0 - np.sin(psi_cell / 2.0))
    #     gamma_y0 = 1.0 / beta_y0
    #     self.assertAlmostEqual(beta_x0, FODO.m_h.twiss_solution[0][0])
    #     self.assertAlmostEqual(gamma_x0, FODO.m_h.twiss_solution[2][0])
    #     self.assertAlmostEqual(beta_y0, FODO.m_v.twiss_solution[0][0])
    #     self.assertAlmostEqual(gamma_y0, FODO.m_v.twiss_solution[2][0])
