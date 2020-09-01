from unittest import TestCase

import numpy as np

from accelerator.transfer_matrix import TransferMatrix, TwissTransferMatrix
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import Quadrupole
from accelerator.lattice import Lattice


class TestTransferMatrix(TestCase):
    def test_init(self):
        t_m = TransferMatrix([[1, 0], [0, 1]])
        assert t_m.shape == (2, 2)
        assert hasattr(t_m, "twiss")
        with self.assertRaises(ValueError):
            TransferMatrix([1, 1, 1])
        with self.assertRaises(ValueError):
            TransferMatrix([[1, 1], [1, 2, 3]])
        with self.assertRaises(ValueError):
            TransferMatrix([[1, 1, 1], [1, 2, 3], [3, 3, 3]])
        with self.assertRaises(ValueError):
            TransferMatrix([[1, 1], [1, 2], [3, 3]])

    def test_twiss(self):
        t_m = TransferMatrix([[1, 0], [0, 1]])
        assert t_m.twiss.shape == (3, 3)
        assert np.allclose(t_m.twiss, np.diag([1, 1, 1]))


class TestTwissTransferMatrix(TestCase):
    def test_init(self):
        t_m = TwissTransferMatrix([[1, -2, 1], [0, 1, -1], [0, 0, 1]])
        assert t_m.shape == (3, 3)
        assert hasattr(t_m, "invariant")
        with self.assertRaises(ValueError):
            TwissTransferMatrix([1, 1, 1])
        with self.assertRaises(ValueError):
            TwissTransferMatrix([[1, 1], [1, 2, 3]])
        with self.assertRaises(ValueError):
            TwissTransferMatrix([[1, 1], [1, 2]])
        with self.assertRaises(ValueError):
            TwissTransferMatrix([[1, 1], [1, 2], [3, 3]])

    def test_invariant(self):
        # twiss invariant in a drift:
        t_m = TwissTransferMatrix([[1, -2, 1], [0, 1, -1], [0, 0, 1]])
        assert t_m.invariant is None

        # twiss invariant in a FODO:
        L = 1
        f = 0.8
        FODO = [
            Quadrupole(f * 2),
            Drift(L),
            Quadrupole(-f),
            Drift(L),
            Quadrupole(f * 2),
        ]
        FODO = Lattice(FODO)

        # From the course exmple notebook
        # FODO phase advance
        psi_cell = np.arccos(1. - L**2 / (2.*f**2))

        # Calc. periodic Twiss solutions
        beta_x0 = 2.*L / np.sin(psi_cell) * (1. + np.sin(psi_cell / 2.))
        gamma_x0 = 1./beta_x0
        beta_y0 = 2.*L / np.sin(psi_cell) * (1. - np.sin(psi_cell / 2.))
        gamma_y0 = 1./beta_y0
        self.assertAlmostEqual(beta_x0, FODO.m_h.twiss.invariant[0][0])
        self.assertAlmostEqual(gamma_x0, FODO.m_h.twiss.invariant[2][0])
        self.assertAlmostEqual(beta_y0, FODO.m_v.twiss.invariant[0][0])
        self.assertAlmostEqual(gamma_y0, FODO.m_v.twiss.invariant[2][0])
