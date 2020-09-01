from unittest import TestCase

import numpy as np

from accelerator.elements.dipole import Dipole
from accelerator.lattice import Lattice

n_dipoles = 6
perim = 1e3  # 1km
l_dipole = perim / n_dipoles
angle_dipole = 2 * np.pi / n_dipoles
rho_dipole = l_dipole / angle_dipole


class TestDipole(TestCase):
    def test_init(self):
        dipole = Dipole(angle_dipole, rho_dipole)
        assert dipole.length == l_dipole
        assert dipole._m_h == None
        assert dipole._m_v == None

    def test_transfer_matrix(self):
        dipole = Dipole(angle_dipole, rho_dipole)
        expected_transfer_matrix_h = np.array(
            [[-0.48338212, 0.91672664], [-0.83595446, -0.48338212]]
        )
        expected_transfer_matrix_v = np.array([[1, l_dipole], [0, 1]])
        out = dipole.transfer_matrix()
        assert np.allclose(out[0], expected_transfer_matrix_h)
        assert np.allclose(out[1], expected_transfer_matrix_v)
        assert np.allclose(dipole.m_h, out[0])
        assert np.allclose(dipole.m_v, out[1])

    def test_slice(self):
        dipole = Dipole(angle_dipole, rho_dipole)
        assert len(dipole.slice(10)) == 10
        assert isinstance(dipole.slice(10), Lattice)
        assert np.allclose(dipole.slice(10).m_h, dipole.m_h)
        assert np.allclose(dipole.slice(10).m_v, dipole.m_v)

    def test_repr(self):
        repr(Dipole(angle_dipole, rho_dipole))

    def test_dxztheta_ds(self):
        dipole = Dipole(angle_dipole, rho_dipole)
        # TODO: do the math to check this.
        dipole._dxztheta_ds(0, l_dipole)

    def test_plot(self):
        dipole = Dipole(angle_dipole, rho_dipole)
        dipole.plot()
