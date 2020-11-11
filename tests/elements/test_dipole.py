from unittest import TestCase

import numpy as np

from accelerator.elements.dipole import Dipole, DipoleThin
from accelerator.lattice import Lattice

n_dipoles = 6
perim = 1e3  # 1km
l_dipole = perim / n_dipoles
angle_dipole = 2 * np.pi / n_dipoles
rho_dipole = l_dipole / angle_dipole


class TestDipole(TestCase):
    def test_init(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        assert dipole.length == l_dipole
        assert dipole.name.startswith("dipole_")
        dipole = Dipole(rho_dipole, angle_dipole, name="some_name")
        assert dipole.name == "some_name"

    def test_transfer_matrix(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        expected_transfer_matrix = np.array(
            [
                [0.5, 1.37832224e02, 0, 0, 7.95774715e01],
                [-5.44139809e-03, 0.5, 0, 0, 8.66025404e-01],
                [0, 0, 1, l_dipole, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        m = dipole._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(dipole.m, m)

    def test_slice(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        assert len(dipole.slice(10)) == 10
        assert isinstance(dipole.slice(10), Lattice)
        assert np.allclose(dipole.slice(10).m, dipole.m)
        assert dipole.slice(10)[0].name == dipole.name + "_slice_0"

    def test_repr(self):
        repr(Dipole(rho_dipole, angle_dipole))

    def test_dxztheta_ds(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        # TODO: do the math to check this.
        dipole._dxztheta_ds(0, l_dipole)

    def test_serialize(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        dic = dipole._serialize()
        assert dic["element"] == "Dipole"
        assert dic["rho"] == rho_dipole
        assert dic["theta"] == angle_dipole
        assert dic["name"] == dipole.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        dipole = Dipole(rho_dipole, angle_dipole)
        dipole.rho = 2 * dipole.rho
        dipole.theta = 2 * dipole.theta
        dic = dipole._serialize()
        assert dic["element"] == "Dipole"
        assert dic["rho"] == dipole.rho
        assert dic["theta"] == dipole.theta
        assert dic["name"] == dipole.name

    def test_copy(self):
        dipole = Dipole(rho_dipole, angle_dipole)
        copy = dipole.copy()
        assert copy.rho == dipole.rho
        assert copy.theta == dipole.theta
        assert copy.name == dipole.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        dipole = Dipole(rho_dipole, angle_dipole)
        dipole.rho = 2 * dipole.rho
        dipole.theta = 2 * dipole.theta
        copy = dipole.copy()
        assert copy.rho == dipole.rho
        assert copy.theta == dipole.theta
        assert copy.name == dipole.name


class TestDipoleThin(TestCase):
    def test_init(self):
        dipole = DipoleThin(angle_dipole)
        assert dipole.length == 0
        assert dipole.name.startswith("dipole_thin_")
        dipole = DipoleThin(angle_dipole, name="some_name")
        assert dipole.name == "some_name"

    def test_transfer_matrix(self):
        dipole = DipoleThin(angle_dipole)
        expected_transfer_matrix = np.identity(5)
        m = dipole._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(dipole.m, m)

    def test_repr(self):
        repr(DipoleThin(angle_dipole))

    def test_dxztheta_ds(self):
        dipole = DipoleThin(angle_dipole)
        # TODO: do the math to check this.
        dipole._dxztheta_ds(0, l_dipole)

    def test_serialize(self):
        dipole = DipoleThin(angle_dipole)
        dic = dipole._serialize()
        assert dic["element"] == "DipoleThin"
        assert dic["theta"] == angle_dipole
        assert dic["name"] == dipole.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        dipole = DipoleThin(angle_dipole)
        dipole.theta = 2 * dipole.theta
        dic = dipole._serialize()
        assert dic["element"] == "DipoleThin"
        assert dic["theta"] == dipole.theta
        assert dic["name"] == dipole.name

    def test_copy(self):
        dipole = DipoleThin(angle_dipole)
        copy = dipole.copy()
        assert copy.theta == dipole.theta
        assert copy.name == dipole.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        dipole = DipoleThin(angle_dipole)
        dipole.theta = 2 * dipole.theta
        copy = dipole.copy()
        assert copy.theta == dipole.theta
        assert copy.name == dipole.name
