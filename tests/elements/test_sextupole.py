from unittest import TestCase

import numpy as np

from accelerator.elements.sextupole import SextupoleThin
from accelerator.lattice import Lattice


class TestSextupoleThin(TestCase):
    def test_init(self):
        sextupole = SextupoleThin(1)
        assert sextupole.length == 0
        assert sextupole.name.startswith("sextupole_thin")
        sextupole = SextupoleThin(1, name="sext_f")
        assert sextupole.name == "sext_f"

    def test_transfer_matrix(self):
        sextupole = SextupoleThin(1)
        # the linear part is the identity matrix
        expected_transfer_matrix_h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected_transfer_matrix_v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m_h, m_v = (
            sextupole._get_transfer_matrix_h(),
            sextupole._get_transfer_matrix_v(),
        )
        assert np.allclose(m_h, expected_transfer_matrix_h)
        assert np.allclose(m_v, expected_transfer_matrix_v)
        assert np.allclose(sextupole.m_h, m_h)
        assert np.allclose(sextupole.m_v, m_v)

        # now the non linear part
        matrix = sextupole._non_linear_term(np.array([1.0, 1.0, 0.0]))
        assert np.allclose(matrix, np.array([[0, -1 / 2, 0]]))

    def test_repr(self):
        repr(SextupoleThin(1))

    def test_serialize(self):
        sextupole = SextupoleThin(0.6)
        dic = sextupole._serialize()
        assert dic["element"] == "SextupoleThin"
        assert dic["k"] == sextupole.k
        assert dic["name"] == sextupole.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        sextupole = SextupoleThin(0.6)
        sextupole.f = 0.8
        dic = sextupole._serialize()
        assert dic["element"] == "SextupoleThin"
        assert dic["k"] == sextupole.k
        assert dic["name"] == sextupole.name

    def test_plot(self):
        sextupole = SextupoleThin(0.8)
        sextupole.plot()

    def test_copy(self):
        sextupole = SextupoleThin(1)
        copy = sextupole.copy()
        assert copy.k == sextupole.k
        assert copy.name == sextupole.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        sextupole = SextupoleThin(1)
        sextupole.k = 2
        copy = sextupole.copy()
        assert copy.k == sextupole.k
        assert copy.name == sextupole.name
