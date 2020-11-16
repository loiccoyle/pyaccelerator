from unittest import TestCase

import numpy as np

from pyaccelerator.elements.sextupole import SextupoleThin


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
        expected_transfer_matrix = np.identity(5)
        m = sextupole._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(sextupole.m, m)

        # now the non linear part
        term = sextupole._non_linear_term(np.array([2, 0, 1, 0, 0]))
        assert np.allclose(term, np.array([[0, -1.5, 0, 2, 0]]))

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
