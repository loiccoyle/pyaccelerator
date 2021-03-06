from unittest import TestCase

import numpy as np

from pyaccelerator.elements.drift import Drift
from pyaccelerator.lattice import Lattice


class TestDrift(TestCase):
    def test_init(self):
        drift = Drift(1)
        assert drift.length == 1
        assert drift.name.startswith("drift_")
        drift = Drift(1, name="some_name")
        assert drift.name == "some_name"

    def test_transfer_matrix(self):
        drift = Drift(1)
        expected_transfer_matrix = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        m = drift._get_transfer_matrix()
        assert np.allclose(m, expected_transfer_matrix)
        assert np.allclose(drift.m, m)

    def test_slice(self):
        drift = Drift(1)
        assert len(drift.slice(10)) == 10
        assert isinstance(drift.slice(10), Lattice)
        assert np.allclose(drift.slice(10).m, drift.m)
        assert drift.slice(10)[0].name == drift.name + "_slice_0"

    def test_repr(self):
        repr(Drift(1))

    def test_dxztheta_ds(self):
        drift = Drift(1)
        assert np.allclose(drift._dxztheta_ds(0, 1), [1, 0, 0])

    def test_serialize(self):
        drift = Drift(2)
        dic = drift._serialize()
        assert dic["element"] == "Drift"
        assert dic["l"] == drift.l
        assert dic["name"] == drift.name

        # make sure that if the instance's attribute is changed
        # the serialization takes the new values.
        drift = Drift(2)
        drift.l = 5
        dic = drift._serialize()
        assert dic["element"] == "Drift"
        assert dic["l"] == drift.l
        assert dic["name"] == drift.name

    def test_copy(self):
        drift = Drift(1)
        copy = drift.copy()
        assert copy.l == drift.l
        assert copy.name == drift.name

        # make sure that if the instance's attribute is changed
        # copying takes the new values.
        drift = Drift(1)
        drift.l = 2
        copy = drift.copy()
        assert copy.l == drift.l
        assert copy.name == drift.name
