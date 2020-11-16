from unittest import TestCase

import numpy as np

from pyaccelerator.elements import utils
from pyaccelerator.elements.drift import Drift


class TestUtils(TestCase):
    def test_straight_element(self):
        assert np.allclose(utils.straight_element(0, 1), np.array([1, 0, 0]))
        assert np.allclose(utils.straight_element(np.pi / 2, 1), np.array([0, 1, 0]))
        assert np.allclose(
            utils.straight_element(np.pi / 4, 1),
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
        )

    def test_bent_element(self):
        # TODO: do the math
        utils.bent_element(np.pi / 4, 1, 1)

    def test_deserialize(self):
        serial = {"element": "Drift", "l": 1}
        drift = utils.deserialize(serial)
        assert isinstance(drift, Drift)
        assert drift.length == serial["l"]
