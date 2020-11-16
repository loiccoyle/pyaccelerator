from unittest import TestCase

import numpy as np

from pyaccelerator.constraints import (
    Constraints,
    FreeParameter,
    TargetDispersion,
    TargetPhasespace,
    TargetTwiss,
)
from pyaccelerator.elements.drift import Drift
from pyaccelerator.elements.quadrupole import QuadrupoleThin
from pyaccelerator.lattice import Lattice


class TestTargetPhasespace(TestCase):
    def test_init(self):
        target = TargetPhasespace("element", [10, 1, 0, 0, 0], [0, 1, 0, 0, 0])
        assert target.element == "element"
        assert target.initial == (0, 1, 0, 0, 0)
        np.allclose(target.value, np.array([10, 1, 0, 0, 0]))

        drift = Drift(1)
        target = TargetPhasespace(drift, [10, 1, 0, 0, 0], [0, 1, 0, 0, 0])
        assert target.element == drift.name
        assert target.initial == (0, 1, 0, 0, 0)
        np.allclose(target.value, np.array([10, 1, 0, 0, 0]))

    def test_repr(self):
        target = TargetPhasespace("element", [10, 1, 0, 0, 0], [0, 1, 0, 0, 0])
        repr(target)


class TestTargetTwiss(TestCase):
    def test_init(self):
        target = TargetTwiss("element", [1, 2, 3], plane="h")
        assert target.element == "element"
        np.allclose(target.value, np.array([1, 2, 3]))
        assert target.plane == "h"

        drift = Drift(1)
        target = TargetTwiss(drift, [1, 2, 3], plane="h")
        assert target.element == drift.name
        np.allclose(target.value, np.array([1, 2, 3]))
        assert target.plane == "h"

    def test_repr(self):
        target = TargetTwiss("element", [1, 2, 3], plane="h")
        repr(target)


class TestTargetDispersion(TestCase):
    def test_init(self):
        target = TargetDispersion("element", 1, plane="h")
        assert target.element == "element"
        assert target.value == 1
        assert target.plane == "h"

        drift = Drift(1)
        target = TargetDispersion(drift, 1, plane="h")
        assert target.element == drift.name
        assert target.value == 1
        assert target.plane == "h"

    def test_repr(self):
        target = TargetDispersion("element", 1, plane="h")
        repr(target)


class TestFreeParameter(TestCase):
    def test_init(self):
        param = FreeParameter("element", "attribute")
        assert param.element == "element"
        assert param.attribute == "attribute"

        drift = Drift(1)
        param = FreeParameter(drift, "l")
        assert param.element == drift.name
        assert param.attribute == "l"

    def test_repr(self):
        param = FreeParameter("element", "attribute")
        repr(param)


class TestConstraints(TestCase):
    def test_init(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        cons = Constraints(lat)
        assert cons.targets == []
        assert cons.free_parameters == []
        assert cons._lattice == lat

    def test_add_target(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        cons = Constraints(lat)
        target = TargetPhasespace("quadrupole", [10, 1, 0, 0, 0], [0, 1, 0, 0, 0])
        cons.add_target(target)
        assert cons.targets[0] == target

    def test_add_free_parameter(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        cons = Constraints(lat)
        cons.add_free_parameter("drift", "l")
        assert cons.free_parameters[0].element == "drift"
        assert cons.free_parameters[0].attribute == "l"

    def test_clear(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        cons = Constraints(lat)
        cons.add_free_parameter("drift", "l")
        target = TargetPhasespace("quadrupole", [10, 1, 0, 0, 0], [0, 1, 0, 0, 0])
        cons.add_target(target)
        cons.clear()
        assert cons.targets == []
        assert cons.free_parameters == []

    def test_repr(self):
        lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
        cons = Constraints(lat)
        repr(cons)

    def test_match(self):
        lat = Lattice([Drift(1)])
        with self.assertRaises(ValueError):
            lat.constraints.match()
        lat.constraints.add_free_parameter("drift", "l")
        with self.assertRaises(ValueError):
            # missing target
            lat.constraints.match()
        lat.constraints.clear()
        target = TargetPhasespace("drift", [2, None, None, None, None], [1, 1, 0, 0, 0])
        lat.constraints.add_target(target)
        with self.assertRaises(ValueError):
            # missing free parameter
            lat.constraints.match()

        # Compute drift length to reach a x coord of 10 meters:
        lat = Lattice([Drift(1)])
        lat.constraints.add_free_parameter(element="drift", attribute="l")
        target = TargetPhasespace(
            "drift", [10, None, None, None, None], [0, 1, 0, 0, 0]
        )
        lat.constraints.add_target(target)
        matched_lat, res = lat.constraints.match()
        assert res.success
        self.assertAlmostEqual(matched_lat[0].l, 10)
        # make sure the original did not change.
        assert lat[0].l == 1

        # Compute drift length to reach a x coord of 5 meters after the first
        # Drift:
        drift_0 = Drift(1)
        drift_1 = Drift(1)
        lat = Lattice([drift_0, drift_1])
        lat.constraints.add_free_parameter(drift_0, "l")
        target = TargetPhasespace(drift_0, [5, None, None, None, None], [0, 1, 0, 0, 0])
        lat.constraints.add_target(target)
        matched_lat, res = lat.constraints.match()
        assert res.success
        self.assertAlmostEqual(matched_lat[0].l, 5)
        self.assertAlmostEqual(matched_lat[1].l, 1)
        # make sure the original did not change.
        assert lat[0].l == 1
        assert lat[1].l == 1

        # Compute drift length to reach a x coord of 5 meters after the second
        # Drift with equal lengths of both Drifts:
        drift_0 = Drift(1)
        drift_1 = Drift(1)
        lat = Lattice([drift_0, drift_1])
        lat.constraints.add_free_parameter("drift", "l")
        target = TargetPhasespace(drift_1, [5, None, None, None, None], [0, 1, 0, 0, 0])
        lat.constraints.add_target(target)
        matched_lat, res = lat.constraints.match()
        assert res.success
        self.assertAlmostEqual(matched_lat[0].l, 2.5)
        self.assertAlmostEqual(matched_lat[1].l, 2.5)
        # make sure the original did not change.
        assert lat[0].l == 1
        assert lat[1].l == 1

        # Compute the quad strengths of a fodo to get a beta min of 0.5 meters
        lat = Lattice(
            [
                QuadrupoleThin(1.6, name="quad_f"),
                Drift(1),
                QuadrupoleThin(-0.8, name="quad_d"),
                Drift(1),
                QuadrupoleThin(1.6, name="quad_f"),
            ]
        )
        lat.constraints.add_free_parameter("quad_f", "f")
        lat.constraints.add_free_parameter("quad_d", "f")
        target = TargetTwiss("quad_d", [0.5, None, None], plane="h")
        lat.constraints.add_target(target)
        matched, opt_res = lat.constraints.match()
        s, beta, *_ = matched.twiss()
        self.assertAlmostEqual(min(beta), 0.5)
        assert opt_res.success

        # same thing but now with constraints such that the magnet strengths are
        # equal
        matched, opt_res = lat.constraints.match(
            constraints=({"type": "eq", "fun": lambda x: x[0] + 2 * x[1]})
        )
        s, beta, *_ = matched.twiss()
        self.assertAlmostEqual(min(beta), 0.5)
        assert matched[0].f == -2 * matched[2].f

    def test_repr(self):
        lat = Lattice([Drift(1)])
        repr(lat.constraints)
