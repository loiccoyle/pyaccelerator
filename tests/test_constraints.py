from unittest import TestCase

import numpy as np

from accelerator.constraints import Constraints, FreeParameter, Target
from accelerator.elements.drift import Drift
from accelerator.elements.quadrupole import QuadrupoleThin
from accelerator.lattice import Lattice


class TestTarget(TestCase):
    def test_init(self):
        target = Target("element", [1, 2, 3], twiss=[1, 0, 1], plane="h")
        assert target.twiss == (1, 0, 1)
        assert target.phasespace == None
        assert np.allclose(target.value, np.array([1, 2, 3]))

        target = Target("element", [10, 1, 0], phasespace=[0, 1, 0], plane="h")
        assert target.twiss == None
        assert target.phasespace == (0, 1, 0)
        assert np.allclose(target.value, np.array([10, 1, 0]))

        drift = Drift(1)
        target = Target(drift, [10, 1, 0], phasespace=[0, 1, 0], plane="h")
        assert target.element == drift.name
        assert target.phasespace == (0, 1, 0)
        assert target.twiss == None
        assert np.allclose(target.value, np.array([10, 1, 0]))

    def test_repr(self):
        target = Target("element", [1, 2, 3], [1, 0, 1], "h")
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
        cons.add_target("quadrupole", [10, 1, 0], phasespace=[0, 1, 0], plane="h")
        assert cons.targets[0].element == "quadrupole"
        assert np.allclose(cons.targets[0].value, np.array([10, 1, 0]))
        assert cons.targets[0].phasespace == (0, 1, 0)
        assert cons.targets[0].plane == "h"

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
        cons.add_target("quadrupole", [10, 1], [0, 1], plane="h")
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
        lat.constraints.add_target("drift", [2, None, None], phasespace=[1, 1, 0])
        with self.assertRaises(ValueError):
            # missing free parameter
            lat.constraints.match()

        lat.constraints.clear()
        with self.assertRaises(ValueError):
            lat.constraints.add_target("drift", [1, None, None], twiss="twiss_solution")

        # Compute drift length to reach a x coord of 10 meters:
        lat = Lattice([Drift(1)])
        lat.constraints.add_free_parameter(element="drift", attribute="l")
        lat.constraints.add_target(
            element="drift", value=[10, None, None], phasespace=[0, 1, 0], plane="h"
        )
        matched_lat, res = lat.constraints.match()
        assert res.success
        self.assertAlmostEqual(matched_lat[0].l, 10)
        # make sure the original did not change.
        assert lat[0].l == 1

        # Compute drift length to reach a beta of 5 meters from initial twiss
        # parameters [1, 0, 1]:
        lat = Lattice([Drift(1)])
        lat.constraints.add_free_parameter("drift", "l")
        lat.constraints.add_target("drift", [5, None, None], twiss=[1, 0, 1], plane="h")
        matched_lat, res = lat.constraints.match()
        assert res.success
        self.assertAlmostEqual(matched_lat[0].l, 2)
        # make sure the original did not change.
        assert lat[0].l == 1

        # Compute drift length to reach a x coord of 5 meters after the first
        # Drift:
        drift_0 = Drift(1)
        drift_1 = Drift(1)
        lat = Lattice([drift_0, drift_1])
        lat.constraints.add_free_parameter(drift_0, "l")
        lat.constraints.add_target(
            drift_0, [5, None, None], phasespace=[0, 1, 0], plane="h"
        )
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
        lat.constraints.add_target(
            drift_1, [5, None, None], phasespace=[0, 1, 0], plane="h"
        )
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
        lat.constraints.add_target(
            "quad_d", [0.5, None, None], twiss="solution", plane="h"
        )
        matched, opt_res = lat.constraints.match()
        beta, *_, s = matched.transport(twiss=matched.m_h.twiss_solution)
        self.assertAlmostEqual(min(beta), 0.5)

        # same thing but now with constraints such that the magnet strengths are
        # equal
        matched, opt_res = lat.constraints.match(
            constraints=({"type": "eq", "fun": lambda x: x[0] + 2*x[1]})
        )
        beta, *_, s = matched.transport(twiss=matched.m_h.twiss_solution)
        self.assertAlmostEqual(min(beta), 0.5)
        assert matched[0].f == -2*matched[2].f

    def test_repr(self):
        lat = Lattice([Drift(1)])
        repr(lat.constraints)
