from unittest import TestCase

import numpy as np

from pyaccelerator.beam import Beam


class TestBeam(TestCase):
    def test_init(self):
        beam = Beam(emittance=3.5e-6)
        assert beam.emittance_h == 3.5e-6
        assert beam.emittance_v == 3.5e-6
        assert beam.gamma_relativistic == 6928.628011131436
        assert beam.beta_relativistic == 0.9999999895846106
        assert beam.geo_emittance_h == 5.05150519097116e-10
        assert beam.geo_emittance_v == 5.05150519097116e-10

        beam = Beam(emittance=(3.5e-6, 2.5e-6))
        assert beam.emittance_h == 3.5e-6
        assert beam.emittance_v == 2.5e-6
        assert beam.gamma_relativistic == 6928.628011131436
        assert beam.beta_relativistic == 0.9999999895846106
        assert beam.geo_emittance_h == 5.05150519097116e-10
        assert beam.geo_emittance_v == 3.608217993550829e-10

    def test_ellipse(self):
        beam = Beam()
        x, x_prime, y, y_prime, dp = beam.ellipse([1, 0, 1])
        # almost equal because we scan a finite number of angles.
        self.assertAlmostEqual(x.max(), np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x.min(), -np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x_prime.max(), np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x_prime.min(), -np.sqrt(beam.geo_emittance_h))
        assert dp.shape == x.shape
        self.assertAlmostEqual(y.max(), np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y.min(), -np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y_prime.max(), np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y_prime.min(), -np.sqrt(beam.geo_emittance_v))
        assert dp.shape == y.shape

        x, x_prime, y, y_prime, dp = beam.ellipse([1 / 2, 0, 2], twiss_v=[1, 0, 1])
        # almost equal because we scan a finite number of angles.
        self.assertAlmostEqual(x.max(), np.sqrt(beam.geo_emittance_h * 1 / 2))
        self.assertAlmostEqual(x.min(), -np.sqrt(beam.geo_emittance_h * 1 / 2))
        self.assertAlmostEqual(x_prime.max(), np.sqrt(beam.geo_emittance_h * 2))
        self.assertAlmostEqual(x_prime.min(), -np.sqrt(beam.geo_emittance_h * 2))
        assert dp.shape == x.shape
        self.assertAlmostEqual(y.max(), np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y.min(), -np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y_prime.max(), np.sqrt(beam.geo_emittance_v))
        self.assertAlmostEqual(y_prime.min(), -np.sqrt(beam.geo_emittance_v))
        assert dp.shape == y.shape

        with self.assertRaises(ValueError):
            # twiss clojure not met
            beam.ellipse([1, 0, 2])

    def test_match(self):
        beam = Beam(n_particles=int(1e6), sigma_energy=0.001)
        x, x_prime, y, y_prime, dp = beam.match([1, 0, 1])
        self.assertAlmostEqual(x.mean(), 0, places=3)
        self.assertAlmostEqual(x.std(), np.sqrt(beam.geo_emittance_h), places=3)
        self.assertAlmostEqual(x_prime.mean(), 0, places=3)
        self.assertAlmostEqual(x_prime.std(), np.sqrt(beam.geo_emittance_h), places=3)
        assert dp.shape == x.shape
        self.assertAlmostEqual(y.mean(), 0, places=3)
        self.assertAlmostEqual(y.std(), np.sqrt(beam.geo_emittance_v), places=3)
        self.assertAlmostEqual(y_prime.mean(), 0, places=3)
        self.assertAlmostEqual(y_prime.std(), np.sqrt(beam.geo_emittance_v), places=3)
        assert dp.shape == y.shape
        self.assertAlmostEqual(dp.std(), beam.sigma_p / beam.p, places=3)

        x, x_prime, y, y_prime, dp = beam.match([1 / 2, 0, 2], twiss_v=[1, 0, 1])
        self.assertAlmostEqual(x.mean(), 0, places=3)
        self.assertAlmostEqual(x.std(), np.sqrt(beam.geo_emittance_h * 1 / 2), places=3)
        self.assertAlmostEqual(x_prime.mean(), 0, places=3)
        self.assertAlmostEqual(
            x_prime.std(), np.sqrt(beam.geo_emittance_h * 2), places=3
        )
        assert dp.shape == x.shape
        self.assertAlmostEqual(y.mean(), 0, places=3)
        self.assertAlmostEqual(y.std(), np.sqrt(beam.geo_emittance_v), places=3)
        self.assertAlmostEqual(y_prime.mean(), 0, places=3)
        self.assertAlmostEqual(y_prime.std(), np.sqrt(beam.geo_emittance_v), places=3)
        assert dp.shape == y.shape
        self.assertAlmostEqual(dp.std(), beam.sigma_p / beam.p, places=3)
        self.assertAlmostEqual(dp.std(), beam.sigma_p / beam.p, places=3)

        with self.assertRaises(ValueError):
            beam.match([1, 0, 2])

    def test_repr(self):
        beam = Beam()
        repr(beam)
