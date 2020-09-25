from unittest import TestCase

import numpy as np

from accelerator.beam import Beam


class TestBeam(TestCase):
    def test_init(self):
        beam = Beam(emittance=3.5e-6)
        assert beam.emittance_h == 3.5e-6
        assert beam.emittance_v == 3.5e-6
        assert beam.gamma_relativistic == 6927.628011127815
        assert beam.beta_relativistic == 0.9999999895816034
        assert beam.geo_emittance_h == 5.05223437350036e-10
        assert beam.geo_emittance_v == 5.05223437350036e-10

        beam = Beam(emittance=(3.5e-6, 2.5e-6))
        assert beam.emittance_h == 3.5e-6
        assert beam.emittance_v == 2.5e-6
        assert beam.gamma_relativistic == 6927.628011127815
        assert beam.beta_relativistic == 0.9999999895816034
        assert beam.geo_emittance_h == 5.05223437350036e-10
        assert beam.geo_emittance_v == 3.608738838214543e-10

    def test_ellipse(self):
        beam = Beam()
        x, x_prime = beam.ellipse(twiss=[1, 0, 1])
        # almost equal because we scan a finite number of angles.
        self.assertAlmostEqual(x.max(), np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x.min(), -np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x_prime.max(), np.sqrt(beam.geo_emittance_h))
        self.assertAlmostEqual(x_prime.min(), -np.sqrt(beam.geo_emittance_h))

        x, x_prime = beam.ellipse(twiss=[1 / 2, 0, 2])
        # almost equal because we scan a finite number of angles.
        self.assertAlmostEqual(x.max(), np.sqrt(beam.geo_emittance_h * 1 / 2))
        self.assertAlmostEqual(x.min(), -np.sqrt(beam.geo_emittance_h * 1 / 2))
        self.assertAlmostEqual(x_prime.max(), np.sqrt(beam.geo_emittance_h * 2))
        self.assertAlmostEqual(x_prime.min(), -np.sqrt(beam.geo_emittance_h * 2))

    def test_match(self):
        beam = Beam(n_particles=int(1e6))
        x, x_prime = beam.match(twiss=[1, 0, 1])
        self.assertAlmostEqual(x.mean(), 0, places=3)
        self.assertAlmostEqual(x.std(), np.sqrt(beam.geo_emittance_h), places=3)
        self.assertAlmostEqual(x_prime.mean(), 0, places=3)
        self.assertAlmostEqual(x_prime.std(), np.sqrt(beam.geo_emittance_v), places=3)

        x, x_prime = beam.match(twiss=[1 / 2, 0, 2])
        self.assertAlmostEqual(x.mean(), 0, places=3)
        self.assertAlmostEqual(x.std(), np.sqrt(beam.geo_emittance_h * 1 / 2), places=3)
        self.assertAlmostEqual(x_prime.mean(), 0, places=3)
        self.assertAlmostEqual(
            x_prime.std(), np.sqrt(beam.geo_emittance_v * 2), places=3
        )

    def test_plot(self):
        beam = Beam()
        beam.plot([1, 0, 1])

    def test_repr(self):
        beam = Beam()
        repr(beam)
