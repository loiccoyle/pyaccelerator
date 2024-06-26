import numpy as np
from scipy.fftpack import fft

PI2I = 2 * np.pi * complex(0, 1)


class HarmonicAnalysis:
    def __init__(self, samples: np.ndarray, zero_pad: bool = False, hann: bool = True):
        self._samples = samples
        self._compute_orbit()
        if zero_pad:
            self._pad_signal()
        self._length = len(self._samples)
        self._int_range = np.arange(self._length)
        self._hann_window = None
        if hann:
            self._hann_window = np.hanning(self._length)

    def laskar_method(self, num_harmonics: int):
        samples = self._samples[:]  # Copy the samples array.
        n = self._length
        coefficients = []
        frequencies = []
        for _ in range(num_harmonics):
            # Compute this harmonic frequency and coefficient.
            dft_data = fft(samples)
            frequency = self._jacobsen(dft_data)
            coefficient = HarmonicAnalysis._compute_coef(samples, frequency * n) / n

            # Store frequency and amplitude
            coefficients.append(coefficient)
            frequencies.append(frequency)

            # Subtract the found pure tune from the signal
            new_signal = coefficient * np.exp(PI2I * frequency * self._int_range)
            samples = samples - new_signal

        coefficients, frequencies = zip(
            *sorted(
                zip(coefficients, frequencies),
                key=lambda tup: np.abs(tup[0]),
                reverse=True,
            )
        )
        return frequencies, coefficients

    def _pad_signal(self):
        """Pads the signal with zeros to a "good" FFT size."""
        length = len(self._samples)
        # TODO Think proper pad size
        pad_length = (1 << (length - 1).bit_length()) - length
        # pad_length = 6600 - length
        self._samples = np.pad(self._samples, (0, pad_length), "constant")
        # self._samples = self._samples[:6000]

    def _jacobsen(self, dft_values):
        """This method interpolates the real frequency of the
        signal using the three highest peaks in the FFT.
        """
        k = np.argmax(np.abs(dft_values))
        n = self._length
        r = dft_values
        delta = np.tan(np.pi / n) / (np.pi / n)
        kp = (k + 1) % n
        km = (k - 1) % n
        delta = delta * np.real((r[km] - r[kp]) / (2 * r[k] - r[km] - r[kp]))
        return (k + delta) / n

    @staticmethod
    def _compute_coef(samples, kprime):
        """
        Computes the coefficient of the Discrete Time Fourier
        Transform corresponding to the given frequency (kprime).
        """
        n = len(samples)
        freq = kprime / n
        exponents = np.exp(-PI2I * freq * np.arange(n))
        coef = np.sum(exponents * samples)
        return coef

    def _compute_orbit(self):
        self.closed_orbit = np.mean(self._samples)
        self.closed_orbit_rms = np.std(self._samples)
        self.peak_to_peak = np.max(self._samples) - np.min(self._samples)
