"""
core/eq_engine.py
==================
Two equalizer engines built on top of core/filters.py:

  GraphicEQ    - a fixed bank of N band-pass FIR filters (parallel structure).
                 Each band has one knob: gain (dB). This is the "sliders"
                 EQ you see on a stereo / DAW.

  ParametricEQ - a cascade (series) of IIR biquad peaking/shelving filters.
                 Each band has three knobs: center freq, gain (dB), Q
                 (bandwidth). More flexible, closer to what's inside a
                 real mixing console.
"""

import numpy as np
from scipy.signal import lfilter

from core.filters import (
    design_bandpass_fir, convolve_fft,
    peaking_eq_coeffs, low_shelf_coeffs, high_shelf_coeffs,
)

# Standard ISO-ish center frequencies for a 10-band graphic EQ
TEN_BAND_CENTERS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
FIVE_BAND_CENTERS = [60, 250, 1000, 4000, 12000]


def _band_edges(centers, fs):
    """
    Turn a list of center frequencies into (low, high) edges per band,
    splitting each pair of neighbouring centers at their geometric mean
    (the natural crossover point on a log-frequency axis). The lowest
    band's low edge is 0 Hz (pure low-pass) and the highest band's high
    edge is the Nyquist frequency (pure high-pass).
    """
    centers = list(centers)
    edges = [0.0]
    for c1, c2 in zip(centers[:-1], centers[1:]):
        edges.append(np.sqrt(c1 * c2))  # geometric mean crossover
    edges.append(fs / 2)
    return list(zip(edges[:-1], edges[1:]))


class GraphicEQ:
    """
    Parallel bank-of-filters graphic equalizer.

    y[n] = sum_i  g_i * (x[n] * h_i[n])

    where h_i is the FIR band-pass kernel for band i and g_i is that
    band's linear gain (from its dB slider). Each band is filtered
    independently (this is why it's called a "parallel" structure) and
    the results are summed.
    """

    def __init__(self, fs: int, centers=None, numtaps: int = 401):
        self.fs = fs
        self.centers = centers if centers is not None else TEN_BAND_CENTERS
        self.numtaps = numtaps
        self.edges = _band_edges(self.centers, fs)
        self.kernels = [design_bandpass_fir(lo, hi, fs, numtaps) for lo, hi in self.edges]
        self.gains_db = [0.0] * len(self.centers)

    def set_gain(self, band_index: int, gain_db: float):
        self.gains_db[band_index] = gain_db

    def set_gains(self, gains_db):
        assert len(gains_db) == len(self.centers), "gains list must match number of bands"
        self.gains_db = list(gains_db)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply the full filter bank to a signal and return the equalized output."""
        y = np.zeros(len(x) + self.numtaps - 1)
        for h, g_db in zip(self.kernels, self.gains_db):
            g_lin = 10 ** (g_db / 20)
            if g_lin == 0:
                continue
            y += g_lin * convolve_fft(x, h)
        # trim the FIR group delay tail back to original length
        return y[: len(x)]

    def band_kernel_response(self, band_index: int, n_fft: int = 4096):
        """Return (freqs, magnitude_db) for one band's FIR kernel, for plotting."""
        h = self.kernels[band_index]
        H = np.fft.rfft(h, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1 / self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-12)
        return freqs, mag_db

    def composite_response(self, n_fft: int = 4096):
        """
        Return (freqs, magnitude_db) of the CURRENT gain-weighted sum of
        all bands -- i.e. the actual frequency response the audio will
        experience with the current slider settings.
        """
        freqs = np.fft.rfftfreq(n_fft, d=1 / self.fs)
        H_total = np.zeros(len(freqs), dtype=complex)
        for h, g_db in zip(self.kernels, self.gains_db):
            g_lin = 10 ** (g_db / 20)
            H_total += g_lin * np.fft.rfft(h, n=n_fft)
        mag_db = 20 * np.log10(np.abs(H_total) + 1e-12)
        return freqs, mag_db

    def composite_kernel(self):
        """
        Collapse the whole parallel filter bank into a single equivalent
        FIR impulse response: since every band shares the same kernel
        length and the bank is a parallel (summed) structure, the
        composite impulse response is just the gain-weighted SUM of the
        individual band kernels (linearity/superposition). This single
        FIR filter is what phase-response / group-delay / FIR-vs-IIR
        analysis operates on.
        """
        h_total = np.zeros(self.numtaps)
        for h, g_db in zip(self.kernels, self.gains_db):
            g_lin = 10 ** (g_db / 20)
            h_total += g_lin * h
        return h_total


class ParametricEQ:
    """
    Series (cascade) parametric equalizer built from IIR biquads.

    y[n] = Band_K( ... Band_2( Band_1( x[n] ) ) ... )

    Each band is one of:
      'peak'  - boost/cut around f0 with bandwidth set by Q
      'lowshelf'  - boost/cut everything below f0
      'highshelf' - boost/cut everything above f0
    """

    def __init__(self, fs: int):
        self.fs = fs
        self.bands = []  # list of dicts: {type, f0, gain_db, Q}

    def add_band(self, f0: float, gain_db: float = 0.0, Q: float = 1.0, kind: str = "peak"):
        self.bands.append({"type": kind, "f0": f0, "gain_db": gain_db, "Q": Q})
        return len(self.bands) - 1  # index of the new band

    def set_band(self, index: int, f0=None, gain_db=None, Q=None):
        b = self.bands[index]
        if f0 is not None:
            b["f0"] = f0
        if gain_db is not None:
            b["gain_db"] = gain_db
        if Q is not None:
            b["Q"] = Q

    def _coeffs(self, band):
        if band["type"] == "peak":
            return peaking_eq_coeffs(band["f0"], band["gain_db"], band["Q"], self.fs)
        elif band["type"] == "lowshelf":
            return low_shelf_coeffs(band["f0"], band["gain_db"], band["Q"], self.fs)
        elif band["type"] == "highshelf":
            return high_shelf_coeffs(band["f0"], band["gain_db"], band["Q"], self.fs)
        raise ValueError(f"Unknown band type {band['type']}")

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Cascade every band's IIR difference equation over the signal.
        Uses scipy's lfilter (a fast C implementation of the SAME
        Direct-Form-I recurrence as core.filters.apply_iir) since a
        pure-Python sample loop is too slow for a full audio track;
        core.filters.apply_iir is kept as the from-scratch reference
        for verifying correctness on short test signals.
        """
        y = x.astype(float).copy()
        for band in self.bands:
            b, a = self._coeffs(band)
            y = lfilter(b, a, y)
        return y

    def composite_response(self, n_fft: int = 4096):
        """Return (freqs, magnitude_db) of the cascaded response of all bands."""
        freqs = np.fft.rfftfreq(n_fft * 2, d=1 / self.fs)[: n_fft + 1]
        H_total = np.ones(len(freqs), dtype=complex)
        w = 2 * np.pi * freqs / self.fs
        for band in self.bands:
            b, a = self._coeffs(band)
            num = b[0] + b[1] * np.exp(-1j * w) + b[2] * np.exp(-2j * w)
            den = a[0] + a[1] * np.exp(-1j * w) + a[2] * np.exp(-2j * w)
            H_total *= num / den
        mag_db = 20 * np.log10(np.abs(H_total) + 1e-12)
        return freqs, mag_db

    def combined_coeffs(self):
        """
        Collapse the whole biquad cascade into a single equivalent
        (b, a) transfer function: since cascaded (series) LTI systems
        multiply their transfer functions, H_total(z) = Π H_i(z), the
        combined numerator/denominator polynomials are obtained by
        polynomial multiplication (= convolution of coefficient arrays)
        of each stage's b's together and a's together. If there are no
        bands, returns the identity system (b=[1], a=[1]).
        """
        b_total = np.array([1.0])
        a_total = np.array([1.0])
        for band in self.bands:
            b, a = self._coeffs(band)
            b_total = np.convolve(b_total, b)
            a_total = np.convolve(a_total, a)
        return b_total, a_total
