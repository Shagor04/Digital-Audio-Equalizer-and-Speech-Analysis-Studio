"""
core/analysis.py
=================
Quantitative signal analysis -- turns "the audio sounds different" into
actual numbers, for the report and for viva questions.

Metrics implemented (all standard DSP feature-extraction quantities):
  - RMS (root-mean-square level)
  - Peak amplitude
  - Energy  (sum of x[n]^2 -- Parseval's theorem connects this to the
    frequency-domain energy too, worth mentioning in the report)
  - Spectral centroid (the "center of mass" of the spectrum -- a rough
    proxy for perceived brightness)
  - Dominant frequency (the frequency bin with the most energy)
  - Zero-crossing rate (how often the signal changes sign -- a cheap
    proxy for "how much high-frequency content" a signal has)
"""

import numpy as np


def compute_rms(x: np.ndarray) -> float:
    """RMS = sqrt( (1/N) * sum(x[n]^2) )"""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def compute_peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def compute_energy(x: np.ndarray) -> float:
    """Energy = sum(x[n]^2) (Parseval: equals (1/N)*sum|X[k]|^2 in the DFT domain)."""
    return float(np.sum(x.astype(np.float64) ** 2))


def compute_magnitude_spectrum(x: np.ndarray, fs: float, n_fft: int = None):
    """Return (freqs, magnitude_db) -- single-sided FFT magnitude spectrum in dB."""
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    mag_db = 20 * np.log10(np.abs(X) / len(x) + 1e-12)
    return freqs, mag_db


def compute_spectral_centroid(x: np.ndarray, fs: float) -> float:
    """
    Spectral centroid: C = sum_k( f_k * |X[k]| ) / sum_k( |X[k]| )
    A rough "center of mass" of the spectrum -- higher = brighter/more
    treble-heavy sound. Computed on LINEAR magnitude (not dB), since the
    centroid is an energy-weighted average and dB would distort the weighting.
    """
    n_fft = int(2 ** np.ceil(np.log2(len(x))))
    X = np.abs(np.fft.rfft(x, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    denom = np.sum(X) + 1e-12
    return float(np.sum(freqs * X) / denom)


def compute_dominant_frequency(x: np.ndarray, fs: float) -> float:
    """The frequency bin with the largest magnitude (excluding DC)."""
    n_fft = int(2 ** np.ceil(np.log2(len(x))))
    X = np.abs(np.fft.rfft(x, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    X[0] = 0  # ignore DC
    idx = np.argmax(X)
    return float(freqs[idx])


def compute_zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of samples where the signal changes sign -- a cheap proxy
    for how much high-frequency content is present."""
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = np.sum(signs[:-1] != signs[1:])
    return float(crossings / len(x))


def summarize(x: np.ndarray, fs: float) -> dict:
    """Compute the full metric set at once (used by both CLI and GUI)."""
    return {
        "duration_s": len(x) / fs,
        "sample_rate": fs,
        "rms": compute_rms(x),
        "peak": compute_peak(x),
        "energy": compute_energy(x),
        "spectral_centroid_hz": compute_spectral_centroid(x, fs),
        "dominant_freq_hz": compute_dominant_frequency(x, fs),
        "zero_crossing_rate": compute_zero_crossing_rate(x),
    }


def format_comparison_table(before: dict, after: dict) -> str:
    """Pretty-print a before/after metrics table for the terminal or a report."""
    rows = [
        ("Duration (s)", f"{before['duration_s']:.2f}", f"{after['duration_s']:.2f}"),
        ("RMS", f"{before['rms']:.4f}", f"{after['rms']:.4f}"),
        ("Peak", f"{before['peak']:.4f}", f"{after['peak']:.4f}"),
        ("Energy", f"{before['energy']:.2f}", f"{after['energy']:.2f}"),
        ("Spectral Centroid (Hz)", f"{before['spectral_centroid_hz']:.1f}", f"{after['spectral_centroid_hz']:.1f}"),
        ("Dominant Freq (Hz)", f"{before['dominant_freq_hz']:.1f}", f"{after['dominant_freq_hz']:.1f}"),
        ("Zero-Crossing Rate", f"{before['zero_crossing_rate']:.4f}", f"{after['zero_crossing_rate']:.4f}"),
    ]
    w0 = max(len(r[0]) for r in rows) + 2
    lines = [f"{'Metric':<{w0}}{'Original':>14}{'Processed':>14}"]
    lines.append("-" * (w0 + 28))
    for name, b, a in rows:
        lines.append(f"{name:<{w0}}{b:>14}{a:>14}")
    return "\n".join(lines)


DEFAULT_BANDS = [
    ("Bass", 20, 250),
    ("Low-mid", 250, 500),
    ("Mid", 500, 2000),
    ("High-mid", 2000, 4000),
    ("Treble", 4000, 20000),
]


def compute_band_energy(x: np.ndarray, fs: float, bands=None) -> dict:
    """
    Fraction of total spectral energy falling in each named frequency band
    (Parseval's theorem: energy in the frequency domain via |X[k]|^2 sums
    to the same total as time-domain energy, so this is a legitimate
    energy decomposition, not just a magnitude-spectrum eyeball estimate).
    Returns {band_name: percentage} summing to ~100.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    n_fft = int(2 ** np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    power = np.abs(X) ** 2
    total = np.sum(power) + 1e-12

    result = {}
    for name, lo, hi in bands:
        hi = min(hi, fs / 2)
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(np.sum(power[mask]) / total * 100.0)
    return result


def format_band_energy_bars(before: dict, after: dict, bar_width: int = 20) -> str:
    """ASCII bar-chart comparison of band energy, for terminal/report output."""
    lines = [f"{'Band':<10}{'Before':>8}   {'':<{bar_width}}   {'After':>8}"]
    for name in before:
        b, a = before[name], after[name]
        bar_b = "#" * int(round(b / 100 * bar_width))
        bar_a = "#" * int(round(a / 100 * bar_width))
        lines.append(f"{name:<10}{b:>6.1f}%  {bar_b:<{bar_width}}  {a:>6.1f}%  {bar_a}")
    return "\n".join(lines)
