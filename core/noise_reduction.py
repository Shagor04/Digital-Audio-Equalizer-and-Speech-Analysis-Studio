"""
core/noise_reduction.py
========================
Simple, explainable noise-reduction methods -- deliberately kept to
techniques that are standard undergraduate DSP (no ML), so every step can
be justified mathematically in a report/viva:

  - Low-pass / High-pass  : remove noise known to live above/below a cutoff
  - Notch (band-stop)      : remove a single interfering tone, e.g. 50 Hz
                              mains hum (Bangladesh mains frequency)
  - Spectral gating        : classic "noise gate in the frequency domain" --
                              estimate a noise floor via STFT, then attenuate
                              any time-frequency bin below that floor.
"""

import numpy as np
from scipy.signal import lfilter, stft, istft

from core.filters import lowpass_biquad_coeffs, highpass_biquad_coeffs, notch_coeffs


def remove_hum(x: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """
    Remove mains-frequency hum using a notch (band-reject) biquad centered
    at f0 (default 50 Hz -- the mains frequency in Bangladesh and most of
    the world; use 60 Hz for North America). Also notches the 2nd harmonic
    (2*f0) since hum from AC mains is rarely a pure sinusoid -- it usually
    has significant energy at harmonics too.
    """
    y = x.astype(float).copy()
    for harmonic in (1, 2):
        b, a = notch_coeffs(f0 * harmonic, Q, fs)
        y = lfilter(b, a, y)
    return y


def apply_lowpass(x: np.ndarray, cutoff_hz: float, fs: float, Q: float = 0.707) -> np.ndarray:
    """Remove noise/content above cutoff_hz (e.g. hiss, hum harmonics)."""
    b, a = lowpass_biquad_coeffs(cutoff_hz, Q, fs)
    return lfilter(b, a, x)


def apply_highpass(x: np.ndarray, cutoff_hz: float, fs: float, Q: float = 0.707) -> np.ndarray:
    """Remove noise/content below cutoff_hz (e.g. low rumble, DC drift, wind noise)."""
    b, a = highpass_biquad_coeffs(cutoff_hz, Q, fs)
    return lfilter(b, a, x)


def apply_bandstop(x: np.ndarray, f0: float, fs: float, Q: float = 5.0) -> np.ndarray:
    """Remove a narrow band of noise/interference centered at f0."""
    b, a = notch_coeffs(f0, Q, fs)
    return lfilter(b, a, x)


def spectral_gate(x: np.ndarray, fs: float, noise_reduce_db: float = 25.0,
                   noise_percentile: float = 50.0, margin_db: float = 10.0,
                   nperseg: int = 1024) -> np.ndarray:
    """
    Basic spectral gating (a simplified spectral-subtraction-family noise
    reducer), the same idea used in "denoise" tools:

      1. Take the STFT of the whole signal.
      2. For each frequency bin, estimate the noise floor as the median
         magnitude of that bin across time (assumes noise/silence occupies
         at least half the recording -- true for most speech/podcast
         material with pauses, less true for wall-to-wall music, which is
         a limitation worth noting in the report).
      3. Any time-frequency cell within margin_db of that noise floor is
         treated as noise and attenuated by noise_reduce_db; everything
         comfortably above the floor (i.e. likely real signal) passes
         through unchanged.
      4. Inverse STFT back to the time domain.

    This is intentionally simple (no smoothing/musical-noise suppression)
    to keep it explainable for a course report, at the cost of some
    "watery" artifacts on aggressive settings -- worth mentioning as a
    limitation in the report.
    """
    f, t, Z = stft(x, fs=fs, nperseg=nperseg)
    mag = np.abs(Z)
    phase = np.angle(Z)
    mag_db = 20 * np.log10(mag + 1e-12)

    noise_floor_db = np.percentile(mag_db, noise_percentile, axis=1, keepdims=True)
    threshold_db = noise_floor_db + margin_db
    gate_gain_linear = 10 ** (-noise_reduce_db / 20)

    mask = mag_db > threshold_db
    gain = np.where(mask, 1.0, gate_gain_linear)

    Z_clean = mag * gain * np.exp(1j * phase)
    _, y = istft(Z_clean, fs=fs, nperseg=nperseg)
    # istft can change length slightly; trim/pad back to match input
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


NOISE_REDUCTION_METHODS = {
    "none": lambda x, fs, **kw: x,
    "lowpass": lambda x, fs, cutoff=8000.0, **kw: apply_lowpass(x, cutoff, fs),
    "highpass": lambda x, fs, cutoff=80.0, **kw: apply_highpass(x, cutoff, fs),
    "bandstop": lambda x, fs, f0=50.0, Q=5.0, **kw: apply_bandstop(x, f0, fs, Q),
    "spectral_gate": lambda x, fs, noise_reduce_db=25.0, **kw: spectral_gate(x, fs, noise_reduce_db),
    "hum_removal": lambda x, fs, f0=50.0, Q=30.0, **kw: remove_hum(x, fs, f0, Q),
}
