"""
core/filters.py
================
Filter design primitives for the Audio Equalizer project (CSE220).

Two filter families are implemented, corresponding to the two EQ modes:

1. FIR band-pass filters (windowed-sinc design) -> used by the Graphic EQ.
   Built directly from the ideal low-pass impulse response (a sinc), which
   is exactly the material from the Fourier / filtering part of the course.

2. IIR biquad peaking filters (RBJ Audio-EQ-Cookbook design) -> used by the
   Parametric EQ. These come from bilinear-transforming an analog RLC
   resonator prototype, and are implemented as a difference equation
   (direct-form LTI system) -- i.e. exactly the y[n] = ... recurrence
   relation from the "LTI systems" part of the course.

Both a *naive* O(N*M) direct-convolution (for verification on short
signals) and an FFT-based convolution (for actually processing full-length
audio, using the Convolution Theorem) are provided.
"""

import numpy as np
from scipy.fft import next_fast_len


# ---------------------------------------------------------------------------
# 1. FIR filter design (windowed sinc) - used by the Graphic EQ
# ---------------------------------------------------------------------------

def _sinc_lowpass_kernel(cutoff_hz: float, fs: float, numtaps: int, window: str = "hamming") -> np.ndarray:
    """
    Design an ideal low-pass FIR filter via the windowed-sinc method.

    The ideal (brick-wall) low-pass filter has impulse response
        h_ideal[n] = 2*fc * sinc(2*fc*(n - M/2))
    where fc = cutoff_hz / fs (normalized cutoff) and M = numtaps - 1.

    This is infinite and non-causal, so it is truncated to `numtaps`
    samples and multiplied by a window function to reduce Gibbs ringing
    (spectral leakage) in the resulting frequency response.

    Parameters
    ----------
    cutoff_hz : cutoff frequency in Hz
    fs        : sampling rate in Hz
    numtaps   : filter length (odd number recommended -> Type I FIR, linear phase)
    window    : 'hamming', 'hann', or 'blackman'

    Returns
    -------
    h : ndarray, shape (numtaps,) -- the FIR filter coefficients
    """
    if numtaps % 2 == 0:
        numtaps += 1  # force odd length for a symmetric, linear-phase filter

    fc = cutoff_hz / fs  # normalized cutoff, 0 < fc < 0.5
    M = numtaps - 1
    n = np.arange(numtaps)

    # Ideal low-pass impulse response (numpy's sinc(x) = sin(pi x)/(pi x))
    h = 2 * fc * np.sinc(2 * fc * (n - M / 2))

    if window == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    elif window == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / M)
    elif window == "blackman":
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 0.08 * np.cos(4 * np.pi * n / M)
    else:
        raise ValueError(f"Unknown window '{window}'")

    h = h * w
    return h


def design_bandpass_fir(low_hz: float, high_hz: float, fs: float,
                         numtaps: int = 401, window: str = "hamming") -> np.ndarray:
    """
    Design a band-pass FIR filter as the *difference of two low-pass filters*:

        h_bp = h_lp(high_hz) - h_lp(low_hz)

    In the frequency domain, H_lp(high) passes [0, high] and H_lp(low)
    passes [0, low]; subtracting removes [0, low], leaving the band
    (low, high) -- this is a direct consequence of linearity of the
    Fourier transform (H_bp(w) = H_lp2(w) - H_lp1(w)).

    Special-cased at the spectrum edges:
      - low_hz <= 0        -> pure low-pass (this becomes the lowest EQ band)
      - high_hz >= fs/2     -> pure high-pass (this becomes the highest EQ band)
    """
    nyq = fs / 2
    high_hz = min(high_hz, nyq * 0.999)

    if low_hz <= 0:
        return _sinc_lowpass_kernel(high_hz, fs, numtaps, window)

    if high_hz >= nyq * 0.999 and low_hz > 0:
        # high-pass = all-pass (delta) minus low-pass at low_hz
        h_lp = _sinc_lowpass_kernel(low_hz, fs, numtaps, window)
        delta = np.zeros_like(h_lp)
        delta[len(delta) // 2] = 1.0
        return delta - h_lp

    h_hi = _sinc_lowpass_kernel(high_hz, fs, numtaps, window)
    h_lo = _sinc_lowpass_kernel(low_hz, fs, numtaps, window)
    return h_hi - h_lo


# ---------------------------------------------------------------------------
# 2. Convolution: naive (verification) and FFT-based (production)
# ---------------------------------------------------------------------------

def convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Direct (naive) linear convolution, O(N*M).
    y[n] = sum_k x[k] h[n-k]
    Only use this for SHORT signals (e.g. to verify against FFT convolution,
    or to plot a filter's step/impulse response) -- for a full song this
    would take far too long.
    """
    N, M = len(x), len(h)
    y = np.zeros(N + M - 1)
    for n in range(len(y)):
        total = 0.0
        k_lo = max(0, n - M + 1)
        k_hi = min(N - 1, n)
        for k in range(k_lo, k_hi + 1):
            total += x[k] * h[n - k]
        y[n] = total
    return y


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Linear convolution via the FFT, using the Convolution Theorem:
        x[n] * h[n]  <-->  X(f) . H(f)
    i.e. convolution in time = multiplication in frequency.
    We zero-pad both signals to length N+M-1 (to avoid circular-convolution
    wraparound) before multiplying in the frequency domain. This is what
    actually processes full-length audio in this project -- it is exact
    (not approximate) linear convolution, just computed efficiently.

    The padded length is rounded UP to the next "5-smooth" FFT-friendly
    size (via scipy.fft.next_fast_len) rather than used as-is: an FFT of
    a length with large prime factors can be an order of magnitude slower
    than one just slightly longer with only small prime factors -- this
    matters a lot here since audio signals are hundreds of thousands of
    samples long.
    """
    N, M = len(x), len(h)
    n_fft = next_fast_len(N + M - 1)
    X = np.fft.rfft(x, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    y = np.fft.irfft(X * H, n=n_fft)
    return y[: N + M - 1]


# ---------------------------------------------------------------------------
# 3. IIR biquad peaking filter (RBJ Audio-EQ-Cookbook) - used by Parametric EQ
# ---------------------------------------------------------------------------

def peaking_eq_coeffs(f0: float, gain_db: float, Q: float, fs: float):
    """
    Design a 2nd-order (biquad) peaking/notch filter: boosts or cuts a band
    centered at f0 with bandwidth controlled by Q, leaving frequencies far
    from f0 unaffected (unity gain).

    This is the discretized (bilinear-transform) version of an analog RLC
    band-pass resonator added on top of a flat response. Reference:
    Robert Bristow-Johnson, "Audio EQ Cookbook".

    Returns (b, a) difference-equation coefficients such that:
        a0*y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    (coefficients returned already normalized so a0 = 1)
    """
    A = 10 ** (gain_db / 40)          # sqrt of linear gain
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def low_shelf_coeffs(f0: float, gain_db: float, Q: float, fs: float):
    """RBJ low-shelf biquad: boosts/cuts everything BELOW f0 (bass shelf)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def high_shelf_coeffs(f0: float, gain_db: float, Q: float, fs: float):
    """RBJ high-shelf biquad: boosts/cuts everything ABOVE f0 (treble shelf)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def notch_coeffs(f0: float, Q: float, fs: float):
    """
    RBJ notch (band-reject) biquad: passes everything EXCEPT a narrow band
    around f0, which is driven to near-zero gain. Used for removing a single
    interfering tone -- e.g. 50 Hz mains hum (the mains frequency in
    Bangladesh and most of the world outside the Americas).

    Higher Q = narrower notch (removes less of the surrounding spectrum).
    A Q around 20-30 is typical for hum removal so it doesn't audibly dull
    nearby musical content.
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    b0 = 1
    b1 = -2 * cos_w0
    b2 = 1
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def lowpass_biquad_coeffs(f0: float, Q: float, fs: float):
    """RBJ 2nd-order Butterworth-ish low-pass biquad (Q=0.707 -> maximally
    flat / Butterworth response). Used by the noise-reduction module, as
    opposed to the FIR bandpass filters used inside the graphic EQ."""
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def highpass_biquad_coeffs(f0: float, Q: float, fs: float):
    """RBJ 2nd-order Butterworth-ish high-pass biquad."""
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)

    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_iir(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Apply an IIR difference equation directly (Direct Form I):
        y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
    Implemented as an explicit sample-by-sample recurrence so the LTI
    difference-equation structure is visible (this is what "IIR" means:
    the output feeds back into itself, unlike the FIR case above).
    """
    y = np.zeros(len(x) + 2)
    xpad = np.concatenate([np.zeros(2), x])
    for n in range(2, len(xpad)):
        y[n] = (b[0] * xpad[n] + b[1] * xpad[n - 1] + b[2] * xpad[n - 2]
                - a[1] * y[n - 1] - a[2] * y[n - 2])
    return y[2:]
