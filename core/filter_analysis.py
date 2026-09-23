"""
core/filter_analysis.py
========================
Deeper filter-theory analysis, on top of the magnitude response already
computed in core/eq_engine.py:

  - Phase response      : H(f) = |H(f)| * e^(j*phase(f))
  - Group delay         : tau_g(w) = -d(phase)/d(w)  -- "how long each
                            frequency component is delayed by the filter"
  - IIR stability check : via pole locations (roots of the denominator)
  - FIR vs IIR structural + timing comparison

Both the Graphic EQ (parallel FIR bank) and Parametric EQ (cascaded IIR
biquads) reduce to a single (b, a) pair for this analysis:
  - GraphicEQ's composite impulse response is just the gain-weighted SUM
    of its FIR band kernels (linearity of the parallel structure) -> a
    single FIR filter with a = [1].
  - ParametricEQ's composite transfer function is the PRODUCT of each
    biquad's transfer function (series/cascade structure) -> multiply
    out via polynomial convolution of the b's and of the a's.
Once reduced to (b, a), scipy.signal.freqz / group_delay work uniformly
for both, which is why those helper methods live on the EQ classes
themselves (see core/eq_engine.py: composite_kernel / combined_coeffs).
"""

import time
import numpy as np
from scipy.signal import freqz, group_delay


def compute_phase_response(b, a, fs, n_fft=4096):
    """Return (freqs, phase_radians), unwrapped so it doesn't jump by 2*pi."""
    w, h = freqz(b, a, worN=n_fft, fs=fs)
    phase = np.unwrap(np.angle(h))
    return w, phase


def compute_group_delay(b, a, fs, n_fft=4096):
    """
    Return (freqs, group_delay_ms) = -d(phase)/d(omega), converted from
    samples to milliseconds. Physically: how many milliseconds each
    frequency component is delayed by passing through the filter. A
    linear-phase filter (symmetric FIR) has CONSTANT group delay across
    all frequencies -- a key theoretical property worth demonstrating.
    """
    w, gd_samples = group_delay((b, a), w=n_fft, fs=fs)
    gd_ms = gd_samples / fs * 1000.0
    return w, gd_ms


def check_iir_stability(a):
    """
    An IIR filter (or cascade) is stable iff every pole (root of the
    denominator polynomial a) lies strictly inside the unit circle,
    |z| < 1. Returns (is_stable, pole_magnitudes).
    """
    poles = np.roots(a)
    mags = np.abs(poles)
    return bool(np.all(mags < 1.0)), mags


def measure_filter_performance(freqs, mag_db, target_freq_hz, target_gain_db):
    """
    Compare a filter's DESIGNED target (e.g. "+6 dB peak at 1000 Hz") to
    what its actual frequency response delivers:
      - measured gain at the target frequency (nearest FFT bin)
      - measured peak location (where the response actually maximizes,
        which can differ slightly from the design target due to
        discretization / bilinear-transform warping)
      - -3dB bandwidth around that peak
      - gain error = measured - target
    """
    idx_target = int(np.argmin(np.abs(freqs - target_freq_hz)))
    measured_gain = float(mag_db[idx_target])
    gain_error = measured_gain - target_gain_db

    idx_peak = int(np.argmax(mag_db)) if target_gain_db >= 0 else int(np.argmin(mag_db))
    measured_peak_freq = float(freqs[idx_peak])
    peak_val = mag_db[idx_peak]

    # -3dB bandwidth around the peak (search outward from the peak index)
    half_level = peak_val - 3.0 if target_gain_db >= 0 else peak_val + 3.0
    lo = idx_peak
    while lo > 0 and ((mag_db[lo] > half_level) if target_gain_db >= 0 else (mag_db[lo] < half_level)):
        lo -= 1
    hi = idx_peak
    while hi < len(mag_db) - 1 and ((mag_db[hi] > half_level) if target_gain_db >= 0 else (mag_db[hi] < half_level)):
        hi += 1
    bandwidth_hz = float(freqs[hi] - freqs[lo]) if hi > lo else float("nan")

    return {
        "target_freq_hz": target_freq_hz,
        "target_gain_db": target_gain_db,
        "measured_gain_db": measured_gain,
        "gain_error_db": gain_error,
        "measured_peak_freq_hz": measured_peak_freq,
        "bandwidth_3db_hz": bandwidth_hz,
    }


def time_processing(process_fn, x, n_repeats=3):
    """Measure average processing time in milliseconds over n_repeats runs."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        process_fn(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times))


def compare_fir_vs_iir(graphic_eq, parametric_eq, x, fs):
    """
    Build the standard FIR-vs-IIR structural + measured comparison table
    for the report: structure, stability, phase linearity, coefficient
    count, and actual measured processing time on signal x.
    """
    fir_b = graphic_eq.composite_kernel()
    fir_time_ms = time_processing(graphic_eq.process, x)
    fir_freqs, fir_phase = compute_phase_response(fir_b, [1.0], fs)
    # a linear-phase (symmetric) FIR has phase that is exactly linear in f;
    # measure linearity via how well a straight line fits the unwrapped phase
    fit = np.polyfit(fir_freqs, fir_phase, 1)
    residual = fir_phase - np.polyval(fit, fir_freqs)
    fir_phase_linearity_error = float(np.std(residual))

    iir_b, iir_a = parametric_eq.combined_coeffs()
    iir_time_ms = time_processing(parametric_eq.process, x)
    is_stable, pole_mags = check_iir_stability(iir_a)

    return {
        "fir": {
            "structure": "Parallel bank of 10 band-pass FIR filters",
            "num_coeffs": len(fir_b),
            "stability": "Inherently stable (no poles)",
            "phase": "Linear (symmetric kernel)",
            "phase_linearity_std_rad": fir_phase_linearity_error,
            "processing_time_ms": fir_time_ms,
        },
        "iir": {
            "structure": f"Cascade of {len(parametric_eq.bands)} biquad section(s)",
            "num_coeffs": len(iir_b) + len(iir_a),
            "stability": f"{'Stable' if is_stable else 'UNSTABLE'} (max |pole|={pole_mags.max() if len(pole_mags) else 0:.4f})",
            "phase": "Nonlinear (typical for IIR)",
            "processing_time_ms": iir_time_ms,
        },
    }


def format_fir_vs_iir_table(comparison: dict) -> str:
    fir, iir = comparison["fir"], comparison["iir"]
    rows = [
        ("Structure", fir["structure"], iir["structure"]),
        ("Stability", fir["stability"], iir["stability"]),
        ("Phase", fir["phase"], iir["phase"]),
        ("Coefficients", str(fir["num_coeffs"]), str(iir["num_coeffs"])),
        ("Processing time", f"{fir['processing_time_ms']:.2f} ms", f"{iir['processing_time_ms']:.2f} ms"),
    ]
    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1]) for r in rows) + 2
    lines = [f"{'Property':<{w0}}{'FIR (Graphic EQ)':<{w1}}IIR (Parametric EQ)"]
    lines.append("-" * min(w0 + w1 + 24, 100))
    for name, f, i in rows:
        lines.append(f"{name:<{w0}}{f:<{w1}}{i}")
    return "\n".join(lines)
