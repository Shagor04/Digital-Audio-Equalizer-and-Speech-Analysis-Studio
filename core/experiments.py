"""
core/experiments.py
====================
Controlled experiments for the report/viva -- each one varies ONE
parameter and measures the effect, which is much stronger evidence than
a single before/after screenshot:

  A. Bass boost sweep    : gain in {0, +3, +6, +9} dB -> spectrum, band
                            energy, RMS, peak
  B. Q factor sweep      : Q in {0.5, 1, 2, 5} at fixed f0/gain -> shows
                            how bandwidth narrows as Q increases
  C. FIR vs IIR           : same nominal boost, two implementations ->
                            magnitude, phase, group delay, processing time
  D. 50 Hz hum removal    : before/after spectrum + measured attenuation
  E. Noise reduction      : before/after spectrum + measured noise-floor
                            reduction
"""

import numpy as np

from core.eq_engine import GraphicEQ, ParametricEQ, TEN_BAND_CENTERS
from core.analysis import compute_magnitude_spectrum, compute_band_energy, compute_rms, compute_peak
from core.filter_analysis import compute_phase_response, compute_group_delay, measure_filter_performance, time_processing
from core.noise_reduction import remove_hum, spectral_gate
from core.audio_io import make_demo_music_signal


def experiment_bass_boost(x=None, fs=44100, gains_db=(0, 3, 6, 9)):
    """Sweep the lowest graphic-EQ band's gain and measure the effect."""
    if x is None:
        x, fs = make_demo_music_signal(bars=4)

    results = []
    for g in gains_db:
        eq = GraphicEQ(fs)
        gains = [0] * len(TEN_BAND_CENTERS)
        gains[0] = g
        eq.set_gains(gains)
        y = eq.process(x)
        freqs, mag = compute_magnitude_spectrum(y, fs)
        results.append({
            "gain_db": g,
            "freqs": freqs,
            "mag_db": mag,
            "band_energy": compute_band_energy(y, fs),
            "rms": compute_rms(y),
            "peak": compute_peak(y),
        })
    return {"x": x, "fs": fs, "results": results}


def experiment_q_factor(fs=44100, f0=1000.0, gain_db=6.0, q_values=(0.5, 1, 2, 5)):
    """Sweep Q at a fixed center frequency/gain and measure the resulting bandwidth."""
    results = []
    for q in q_values:
        eq = ParametricEQ(fs)
        eq.add_band(f0=f0, gain_db=gain_db, Q=q, kind="peak")
        freqs, mag = eq.composite_response()
        perf = measure_filter_performance(freqs, mag, f0, gain_db)
        results.append({
            "Q": q,
            "freqs": freqs,
            "mag_db": mag,
            "bandwidth_hz": perf["bandwidth_3db_hz"],
            "measured_gain_db": perf["measured_gain_db"],
        })
    return {"fs": fs, "f0": f0, "gain_db": gain_db, "results": results}


def experiment_fir_vs_iir(x=None, fs=44100, f0=1000.0, gain_db=6.0):
    """
    Compare a graphic-EQ band centered near f0 against a parametric peak at
    the same f0/gain: magnitude, phase, group delay, and measured
    processing time, for a genuinely matched comparison.
    """
    if x is None:
        x, fs = make_demo_music_signal(bars=4)

    # nearest graphic band to f0
    idx = int(np.argmin([abs(c - f0) for c in TEN_BAND_CENTERS]))
    geq = GraphicEQ(fs)
    gains = [0] * len(TEN_BAND_CENTERS)
    gains[idx] = gain_db
    geq.set_gains(gains)
    fir_b = geq.composite_kernel()
    fir_freqs_mag, fir_mag = geq.composite_response()
    fir_freqs_phase, fir_phase = compute_phase_response(fir_b, [1.0], fs)
    fir_freqs_gd, fir_gd = compute_group_delay(fir_b, [1.0], fs)
    fir_time_ms = time_processing(geq.process, x)

    peq = ParametricEQ(fs)
    peq.add_band(f0=f0, gain_db=gain_db, Q=1.0, kind="peak")
    iir_b, iir_a = peq.combined_coeffs()
    iir_freqs_mag, iir_mag = peq.composite_response()
    iir_freqs_phase, iir_phase = compute_phase_response(iir_b, iir_a, fs)
    iir_freqs_gd, iir_gd = compute_group_delay(iir_b, iir_a, fs)
    iir_time_ms = time_processing(peq.process, x)

    return {
        "fs": fs, "f0": f0, "gain_db": gain_db, "graphic_band_center": TEN_BAND_CENTERS[idx],
        "fir": {"freqs_mag": fir_freqs_mag, "mag_db": fir_mag,
                "freqs_phase": fir_freqs_phase, "phase": fir_phase,
                "freqs_gd": fir_freqs_gd, "gd_ms": fir_gd,
                "processing_time_ms": fir_time_ms, "num_coeffs": len(fir_b)},
        "iir": {"freqs_mag": iir_freqs_mag, "mag_db": iir_mag,
                "freqs_phase": iir_freqs_phase, "phase": iir_phase,
                "freqs_gd": iir_freqs_gd, "gd_ms": iir_gd,
                "processing_time_ms": iir_time_ms, "num_coeffs": len(iir_b) + len(iir_a)},
    }


def experiment_hum_removal(fs=44100, f0=50.0):
    """Inject known 50Hz (+2nd harmonic) hum into the demo track, remove it, measure attenuation."""
    x, fs = make_demo_music_signal(bars=4)
    t = np.arange(len(x)) / fs
    x_hum = x + 0.08 * np.sin(2 * np.pi * f0 * t) + 0.04 * np.sin(2 * np.pi * 2 * f0 * t)
    y = remove_hum(x_hum, fs, f0=f0, Q=30.0)

    def mag_at(sig, freq):
        n = len(sig)
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        idx = np.argmin(np.abs(freqs - freq))
        return 20 * np.log10(np.abs(X[idx]) + 1e-9)

    freqs_before, mag_before = compute_magnitude_spectrum(x_hum, fs)
    freqs_after, mag_after = compute_magnitude_spectrum(y, fs)
    return {
        "fs": fs, "f0": f0,
        "freqs_before": freqs_before, "mag_before": mag_before,
        "freqs_after": freqs_after, "mag_after": mag_after,
        "attenuation_fundamental_db": mag_at(x_hum, f0) - mag_at(y, f0),
        "attenuation_2nd_harmonic_db": mag_at(x_hum, 2 * f0) - mag_at(y, 2 * f0),
    }


def experiment_noise_reduction(fs=44100, noise_level=0.05):
    """Tone bursts + steady background noise -> spectral gate -> measure noise-floor reduction."""
    rng = np.random.default_rng(0)
    duration = 3.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    tone = np.zeros(n)
    tone[fs // 2: fs // 2 + fs] = 0.5 * np.sin(2 * np.pi * 300 * t[fs // 2: fs // 2 + fs])
    noise = noise_level * rng.standard_normal(n)
    x = tone + noise
    y = spectral_gate(x, fs, noise_reduce_db=25.0)

    silent_region = slice(0, fs // 3)  # a region with no tone, noise only
    noise_rms_before = compute_rms(x[silent_region])
    noise_rms_after = compute_rms(y[silent_region])
    reduction_db = 20 * np.log10(noise_rms_before / (noise_rms_after + 1e-12))

    freqs_before, mag_before = compute_magnitude_spectrum(x, fs)
    freqs_after, mag_after = compute_magnitude_spectrum(y, fs)
    return {
        "fs": fs,
        "freqs_before": freqs_before, "mag_before": mag_before,
        "freqs_after": freqs_after, "mag_after": mag_after,
        "noise_rms_before": noise_rms_before, "noise_rms_after": noise_rms_after,
        "noise_reduction_db": reduction_db,
    }
