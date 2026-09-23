"""
visualization/plots.py
=======================
Plotting helpers used to verify and demo the equalizer:
  - frequency response of individual bands / composite EQ curve
  - waveform before vs after
  - spectrogram before vs after
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_graphic_eq_bands(eq, n_fft=4096, savepath=None):
    """Plot each band's individual FIR response (thin lines) plus the
    current gain-weighted composite response (thick line) of a GraphicEQ."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(eq.centers):
        freqs, mag = eq.band_kernel_response(i, n_fft)
        ax.plot(freqs, mag, linewidth=1, alpha=0.5, label=f"{c} Hz band")
    freqs, mag_total = eq.composite_response(n_fft)
    ax.plot(freqs, mag_total, linewidth=2.5, color="black", label="Composite (current gains)")
    ax.set_xscale("log")
    ax.set_xlim(20, eq.fs / 2)
    ax.set_ylim(-40, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Graphic EQ: per-band response + composite")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_parametric_eq_response(eq, n_fft=4096, savepath=None):
    freqs, mag = eq.composite_response(n_fft)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs, mag, linewidth=2, color="darkorange")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for band in eq.bands:
        ax.axvline(band["f0"], color="gray", linewidth=0.5, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlim(20, eq.fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Parametric EQ: cascaded response")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_waveform_comparison(x, y, fs, savepath=None):
    t = np.arange(len(x)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t, x, linewidth=0.5, color="steelblue")
    axes[0].set_title("Original waveform")
    axes[1].plot(np.arange(len(y)) / fs, y, linewidth=0.5, color="crimson")
    axes[1].set_title("Equalized waveform")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_spectrogram_comparison(x, y, fs, savepath=None):
    """
    Plot original vs equalized spectrograms on a SHARED color scale (vmin/vmax
    computed jointly). Without this, matplotlib normalizes each spectrogram's
    dB range independently, which visually hides real EQ differences.
    """
    # compute both spectrograms first so we can share color limits
    Sx, freqs, times = plt.mlab.specgram(x, Fs=fs, NFFT=1024, noverlap=512)[0:3]
    Sy, _, _ = plt.mlab.specgram(y, Fs=fs, NFFT=1024, noverlap=512)[0:3]
    Sx_db = 10 * np.log10(Sx + 1e-12)
    Sy_db = 10 * np.log10(Sy + 1e-12)
    vmin = min(Sx_db.min(), Sy_db.min())
    vmax = max(Sx_db.max(), Sy_db.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    im0 = axes[0].pcolormesh(times, freqs, Sx_db, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
    axes[0].set_title("Original spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    im1 = axes[1].pcolormesh(times, freqs, Sy_db, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
    axes[1].set_title("Equalized spectrogram")
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(im1, ax=axes, label="dB", fraction=0.03, pad=0.02)
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_spectrum_comparison(x, y, fs, savepath=None):
    """
    Overlay the FFT magnitude spectrum of the original (x) vs equalized (y)
    signal on a log-frequency axis -- this is the most direct visual proof
    that the equalizer is actually reshaping the frequency content (as
    opposed to the waveform plot, which mostly shows amplitude/envelope).
    """
    from core.analysis import compute_magnitude_spectrum
    freqs_x, mag_x = compute_magnitude_spectrum(x, fs)
    freqs_y, mag_y = compute_magnitude_spectrum(y, fs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs_x, mag_x, label="Original", color="steelblue", linewidth=1, alpha=0.85)
    ax.plot(freqs_y, mag_y, label="Equalized", color="crimson", linewidth=1, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(20, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Before / After magnitude spectrum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_analysis_dashboard(x, y, fs, metrics_before=None, metrics_after=None, savepath=None):
    """
    A single combined figure for the report: waveform, spectrum, and
    spectrogram comparisons stacked together, optionally with a metrics
    table at the bottom if metrics dicts are supplied
    (see core.analysis.summarize).
    """
    from core.analysis import compute_magnitude_spectrum

    n_rows = 4 if metrics_before else 3
    fig = plt.figure(figsize=(11, 3.2 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[1] * 3 + ([0.6] if metrics_before else []))

    # waveform
    ax0 = fig.add_subplot(gs[0])
    t = np.arange(len(x)) / fs
    ax0.plot(t, x, linewidth=0.4, color="steelblue", alpha=0.8, label="Original")
    ax0.plot(np.arange(len(y)) / fs, y, linewidth=0.4, color="crimson", alpha=0.6, label="Equalized")
    ax0.set_title("Waveform (overlaid)")
    ax0.set_xlabel("Time (s)")
    ax0.legend(fontsize=8)

    # spectrum
    ax1 = fig.add_subplot(gs[1])
    freqs_x, mag_x = compute_magnitude_spectrum(x, fs)
    freqs_y, mag_y = compute_magnitude_spectrum(y, fs)
    ax1.plot(freqs_x, mag_x, color="steelblue", linewidth=1, label="Original")
    ax1.plot(freqs_y, mag_y, color="crimson", linewidth=1, label="Equalized")
    ax1.set_xscale("log")
    ax1.set_xlim(20, fs / 2)
    ax1.set_title("Magnitude spectrum")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # spectrogram (equalized only, to save space -- full comparison is in
    # plot_spectrogram_comparison)
    ax2 = fig.add_subplot(gs[2])
    ax2.specgram(y, Fs=fs, NFFT=1024, noverlap=512, cmap="magma")
    ax2.set_title("Spectrogram (equalized)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    if metrics_before:
        ax3 = fig.add_subplot(gs[3])
        ax3.axis("off")
        rows = [
            ("RMS", f"{metrics_before['rms']:.4f}", f"{metrics_after['rms']:.4f}"),
            ("Peak", f"{metrics_before['peak']:.4f}", f"{metrics_after['peak']:.4f}"),
            ("Spectral Centroid (Hz)", f"{metrics_before['spectral_centroid_hz']:.1f}",
             f"{metrics_after['spectral_centroid_hz']:.1f}"),
            ("Dominant Freq (Hz)", f"{metrics_before['dominant_freq_hz']:.1f}",
             f"{metrics_after['dominant_freq_hz']:.1f}"),
            ("Zero-Crossing Rate", f"{metrics_before['zero_crossing_rate']:.4f}",
             f"{metrics_after['zero_crossing_rate']:.4f}"),
        ]
        table = ax3.table(cellText=[[r[0], r[1], r[2]] for r in rows],
                           colLabels=["Metric", "Original", "Equalized"],
                           loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_phase_and_group_delay(freqs_phase, phase, freqs_gd, group_delay_ms, title_prefix="", savepath=None):
    """
    Two stacked panels: unwrapped phase response on top, group delay
    (in milliseconds) below -- the two are related by
    tau_g(w) = -d(phase)/d(w), so plotting them together makes that
    relationship visible (flat group delay <=> straight-line phase).
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(freqs_phase, phase, color="#8b7cf6", linewidth=1.5)
    axes[0].set_ylabel("Phase (radians)")
    axes[0].set_title(f"{title_prefix} Phase response")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs_gd, group_delay_ms, color="#22d3c9", linewidth=1.5)
    axes[1].set_ylabel("Group delay (ms)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title(f"{title_prefix} Group delay")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_band_energy(before: dict, after: dict, savepath=None):
    """Horizontal bar chart comparing per-band spectral energy before vs after EQ."""
    names = list(before.keys())
    b_vals = [before[n] for n in names]
    a_vals = [after[n] for n in names]
    y = np.arange(len(names))
    h = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y + h / 2, b_vals, height=h, label="Original", color="steelblue")
    ax.barh(y - h / 2, a_vals, height=h, label="Equalized", color="crimson")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Share of total spectral energy (%)")
    ax.set_title("Frequency-band energy distribution")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_experiment_bass_boost(data, savepath=None):
    """Overlaid spectra + band-energy/RMS/peak progression across gain values."""
    results = data["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cmap = plt.cm.Oranges
    for i, res in enumerate(results):
        color = cmap(0.35 + 0.6 * i / max(len(results) - 1, 1))
        axes[0].plot(res["freqs"], res["mag_db"], color=color, linewidth=1,
                     label=f"{res['gain_db']:+.0f} dB")
    axes[0].set_xscale("log")
    axes[0].set_xlim(20, data["fs"] / 2)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Experiment A: Bass-boost gain sweep -- spectrum")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    gains = [r["gain_db"] for r in results]
    bass_pct = [r["band_energy"]["Bass"] for r in results]
    rms = [r["rms"] for r in results]
    ax2 = axes[1]
    ax2.plot(gains, bass_pct, "o-", color="crimson", label="Bass energy share (%)")
    ax2.set_xlabel("Bass-band gain setting (dB)")
    ax2.set_ylabel("Bass energy share (%)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax3 = ax2.twinx()
    ax3.plot(gains, rms, "s--", color="steelblue", label="RMS")
    ax3.set_ylabel("RMS", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_title("Bass energy share & RMS vs gain")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_experiment_q_factor(data, savepath=None):
    """Overlaid magnitude response for each Q + measured bandwidth vs Q."""
    results = data["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    cmap = plt.cm.Purples
    for i, res in enumerate(results):
        color = cmap(0.35 + 0.6 * i / max(len(results) - 1, 1))
        axes[0].plot(res["freqs"], res["mag_db"], color=color, linewidth=1.3, label=f"Q={res['Q']}")
    axes[0].set_xscale("log")
    axes[0].set_xlim(data["f0"] / 8, data["f0"] * 8)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Experiment B: Q sweep @ {data['f0']:.0f}Hz, {data['gain_db']:+.0f}dB")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    Qs = [r["Q"] for r in results]
    bw = [r["bandwidth_hz"] for r in results]
    axes[1].plot(Qs, bw, "o-", color="#7c3aed")
    axes[1].set_xlabel("Q factor")
    axes[1].set_ylabel("-3dB bandwidth (Hz)")
    axes[1].set_title("Measured bandwidth vs Q (higher Q -> narrower band)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_experiment_fir_vs_iir(data, savepath=None):
    """Magnitude, phase, and group delay overlays for a matched FIR vs IIR band."""
    fir, iir = data["fir"], data["iir"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(fir["freqs_mag"], fir["mag_db"], color="#8b7cf6", label="FIR (Graphic EQ)")
    axes[0].plot(iir["freqs_mag"], iir["mag_db"], color="#22d3c9", label="IIR (Parametric EQ)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title(f"Experiment C: FIR vs IIR @ ~{data['f0']:.0f}Hz, {data['gain_db']:+.0f}dB")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fir["freqs_phase"], fir["phase"], color="#8b7cf6")
    axes[1].plot(iir["freqs_phase"], iir["phase"], color="#22d3c9")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_title("Phase response (FIR is linear; IIR is not)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(fir["freqs_gd"], fir["gd_ms"], color="#8b7cf6")
    axes[2].plot(iir["freqs_gd"], iir["gd_ms"], color="#22d3c9")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Group delay (ms)")
    axes[2].set_title("Group delay (FIR is flat; IIR varies)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale("log")
    axes[2].set_xlim(20, data["fs"] / 2)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_experiment_hum_removal(data, savepath=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data["freqs_before"], data["mag_before"], color="steelblue", linewidth=1, label="Before")
    ax.plot(data["freqs_after"], data["mag_after"], color="crimson", linewidth=1, label="After")
    ax.axvline(data["f0"], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(2 * data["f0"], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(20, data["fs"] / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Experiment D: {data['f0']:.0f}Hz hum removal "
                 f"({data['attenuation_fundamental_db']:.1f}dB / "
                 f"{data['attenuation_2nd_harmonic_db']:.1f}dB attenuation)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig


def plot_experiment_noise_reduction(data, savepath=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(data["freqs_before"], data["mag_before"], color="steelblue", linewidth=1, label="Before (noisy)")
    ax.plot(data["freqs_after"], data["mag_after"], color="crimson", linewidth=1, label="After (spectral gate)")
    ax.set_xscale("log")
    ax.set_xlim(20, data["fs"] / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Experiment E: Noise reduction (noise floor reduced {data['noise_reduction_db']:.1f}dB)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    return fig
