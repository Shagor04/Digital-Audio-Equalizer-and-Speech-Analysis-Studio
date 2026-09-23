"""
core/report.py
===============
Generates a single multi-page PDF report combining everything the
analysis modules compute -- input info, before/after metrics, filter
response (magnitude/phase/group delay), band energy, FIR vs IIR
comparison, and clipping-protection status. Meant to be handed in
alongside (or instead of) individual PNGs for the project report.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from core.analysis import summarize, format_comparison_table, compute_band_energy, compute_magnitude_spectrum
from core.audio_io import check_and_normalize
from core.filter_analysis import (
    compute_phase_response, compute_group_delay, measure_filter_performance,
    compare_fir_vs_iir, format_fir_vs_iir_table,
)


def _text_page(pdf, title, body_lines, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold", va="top")
    body = "\n".join(body_lines)
    fig.text(0.08, 0.88, body, fontsize=fontsize, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(x, y, fs, mode, eq_info, graphic_eq=None, parametric_eq=None,
                         target_freq=None, target_gain=None, out_path="output/dsp_report.pdf"):
    """
    Build the full report. `mode` is "graphic" or "parametric"; eq_info is
    a short human-readable description of the active settings (preset name
    or band list) to print on the title page. If both graphic_eq and
    parametric_eq are supplied, an FIR-vs-IIR comparison page is included.
    """
    y_norm, clip_info = check_and_normalize(y)

    before = summarize(x, fs)
    after = summarize(y_norm, fs)
    band_before = compute_band_energy(x, fs)
    band_after = compute_band_energy(y_norm, fs)

    with PdfPages(out_path) as pdf:
        # ---- Page 1: title + input info + metrics table ----
        lines = [
            f"Mode: {mode}",
            f"EQ settings: {eq_info}",
            "",
            f"Sample rate: {fs} Hz",
            f"Duration: {before['duration_s']:.2f} s",
            "",
            "Clipping protection:",
            f"  Peak before normalization : {clip_info['peak_before']:.4f}",
            f"  Peak after normalization  : {clip_info['peak_after']:.4f}",
            f"  Clipping detected         : {clip_info['clipping_detected']}",
            "",
            "Quantitative metrics (Original vs Processed):",
            "",
        ] + format_comparison_table(before, after).split("\n")
        _text_page(pdf, "CSE220 Digital Audio Equalizer -- DSP Analysis Report", lines)

        # ---- Page 2: waveform + spectrum + spectrogram ----
        fig = plt.figure(figsize=(8.5, 11))
        gs = fig.add_gridspec(3, 1)
        ax0 = fig.add_subplot(gs[0])
        t = np.arange(len(x)) / fs
        ax0.plot(t, x, linewidth=0.4, color="steelblue", alpha=0.8, label="Original")
        ax0.plot(np.arange(len(y_norm)) / fs, y_norm, linewidth=0.4, color="crimson", alpha=0.6, label="Processed")
        ax0.set_title("Waveform")
        ax0.legend(fontsize=8)

        ax1 = fig.add_subplot(gs[1])
        fx, mx = compute_magnitude_spectrum(x, fs)
        fy, my = compute_magnitude_spectrum(y_norm, fs)
        ax1.plot(fx, mx, color="steelblue", linewidth=0.8, label="Original")
        ax1.plot(fy, my, color="crimson", linewidth=0.8, label="Processed")
        ax1.set_xscale("log")
        ax1.set_xlim(20, fs / 2)
        ax1.set_title("Magnitude spectrum")
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[2])
        ax2.specgram(y_norm, Fs=fs, NFFT=1024, noverlap=512, cmap="magma")
        ax2.set_title("Spectrogram (processed)")
        ax2.set_xlabel("Time (s)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: phase response + group delay ----
        if mode == "graphic" and graphic_eq is not None:
            b, a = graphic_eq.composite_kernel(), [1.0]
        elif mode == "parametric" and parametric_eq is not None:
            b, a = parametric_eq.combined_coeffs()
        else:
            b, a = None, None

        if b is not None:
            fp, phase = compute_phase_response(b, a, fs)
            fg, gd = compute_group_delay(b, a, fs)
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
            axes[0].plot(fp, phase, color="#8b7cf6")
            axes[0].set_title("Phase response")
            axes[0].set_ylabel("Phase (rad)")
            axes[0].grid(alpha=0.3)
            axes[1].plot(fg, gd, color="#22d3c9")
            axes[1].set_title("Group delay")
            axes[1].set_ylabel("Delay (ms)")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].grid(alpha=0.3)
            fig.suptitle(f"{mode.title()} EQ -- Phase & Group Delay", y=0.98)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Page 4: band energy ----
        fig, ax = plt.subplots(figsize=(8.5, 6))
        names = list(band_before.keys())
        yb = np.arange(len(names))
        h = 0.35
        ax.barh(yb + h / 2, [band_before[n] for n in names], height=h, label="Original", color="steelblue")
        ax.barh(yb - h / 2, [band_after[n] for n in names], height=h, label="Processed", color="crimson")
        ax.set_yticks(yb)
        ax.set_yticklabels(names)
        ax.set_xlabel("Share of total spectral energy (%)")
        ax.set_title("Frequency-band energy distribution")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 5: filter performance + FIR vs IIR (if applicable) ----
        lines = []
        if target_freq is not None and target_gain is not None:
            if mode == "graphic" and graphic_eq is not None:
                freqs_r, mag_r = graphic_eq.composite_response()
            elif mode == "parametric" and parametric_eq is not None:
                freqs_r, mag_r = parametric_eq.composite_response()
            else:
                freqs_r, mag_r = None, None
            if freqs_r is not None:
                perf = measure_filter_performance(freqs_r, mag_r, target_freq, target_gain)
                lines += [
                    "Filter performance (design target vs measured):",
                    f"  Target frequency   : {perf['target_freq_hz']:.1f} Hz",
                    f"  Target gain        : {perf['target_gain_db']:+.2f} dB",
                    f"  Measured gain      : {perf['measured_gain_db']:+.2f} dB",
                    f"  Gain error         : {perf['gain_error_db']:+.4f} dB",
                    f"  Measured peak freq : {perf['measured_peak_freq_hz']:.1f} Hz",
                    f"  -3dB bandwidth     : {perf['bandwidth_3db_hz']:.1f} Hz",
                    "",
                ]

        if graphic_eq is not None and parametric_eq is not None:
            cmp = compare_fir_vs_iir(graphic_eq, parametric_eq, x, fs)
            lines += ["FIR vs IIR comparison:", ""] + format_fir_vs_iir_table(cmp).split("\n")

        if lines:
            _text_page(pdf, "Filter Performance & FIR vs IIR", lines, fontsize=8)

    return out_path
