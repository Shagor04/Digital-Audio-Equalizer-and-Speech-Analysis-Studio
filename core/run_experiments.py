"""
core/run_experiments.py
========================
Runs all 5 controlled experiments (see core/experiments.py) and saves:
  - one PNG per experiment
  - one combined multi-page PDF (experiments_report.pdf) with a text
    summary page per experiment followed by its plot

Usage:
    python3 -m core.run_experiments [output_dir]
or from main.py via --experiment all|bass_boost|q_factor|fir_vs_iir|hum|noise
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from core.experiments import (
    experiment_bass_boost, experiment_q_factor, experiment_fir_vs_iir,
    experiment_hum_removal, experiment_noise_reduction,
)
from visualization.plots import (
    plot_experiment_bass_boost, plot_experiment_q_factor, plot_experiment_fir_vs_iir,
    plot_experiment_hum_removal, plot_experiment_noise_reduction,
)


def _summary_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.5, 3))
    fig.text(0.06, 0.85, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.06, 0.65, "\n".join(lines), fontsize=10, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def run_all(out_dir="output/experiments"):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "experiments_report.pdf")

    with PdfPages(pdf_path) as pdf:
        # --- A: Bass boost sweep ---
        print("Running Experiment A: Bass boost sweep...")
        data = experiment_bass_boost()
        lines = ["Gain (dB)   Bass energy %   RMS      Peak"]
        for r in data["results"]:
            lines.append(f"  {r['gain_db']:+.0f}         {r['band_energy']['Bass']:6.1f}%       "
                          f"{r['rms']:.4f}  {r['peak']:.4f}")
        _summary_page(pdf, "Experiment A: Bass Boost Gain Sweep", lines)
        fig = plot_experiment_bass_boost(data, savepath=os.path.join(out_dir, "exp_A_bass_boost.png"))
        pdf.savefig(fig)
        plt.close(fig)

        # --- B: Q factor sweep ---
        print("Running Experiment B: Q factor sweep...")
        data = experiment_q_factor()
        lines = ["Q       Measured gain (dB)   -3dB Bandwidth (Hz)"]
        for r in data["results"]:
            lines.append(f"  {r['Q']:<6}  {r['measured_gain_db']:+.2f}                {r['bandwidth_hz']:.1f}")
        _summary_page(pdf, "Experiment B: Q Factor Sweep", lines)
        fig = plot_experiment_q_factor(data, savepath=os.path.join(out_dir, "exp_B_q_factor.png"))
        pdf.savefig(fig)
        plt.close(fig)

        # --- C: FIR vs IIR ---
        print("Running Experiment C: FIR vs IIR...")
        data = experiment_fir_vs_iir()
        lines = [
            f"Target: {data['f0']:.0f}Hz, {data['gain_db']:+.0f}dB "
            f"(graphic band used: {data['graphic_band_center']}Hz)",
            "",
            f"FIR -- coefficients: {data['fir']['num_coeffs']}, "
            f"processing time: {data['fir']['processing_time_ms']:.2f} ms",
            f"IIR -- coefficients: {data['iir']['num_coeffs']}, "
            f"processing time: {data['iir']['processing_time_ms']:.2f} ms",
        ]
        _summary_page(pdf, "Experiment C: FIR vs IIR", lines)
        fig = plot_experiment_fir_vs_iir(data, savepath=os.path.join(out_dir, "exp_C_fir_vs_iir.png"))
        pdf.savefig(fig)
        plt.close(fig)

        # --- D: Hum removal ---
        print("Running Experiment D: 50Hz hum removal...")
        data = experiment_hum_removal()
        lines = [
            f"Fundamental (50Hz) attenuation : {data['attenuation_fundamental_db']:.1f} dB",
            f"2nd harmonic (100Hz) attenuation: {data['attenuation_2nd_harmonic_db']:.1f} dB",
        ]
        _summary_page(pdf, "Experiment D: 50Hz Hum Removal", lines)
        fig = plot_experiment_hum_removal(data, savepath=os.path.join(out_dir, "exp_D_hum_removal.png"))
        pdf.savefig(fig)
        plt.close(fig)

        # --- E: Noise reduction ---
        print("Running Experiment E: Noise reduction...")
        data = experiment_noise_reduction()
        lines = [
            f"Noise-only RMS before: {data['noise_rms_before']:.4f}",
            f"Noise-only RMS after : {data['noise_rms_after']:.4f}",
            f"Noise floor reduction: {data['noise_reduction_db']:.1f} dB",
        ]
        _summary_page(pdf, "Experiment E: Noise Reduction", lines)
        fig = plot_experiment_noise_reduction(data, savepath=os.path.join(out_dir, "exp_E_noise_reduction.png"))
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nAll experiments complete. Combined report: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output/experiments"
    run_all(out_dir)
