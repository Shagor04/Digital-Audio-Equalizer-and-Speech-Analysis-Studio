"""
main.py
=======
Command-line entry point for the audio equalizer.

Examples
--------
# Use the built-in synthetic test signal, apply the "bass_boost" graphic preset:
    python3 main.py --synthetic --mode graphic --preset bass_boost

# Process your own file with a parametric preset:
    python3 main.py --input audio/song.wav --mode parametric --preset vocal_presence

# Custom graphic gains (10 comma-separated dB values, one per band):
    python3 main.py --synthetic --mode graphic --gains 6,5,3,1,0,0,0,0,0,0

# List available presets:
    python3 main.py --list-presets
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.audio_io import load_wav, save_wav, make_synthetic_test_signal, make_demo_music_signal, check_and_normalize
from core.eq_engine import GraphicEQ, ParametricEQ, TEN_BAND_CENTERS
from core.analysis import summarize, format_comparison_table, compute_band_energy, format_band_energy_bars
from core.noise_reduction import NOISE_REDUCTION_METHODS, remove_hum
from core.filter_analysis import (
    compute_phase_response, compute_group_delay, measure_filter_performance,
    compare_fir_vs_iir, format_fir_vs_iir_table,
)
from core.report import generate_pdf_report
from presets.presets import GRAPHIC_PRESETS, PARAMETRIC_PRESETS, list_presets
from visualization.plots import (
    plot_graphic_eq_bands, plot_parametric_eq_response,
    plot_waveform_comparison, plot_spectrogram_comparison,
    plot_spectrum_comparison, plot_analysis_dashboard,
    plot_phase_and_group_delay, plot_band_energy,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def build_graphic_eq(fs, preset_name=None, custom_gains=None):
    eq = GraphicEQ(fs)
    if custom_gains is not None:
        eq.set_gains(custom_gains)
    elif preset_name is not None:
        eq.set_gains(GRAPHIC_PRESETS[preset_name])
    return eq


def build_parametric_eq(fs, preset_name=None):
    eq = ParametricEQ(fs)
    bands = PARAMETRIC_PRESETS.get(preset_name, [])
    for band in bands:
        eq.add_band(**band)
    return eq


def main():
    p = argparse.ArgumentParser(description="CSE220 Digital Audio Equalizer")
    p.add_argument("--input", type=str, default=None, help="path to input .wav file")
    p.add_argument("--synthetic", action="store_true", help="use a generated test-tone signal instead of --input")
    p.add_argument("--demo-music", action="store_true", help="use a generated musical demo track (bass+melody+hihats)")
    p.add_argument("--mode", choices=["graphic", "parametric"], default="graphic")
    p.add_argument("--preset", type=str, default="flat")
    p.add_argument("--gains", type=str, default=None,
                    help="comma-separated 10 dB values for graphic mode, overrides --preset")
    p.add_argument("--output", type=str, default=None, help="output wav path")
    p.add_argument("--notch50", action="store_true", help="apply a 50Hz mains-hum notch filter (+ 2nd harmonic)")
    p.add_argument("--noise-reduction", choices=list(NOISE_REDUCTION_METHODS.keys()), default="none",
                    help="apply a noise-reduction stage after EQ")
    p.add_argument("--target-freq", type=float, default=None,
                    help="frequency (Hz) to evaluate filter performance against. If omitted, "
                         "auto-detected from the band with the largest |gain| in the active config.")
    p.add_argument("--target-gain", type=float, default=None,
                    help="design target gain (dB) at --target-freq. If omitted, auto-detected "
                         "alongside --target-freq from the active EQ configuration.")
    p.add_argument("--report", action="store_true", help="generate a combined multi-page PDF analysis report")
    p.add_argument("--experiments", action="store_true",
                    help="run all 5 controlled experiments (bass sweep, Q sweep, FIR vs IIR, hum, "
                         "noise) and save experiments_report.pdf + individual plots to output/experiments/, "
                         "then exit (ignores --mode/--preset/--input)")
    p.add_argument("--list-presets", action="store_true")
    args = p.parse_args()

    if args.list_presets:
        list_presets()
        return

    if args.experiments:
        from core.run_experiments import run_all
        run_all(os.path.join(OUT_DIR, "experiments"))
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. get input audio ---
    if args.demo_music:
        print("Generating musical demo track (bass + melody + hi-hats)...")
        x, fs = make_demo_music_signal()
        save_wav(os.path.join(OUT_DIR, "input_demo_music.wav"), x, fs)
        print(f"  Saved to output/input_demo_music.wav ({len(x)/fs:.1f}s @ {fs}Hz)")
    elif args.synthetic or args.input is None:
        print("No --input given (or --synthetic set): generating synthetic test-tone signal...")
        x, fs = make_synthetic_test_signal()
        save_wav(os.path.join(OUT_DIR, "input_synthetic.wav"), x, fs)
        print(f"  Saved synthetic input to output/input_synthetic.wav ({len(x)/fs:.1f}s @ {fs}Hz)")
    else:
        x, fs = load_wav(args.input)
        print(f"Loaded {args.input}: {len(x)/fs:.1f}s @ {fs}Hz")

    # --- 2. build + apply EQ ---
    if args.mode == "graphic":
        custom_gains = None
        if args.gains:
            custom_gains = [float(g) for g in args.gains.split(",")]
            assert len(custom_gains) == len(TEN_BAND_CENTERS), \
                f"--gains needs {len(TEN_BAND_CENTERS)} comma-separated values"
        eq = build_graphic_eq(fs, preset_name=args.preset, custom_gains=custom_gains)
        print(f"GraphicEQ gains (dB) per band {TEN_BAND_CENTERS}: {eq.gains_db}")
        y = eq.process(x)
        plot_graphic_eq_bands(eq, savepath=os.path.join(OUT_DIR, "eq_response.png"))
        graphic_eq_obj, parametric_eq_obj = eq, build_parametric_eq(fs, preset_name="vocal_presence")
        eq_desc = f"graphic preset '{args.preset}' gains={eq.gains_db}"
    else:
        eq = build_parametric_eq(fs, preset_name=args.preset)
        print(f"ParametricEQ '{args.preset}' bands: {eq.bands}")
        y = eq.process(x)
        plot_parametric_eq_response(eq, savepath=os.path.join(OUT_DIR, "eq_response.png"))
        graphic_eq_obj, parametric_eq_obj = build_graphic_eq(fs, preset_name="bass_boost"), eq
        eq_desc = f"parametric preset '{args.preset}' bands={eq.bands}"

    # --- 3. optional post-processing: 50Hz notch + noise reduction ---
    if args.notch50:
        y = remove_hum(y, fs, f0=50.0, Q=30.0)
        print("Applied 50Hz mains-hum notch filter (+ 2nd harmonic at 100Hz).")
    if args.noise_reduction != "none":
        y = NOISE_REDUCTION_METHODS[args.noise_reduction](y, fs)
        print(f"Applied noise reduction: {args.noise_reduction}")

    # --- 4. clipping protection ---
    y, clip_info = check_and_normalize(y)
    if clip_info["clipping_detected"]:
        print(f"\nClipping detected! Peak was {clip_info['peak_before']:.4f} -- "
              f"normalized down to {clip_info['peak_after']:.4f} "
              f"({clip_info['headroom_db']}dB headroom).")
    else:
        print(f"\nNo clipping: peak level {clip_info['peak_before']:.4f} (within [-1, 1]).")

    # --- 5. save + visualize ---
    out_path = args.output or os.path.join(OUT_DIR, f"output_{args.mode}_{args.preset}.wav")
    save_wav(out_path, y, fs)
    print(f"Saved equalized audio to {out_path}")

    plot_waveform_comparison(x, y, fs, savepath=os.path.join(OUT_DIR, "waveform_compare.png"))
    plot_spectrogram_comparison(x, y, fs, savepath=os.path.join(OUT_DIR, "spectrogram_compare.png"))
    plot_spectrum_comparison(x, y, fs, savepath=os.path.join(OUT_DIR, "spectrum_compare.png"))
    print(f"Saved plots to {OUT_DIR}/ (eq_response.png, waveform_compare.png, "
          f"spectrogram_compare.png, spectrum_compare.png)")

    # --- 6. quantitative analysis ---
    metrics_before = summarize(x, fs)
    metrics_after = summarize(y, fs)
    print()
    print(format_comparison_table(metrics_before, metrics_after))

    plot_analysis_dashboard(x, y, fs, metrics_before, metrics_after,
                             savepath=os.path.join(OUT_DIR, "analysis_dashboard.png"))
    print(f"\nSaved combined analysis dashboard to {OUT_DIR}/analysis_dashboard.png")

    # --- 7. frequency-band energy ---
    band_before = compute_band_energy(x, fs)
    band_after = compute_band_energy(y, fs)
    print()
    print(format_band_energy_bars(band_before, band_after))
    plot_band_energy(band_before, band_after, savepath=os.path.join(OUT_DIR, "band_energy.png"))

    # --- 8. phase response + group delay ---
    if args.mode == "graphic":
        b_phase, a_phase = eq.composite_kernel(), [1.0]
    else:
        b_phase, a_phase = eq.combined_coeffs()
    fp, phase = compute_phase_response(b_phase, a_phase, fs)
    fg, gd = compute_group_delay(b_phase, a_phase, fs)
    plot_phase_and_group_delay(fp, phase, fg, gd, title_prefix=args.mode.title(),
                                savepath=os.path.join(OUT_DIR, "phase_group_delay.png"))
    print(f"Saved phase/group-delay plot to {OUT_DIR}/phase_group_delay.png "
          f"(mean group delay: {np.nanmean(gd):.3f} ms)")

    # --- 9. filter performance (auto-detected from the active config if not given explicitly) ---
    target_freq, target_gain = args.target_freq, args.target_gain
    if target_freq is None or target_gain is None:
        if args.mode == "graphic":
            gains = eq.gains_db
            if any(abs(g) > 1e-6 for g in gains):
                idx = int(np.argmax([abs(g) for g in gains]))
                target_freq, target_gain = TEN_BAND_CENTERS[idx], gains[idx]
        else:
            if eq.bands:
                band = max(eq.bands, key=lambda b: abs(b["gain_db"]))
                if abs(band["gain_db"]) > 1e-6:
                    target_freq, target_gain = band["f0"], band["gain_db"]
        if target_freq is not None:
            print(f"\n(No --target-freq/--target-gain given; auto-detected from the "
                  f"largest-|gain| band: {target_freq}Hz, {target_gain:+.1f}dB)")

    if target_freq is not None and target_gain is not None:
        freqs_r, mag_r = eq.composite_response()
        perf = measure_filter_performance(freqs_r, mag_r, target_freq, target_gain)
        print(f"\nFilter performance @ {target_freq}Hz target {target_gain:+.1f}dB:")
        print(f"  Measured gain      : {perf['measured_gain_db']:+.2f} dB "
              f"(error {perf['gain_error_db']:+.4f} dB)")
        print(f"  Measured peak freq : {perf['measured_peak_freq_hz']:.1f} Hz")
        print(f"  -3dB bandwidth     : {perf['bandwidth_3db_hz']:.1f} Hz")

    # --- 10. FIR vs IIR structural + timing comparison ---
    fir_iir_cmp = compare_fir_vs_iir(graphic_eq_obj, parametric_eq_obj, x, fs)
    print()
    print(format_fir_vs_iir_table(fir_iir_cmp))

    # --- 11. optional PDF report ---
    if args.report:
        report_path = generate_pdf_report(
            x, y, fs, mode=args.mode, eq_info=eq_desc,
            graphic_eq=graphic_eq_obj, parametric_eq=parametric_eq_obj,
            target_freq=target_freq, target_gain=target_gain,
            out_path=os.path.join(OUT_DIR, "dsp_report.pdf"),
        )
        print(f"\nSaved combined PDF report to {report_path}")


if __name__ == "__main__":
    main()
