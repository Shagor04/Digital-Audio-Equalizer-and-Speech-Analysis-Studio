"""
tests/verify_step1.py
======================
Automated correctness checks for the core filter-design/engine math.
Portable: all paths are derived from this file's own location, never
hard-coded, so this runs correctly on any machine/clone of the project.

Run with: python3 tests/verify_step1.py
Exits non-zero (and prints which assertion failed) if anything is wrong.
"""
import sys, os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from core.eq_engine import GraphicEQ, ParametricEQ
from core.filters import peaking_eq_coeffs, apply_iir
from scipy.signal import lfilter
from visualization.plots import plot_graphic_eq_bands, plot_parametric_eq_response

fs = 44100
failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


print("=== Graphic EQ: flat-gain composite response should be ~0 dB ===")
geq = GraphicEQ(fs)
freqs, mag = geq.composite_response()
band = (freqs > 100) & (freqs < 15000)
flat_error = np.max(np.abs(mag[band]))
print(f"  mean={mag[band].mean():.4f} dB, max abs error={flat_error:.4f} dB")
check("Flat-gain composite response is within 0.1 dB of 0 dB", flat_error < 0.1, f"max error={flat_error:.4f}dB")

plot_graphic_eq_bands(geq, savepath=os.path.join(TEST_DIR, "graphic_eq_flat.png"))

geq.set_gains([6, 5, 3, 1, 0, 0, 0, 0, 0, 0])
plot_graphic_eq_bands(geq, savepath=os.path.join(TEST_DIR, "graphic_eq_bassboost.png"))

print("\n=== Graphic EQ: white-noise passthrough at 0dB should preserve RMS ===")
rng = np.random.default_rng(0)
x = rng.standard_normal(fs * 2)
geq.set_gains([0] * 10)
y = geq.process(x)
rms_in, rms_out = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(y ** 2))
rms_error = abs(rms_in - rms_out) / rms_in
print(f"  input RMS={rms_in:.4f}, output RMS={rms_out:.4f}, relative error={rms_error:.4%}")
check("RMS preserved within 1% at unity gain", rms_error < 0.01, f"error={rms_error:.4%}")

print("\n=== Parametric EQ: peaking boost at 1kHz should measure ~+8dB ===")
peq = ParametricEQ(fs)
peq.add_band(f0=1000, gain_db=8, Q=1.5, kind="peak")
peq.add_band(f0=100, gain_db=4, Q=0.7, kind="lowshelf")
freqs2, mag2 = peq.composite_response()
idx_1k = np.argmin(np.abs(freqs2 - 1000))
gain_error = abs(mag2[idx_1k] - 8.0)
print(f"  measured gain at 1kHz = {mag2[idx_1k]:.2f} dB (target +8.00 dB, error={gain_error:.4f} dB)")
check("Measured 1kHz peak gain within 0.1 dB of target", gain_error < 0.1, f"error={gain_error:.4f}dB")
plot_parametric_eq_response(peq, savepath=os.path.join(TEST_DIR, "parametric_eq_test.png"))

print("\n=== IIR: from-scratch difference equation vs scipy.lfilter ===")
b, a = peaking_eq_coeffs(1000, 8, 1.5, fs)
x_short = rng.standard_normal(2000)
y_scratch = apply_iir(x_short, b, a)
y_scipy = lfilter(b, a, x_short)
iir_error = np.max(np.abs(y_scratch - y_scipy))
print(f"  max abs error = {iir_error:.2e}")
check("From-scratch apply_iir matches scipy.lfilter to 1e-10", iir_error < 1e-10, f"error={iir_error:.2e}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) did not pass: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")
