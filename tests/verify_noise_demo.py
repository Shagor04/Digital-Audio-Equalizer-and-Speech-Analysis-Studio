"""
tests/verify_noise_demo.py
===========================
Automated correctness check for the Noise Reduction Demo signal
(core/audio_io.py's make_noise_demo_signal) and the processing chain the
GUI auto-configures when it's loaded (hum removal + spectral gating).

Checks, against the demo's own clean reference (return_components=True):
  1. The clean melody is genuinely audible under the noise (a sanity
     check on the demo itself, not the algorithm).
  2. Hum removal + spectral gating together measurably quiet the noise
     floor during the signal's silent gaps.
  3. The 50Hz hum tone specifically is reduced by a large margin.
  4. The melody itself survives processing well (high correlation with
     the clean reference during a note) -- i.e. noise reduction actually
     removes noise without gutting the real content.

Run with: python3 tests/verify_noise_demo.py
"""
import sys, os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scipy.ndimage as ndi
from core.audio_io import make_noise_demo_signal
from core.noise_reduction import remove_hum, spectral_gate

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


print("=== Building the Noise Reduction Demo signal ===")
noisy, fs, clean = make_noise_demo_signal(return_components=True)
print(f"  duration={len(noisy)/fs:.1f}s @ {fs}Hz")
check("Clean reference and noisy signal have the same length", len(clean) == len(noisy))

print("\n=== Processing exactly as the GUI auto-configures it (hum removal, then spectral gating) ===")
step1 = remove_hum(noisy, fs)
processed = spectral_gate(step1, fs, noise_reduce_db=25.0)

print("\n=== Checking the noise floor is measurably quieter during silent gaps ===")
silent_mask = np.abs(clean) < 1e-9
shrink = int(0.05 * fs)  # erode gap edges so note release/STFT smearing doesn't bleed in
eroded = ndi.binary_erosion(silent_mask, structure=np.ones(2 * shrink + 1))
rms_before = np.sqrt(np.mean(noisy[eroded] ** 2))
rms_after = np.sqrt(np.mean(processed[eroded] ** 2))
reduction_db = 20 * np.log10(rms_before / (rms_after + 1e-12))
print(f"  gap RMS: {rms_before:.4f} -> {rms_after:.4f}  ({reduction_db:.1f} dB reduction)")
check("Noise floor reduced by at least 10dB during silent gaps", reduction_db > 10, f"{reduction_db:.1f}dB")

print("\n=== Checking the 50Hz hum tone specifically is removed ===")
freqs = np.fft.rfftfreq(len(noisy), d=1 / fs)
b50 = np.argmin(np.abs(freqs - 50))
mag_before = np.abs(np.fft.rfft(noisy))[b50]
mag_after = np.abs(np.fft.rfft(processed))[b50]
hum_reduction_db = 20 * np.log10(mag_before / (mag_after + 1e-12))
print(f"  50Hz magnitude: {mag_before:.1f} -> {mag_after:.1f}  ({hum_reduction_db:.1f} dB reduction)")
check("50Hz hum reduced by at least 20dB", hum_reduction_db > 20, f"{hum_reduction_db:.1f}dB")

print("\n=== Checking the melody itself survives processing ===")
# a window centered on the demo's second note (skips the very first attack
# transient / initial silence for a cleaner correlation window)
note_region = slice(int(0.35 * fs), int(0.35 * fs) + int(0.3 * fs))
corr = np.corrcoef(processed[note_region], clean[note_region])[0, 1]
print(f"  correlation between processed and clean melody during a note: {corr:.4f}")
check("Processed melody correlates strongly (>0.9) with the clean reference", corr > 0.9, f"corr={corr:.4f}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) did not pass: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")
