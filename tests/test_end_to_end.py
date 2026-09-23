"""
tests/test_end_to_end.py
=========================
Full pipeline integration test, portable and assertion-based:

    Input signal
        |
        v
    Graphic EQ (bass_boost preset)
        |
        v
    50 Hz hum removal
        |
        v
    Noise reduction (spectral gate)
        |
        v
    Clipping-protection normalization
        |
        v
    Output WAV
        |
        v
    Verify: exists, sample rate preserved, duration preserved,
            no NaN/Inf, amplitude within [-1, 1],
            expected frequency content actually changed

Run with: python3 tests/test_end_to_end.py
"""
import os, sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from core.audio_io import make_demo_music_signal, save_wav, load_wav, check_and_normalize
from core.eq_engine import GraphicEQ
from core.noise_reduction import remove_hum, NOISE_REDUCTION_METHODS
from core.analysis import compute_band_energy

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


print("=== End-to-end pipeline: WAV -> EQ -> hum removal -> noise reduction -> normalize -> WAV ===\n")

# --- 1. build input: demo music + injected 50Hz hum (to prove hum removal actually does something) ---
x, fs = make_demo_music_signal(bars=3)
t = np.arange(len(x)) / fs
x = x + 0.06 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 100 * t)
input_path = os.path.join(OUTPUT_DIR, "e2e_input.wav")
save_wav(input_path, x, fs)
check("Input WAV file was created", os.path.exists(input_path))

# --- 2. load it back (round-trip through disk, like a real user would) ---
x_loaded, fs_loaded = load_wav(input_path)
check("Sample rate preserved through save/load round-trip", fs_loaded == fs, f"{fs_loaded} vs {fs}")
check("Duration preserved through save/load round-trip (within 1 sample)",
      abs(len(x_loaded) - len(x)) <= 1, f"{len(x_loaded)} vs {len(x)} samples")

# --- 3. run the full processing chain (each stage measured separately so
#         we test what each stage actually claims to do, not a compound
#         effect where one stage's output could mask another's) ---
eq = GraphicEQ(fs_loaded)
eq.set_gains([7, 6, 4, 2, 0, 0, 0, 0, 0, 0])  # bass_boost
y_eq = eq.process(x_loaded)

band_before = compute_band_energy(x_loaded, fs_loaded)
band_after_eq = compute_band_energy(y_eq, fs_loaded)
check("Bass-boost EQ measurably increased bass energy share",
      band_after_eq["Bass"] > band_before["Bass"],
      f"{band_before['Bass']:.1f}% -> {band_after_eq['Bass']:.1f}%")

y_hum = remove_hum(y_eq, fs_loaded, f0=50.0, Q=30.0)

def mag_at(sig, freq, fs_):
    n = len(sig)
    X = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1 / fs_)
    idx = np.argmin(np.abs(freqs - freq))
    return 20 * np.log10(np.abs(X[idx]) + 1e-9)

hum_before = mag_at(y_eq, 50, fs_loaded)
hum_after = mag_at(y_hum, 50, fs_loaded)
hum_reduction_db = hum_before - hum_after
check("50Hz hum was measurably reduced (>10dB)", hum_reduction_db > 10,
      f"reduced by {hum_reduction_db:.1f}dB")

y = NOISE_REDUCTION_METHODS["spectral_gate"](y_hum, fs_loaded)
y, clip_info = check_and_normalize(y)

output_path = os.path.join(OUTPUT_DIR, "e2e_output.wav")
save_wav(output_path, y, fs_loaded)
check("Output WAV file was created", os.path.exists(output_path))

# --- 4. sanity checks on the FINAL chained output ---
check("Output has no NaN values", not np.any(np.isnan(y)))
check("Output has no Inf values", not np.any(np.isinf(y)))
check("Output amplitude within [-1, 1] (clipping protection worked)",
      np.max(np.abs(y)) <= 1.0 + 1e-6, f"peak={np.max(np.abs(y)):.4f}")
check("Output length matches input length", len(y) == len(x_loaded), f"{len(y)} vs {len(x_loaded)}")

# --- 5. verify the output WAV on disk round-trips correctly too ---
y_reloaded, fs_reloaded = load_wav(output_path)
check("Output sample rate matches input", fs_reloaded == fs_loaded)
check("Output duration matches input (within 1 sample)",
      abs(len(y_reloaded) - len(x_loaded)) <= 1)

# --- 6. note on a real interaction this test surfaced ---
# Spectral gating uses a per-bin MEDIAN as its noise-floor estimate (see
# core/noise_reduction.py docstring), which assumes the signal occupies
# less than half of each frequency bin's timeline. This demo track's bass
# line is nearly continuous, so chaining spectral_gate after a bass boost
# can end up attenuating much of that boosted bass -- a genuine, documented
# limitation of median-based gating on sustained low end, not a bug. This
# is exactly why the checks above measure the EQ's effect and the hum
# filter's effect independently rather than only checking the fully
# chained output.

# --- cleanup generated test artifacts ---
for p in (input_path, output_path):
    if os.path.exists(p):
        os.remove(p)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) did not pass: {failures}")
    sys.exit(1)
else:
    print("All end-to-end checks passed.")
