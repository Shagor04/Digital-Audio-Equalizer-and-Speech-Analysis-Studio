"""
tests/verify_voice_separation.py
=================================
Automated correctness check for the pitch-based voice separation
(core/voice_separation.py). Builds the synthetic two-talker demo
signal (whose two ground-truth voices we actually have, since it's
generated), separates it, and checks that each separated stream lines
up with the RIGHT ground-truth voice -- not just "some output came
out".

Portable: all paths derived from this file's own location.
Run with: python3 tests/verify_voice_separation.py
Exits non-zero (and prints which assertion failed) if anything is wrong.
"""
import sys, os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
from core.audio_io import make_two_voice_test_signal
from core.voice_separation import separate_voices

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def envelope(sig, frame=1024):
    n_frames = len(sig) // frame
    return np.array([np.sqrt(np.mean(sig[i * frame:(i + 1) * frame] ** 2) + 1e-12)
                      for i in range(n_frames)])


print("=== Building synthetic two-talker demo (110 Hz voice + 210 Hz voice) ===")
mix, fs, true_a, true_b = make_two_voice_test_signal(fs=44100, duration=6.0, seed=2, return_components=True)
print(f"  mix duration={len(mix)/fs:.1f}s @ {fs}Hz")

print("\n=== Running separate_voices(n_voices=2) ===")
t0 = time.time()
voices, info = separate_voices(mix, fs, n_voices=2)
elapsed = time.time() - t0
print(f"  finished in {elapsed:.2f}s, detected pitch registers: "
      f"{[f'{p:.1f}Hz' for p in info['pitch_hz']]}")
print(f"  active_fraction per voice: {[f'{a:.2f}' for a in info['active_fraction']]}")

check("Returned exactly 2 voice streams", len(voices) == 2, f"got {len(voices)}")
check("Each stream matches input length",
      all(len(v) == len(mix) for v in voices),
      f"lengths={[len(v) for v in voices]}")

print("\n=== Checking detected pitch registers are close to the true 110Hz / 210Hz ===")
low_pitch, high_pitch = sorted(info["pitch_hz"])
low_err = abs(low_pitch - 110)
high_err = abs(high_pitch - 210)
print(f"  low register: detected {low_pitch:.1f}Hz vs true 110Hz (error {low_err:.1f}Hz)")
print(f"  high register: detected {high_pitch:.1f}Hz vs true 210Hz (error {high_err:.1f}Hz)")
check("Low-register pitch within 25Hz of true 110Hz voice", low_err < 25, f"error={low_err:.1f}Hz")
check("High-register pitch within 25Hz of true 210Hz voice", high_err < 25, f"error={high_err:.1f}Hz")

print("\n=== Checking each separated stream tracks the RIGHT ground-truth voice ===")
env_sep = [envelope(v) for v in voices]
env_true_a = envelope(true_a)
env_true_b = envelope(true_b)

n = min(len(env_sep[0]), len(env_sep[1]), len(env_true_a), len(env_true_b))


def corr(u, w):
    return float(np.corrcoef(u[:n], w[:n])[0, 1])


# figure out which separated stream (0 or 1) best matches voice A vs voice B
c0a, c0b = corr(env_sep[0], env_true_a), corr(env_sep[0], env_true_b)
c1a, c1b = corr(env_sep[1], env_true_a), corr(env_sep[1], env_true_b)
print(f"  stream0 vs true_A={c0a:.2f}, stream0 vs true_B={c0b:.2f}")
print(f"  stream1 vs true_A={c1a:.2f}, stream1 vs true_B={c1b:.2f}")

if abs(c0a - c1a) + abs(c1b - c0b) > 0:  # always true, just picking the better pairing
    if c0a + c1b >= c0b + c1a:
        best_a, best_b = c0a, c1b
        cross_a, cross_b = c0b, c1a
    else:
        best_a, best_b = c1a, c0b
        cross_a, cross_b = c1b, c0a

print(f"  best pairing: matched correlation A={best_a:.2f}, B={best_b:.2f} "
      f"(vs cross correlation A={cross_a:.2f}, B={cross_b:.2f})")
check("Voice A's matched stream correlates more with true A than the other stream does",
      best_a > cross_a, f"{best_a:.2f} vs {cross_a:.2f}")
check("Voice B's matched stream correlates more with true B than the other stream does",
      best_b > cross_b, f"{best_b:.2f} vs {cross_b:.2f}")
check("Matched correlation for voice A is reasonably strong (>0.4)", best_a > 0.4, f"{best_a:.2f}")
check("Matched correlation for voice B is reasonably strong (>0.4)", best_b > 0.4, f"{best_b:.2f}")

print("\n=== Checking mask conservation (separated voices should sum back close to the mix) ===")
resum = voices[0] + voices[1]
rms_mix = np.sqrt(np.mean(mix ** 2))
rms_diff = np.sqrt(np.mean((resum - mix) ** 2))
relative_diff = rms_diff / (rms_mix + 1e-12)
print(f"  RMS(sum of voices - mix) / RMS(mix) = {relative_diff:.3f}")
check("Sum of separated voices stays within 35% RMS of the original mix", relative_diff < 0.35,
      f"relative_diff={relative_diff:.3f}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) did not pass: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")
