"""
tests/verify_voice_separation.py
=================================
Automated correctness check for the pitch + frequency-amplitude voice
separation (core/voice_separation.py). Builds the synthetic two-talker
and three-talker demo signals (whose ground-truth component voices we
actually have, since they're generated), separates each, and checks
that every separated stream lines up with the RIGHT ground-truth voice
-- not just "some output came out". Each talker is a formant-colored
(vowel-shaped) synthetic voice, not a plain tone -- see
core/audio_io.py's VOWEL_FORMANTS / _voice_talker for how.

Portable: all paths derived from this file's own location.
Run with: python3 tests/verify_voice_separation.py
Exits non-zero (and prints which assertion failed) if anything is wrong.
"""
import sys, os
import itertools

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
from core.audio_io import make_two_voice_test_signal, make_three_voice_test_signal
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


def best_pairing(streams, truths):
    """Try every way of matching separated streams to ground-truth voices
    (there are only 2 or 3 of them, so brute force is fine) and return the
    pairing with the highest total envelope correlation, plus, for
    comparison, the correlation each stream has with every OTHER truth
    voice it wasn't matched to."""
    n = len(streams)
    env_s = [envelope(s) for s in streams]
    env_t = [envelope(t) for t in truths]
    m = min(min(len(e) for e in env_s), min(len(e) for e in env_t))
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(env_s[i][:m], env_t[j][:m])[0, 1]

    best_perm, best_total = None, -1e9
    for perm in itertools.permutations(range(n)):
        total = sum(corr[i, perm[i]] for i in range(n))
        if total > best_total:
            best_total, best_perm = total, perm

    matched = [corr[i, best_perm[i]] for i in range(n)]
    # for each stream, the best it does against a truth voice it was NOT matched to
    cross = [max(corr[i, j] for j in range(n) if j != best_perm[i]) for i in range(n)]
    return best_perm, matched, cross


def check_mask_conservation(voices, mix):
    resum = sum(voices)
    rms_mix = np.sqrt(np.mean(mix ** 2))
    rms_diff = np.sqrt(np.mean((resum - mix) ** 2))
    relative_diff = rms_diff / (rms_mix + 1e-12)
    print(f"  RMS(sum of voices - mix) / RMS(mix) = {relative_diff:.3f}")
    check("Sum of separated voices stays within 35% RMS of the original mix", relative_diff < 0.35,
          f"relative_diff={relative_diff:.3f}")


# =====================================================================
print("=== TWO-VOICE DEMO (~105Hz 'ah/oh' voice + ~220Hz 'ee/eh' voice) ===")
mix, fs, true_1, true_2 = make_two_voice_test_signal(seed=2, return_components=True)
print(f"  mix duration={len(mix)/fs:.1f}s @ {fs}Hz")

t0 = time.time()
voices, info = separate_voices(mix, fs, n_voices=2)
print(f"  finished in {time.time()-t0:.2f}s, detected pitch registers: "
      f"{[f'{p:.1f}Hz' for p in info['pitch_hz']]}")

check("Returned exactly 2 voice streams", len(voices) == 2, f"got {len(voices)}")
check("Each stream matches input length", all(len(v) == len(mix) for v in voices))

true_pitches = [105.0, 220.0]
for label, detected, truth in zip(["low", "high"], sorted(info["pitch_hz"]), true_pitches):
    err = abs(detected - truth)
    print(f"  {label} register: detected {detected:.1f}Hz vs true {truth:.0f}Hz (error {err:.1f}Hz)")
    check(f"{label.capitalize()}-register pitch within 25Hz of true {truth:.0f}Hz voice", err < 25,
          f"error={err:.1f}Hz")

perm, matched, cross = best_pairing(voices, [true_1, true_2])
for i, name in enumerate(["Voice 1 (~105Hz)", "Voice 2 (~220Hz)"]):
    print(f"  {name}: matched corr={matched[i]:.2f} vs best cross corr={cross[i]:.2f}")
    check(f"{name}'s matched stream correlates more with its true voice than any other stream does",
          matched[i] > cross[i], f"{matched[i]:.2f} vs {cross[i]:.2f}")
    check(f"{name}'s matched correlation is reasonably strong (>0.4)", matched[i] > 0.4, f"{matched[i]:.2f}")

check_mask_conservation(voices, mix)

# =====================================================================
print("\n=== THREE-VOICE DEMO (~95Hz, ~165Hz, ~250Hz talkers) ===")
mix3, fs, t1, t2, t3 = make_three_voice_test_signal(seed=3, return_components=True)
print(f"  mix duration={len(mix3)/fs:.1f}s @ {fs}Hz")

t0 = time.time()
voices3, info3 = separate_voices(mix3, fs, n_voices=3)
print(f"  finished in {time.time()-t0:.2f}s, detected pitch registers: "
      f"{[f'{p:.1f}Hz' for p in info3['pitch_hz']]}")

check("Returned exactly 3 voice streams", len(voices3) == 3, f"got {len(voices3)}")
check("Each stream matches input length", all(len(v) == len(mix3) for v in voices3))

true_pitches3 = [95.0, 165.0, 250.0]
for label, detected, truth in zip(["low", "mid", "high"], sorted(info3["pitch_hz"]), true_pitches3):
    err = abs(detected - truth)
    print(f"  {label} register: detected {detected:.1f}Hz vs true {truth:.0f}Hz (error {err:.1f}Hz)")
    check(f"{label.capitalize()}-register pitch within 30Hz of true {truth:.0f}Hz voice", err < 30,
          f"error={err:.1f}Hz")

perm3, matched3, cross3 = best_pairing(voices3, [t1, t2, t3])
for i, name in enumerate(["Voice 1 (~95Hz)", "Voice 2 (~165Hz)", "Voice 3 (~250Hz)"]):
    print(f"  {name}: matched corr={matched3[i]:.2f} vs best cross corr={cross3[i]:.2f}")
    check(f"{name}'s matched stream correlates more with its true voice than any other stream does",
          matched3[i] > cross3[i], f"{matched3[i]:.2f} vs {cross3[i]:.2f}")
    check(f"{name}'s matched correlation is reasonably strong (>0.35)", matched3[i] > 0.35, f"{matched3[i]:.2f}")

check_mask_conservation(voices3, mix3)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) did not pass: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")
