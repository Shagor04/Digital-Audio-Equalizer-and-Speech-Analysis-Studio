"""
core/audio_io.py
=================
WAV read/write (via scipy.io.wavfile, no extra native deps needed) plus
synthetic test-signal generators for developing/demoing the EQ, voice
separation, and noise reduction without needing real audio files.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

from core.filters import peaking_eq_coeffs


def load_wav(path):
    """Return (signal, fs). Signal is float64 in [-1, 1], mono (stereo is
    averaged down to mono for simplicity of the DSP pipeline)."""
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128) / 128.0
    else:  # already float
        x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def save_wav(path, x, fs):
    """Save a float signal (range ~[-1, 1]) as 16-bit PCM wav."""
    x = np.clip(x, -1.0, 1.0)
    data = (x * 32767).astype(np.int16)
    wavfile.write(path, fs, data)


def check_and_normalize(y: np.ndarray, headroom_db: float = 0.5):
    """
    Clipping protection: EQ boosts (especially stacking several +dB bands)
    can easily push |y[n]| above 1.0, which would hard-clip when saved to
    16-bit PCM. Detect that and, if needed, scale the whole signal down so
    its peak sits `headroom_db` below full scale (0 dBFS) instead of
    clipping. Returns (normalized_signal, info_dict) where info_dict is
    always populated so the caller can report "no clipping needed" too.
    """
    peak_before = float(np.max(np.abs(y))) if len(y) else 0.0
    target_peak = 10 ** (-headroom_db / 20)  # e.g. 0.5dB headroom -> ~0.944

    if peak_before > 1.0:
        y_norm = y / peak_before * target_peak
        clipped = True
    else:
        y_norm = y
        clipped = False

    info = {
        "peak_before": peak_before,
        "peak_after": float(np.max(np.abs(y_norm))) if len(y_norm) else 0.0,
        "clipping_detected": clipped,
        "headroom_db": headroom_db,
    }
    return y_norm, info


def make_synthetic_test_signal(fs=44100, duration=4.0, seed=0):
    """
    Build a synthetic test signal that deliberately has energy spread
    across the spectrum, so EQ boosts/cuts are audible and visible:
      - a low "kick"-like thump (60 Hz, amplitude-enveloped, repeating)
      - a mid-range "vocal-ish" tone with harmonics (220 Hz + harmonics,
        slowly vibrato'd)
      - a high "hi-hat"-like burst of filtered noise, repeating
      - a little broadband noise floor throughout
    This is NOT meant to sound musical -- it's a controlled test signal
    so each EQ band's effect can be clearly seen/heard.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 1. low thump every 0.5s: 60 Hz tone with a fast exponential decay envelope
    thump = np.zeros(n)
    period = 0.5
    for onset in np.arange(0, duration, period):
        idx = int(onset * fs)
        seg_len = int(0.15 * fs)
        seg_len = min(seg_len, n - idx)
        if seg_len <= 0:
            continue
        tt = np.arange(seg_len) / fs
        env = np.exp(-tt * 25)
        thump[idx:idx + seg_len] += 0.9 * env * np.sin(2 * np.pi * 60 * tt)

    # 2. mid-range tone with harmonics + slow vibrato (vocal-ish)
    vibrato = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
    f0 = 220 * vibrato
    phase = 2 * np.pi * np.cumsum(f0) / fs
    voice = 0.25 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)
    voice *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # slow amplitude swell

    # 3. high "hi-hat" bursts: bursts of noise, high-pass-ish via differencing
    hihat = np.zeros(n)
    period_h = 0.25
    for onset in np.arange(0, duration, period_h):
        idx = int(onset * fs)
        seg_len = int(0.03 * fs)
        seg_len = min(seg_len, n - idx)
        if seg_len <= 0:
            continue
        burst = rng.standard_normal(seg_len)
        burst = np.diff(burst, prepend=0)  # crude high-pass emphasis
        env = np.exp(-np.arange(seg_len) / fs * 60)
        hihat[idx:idx + seg_len] += 0.3 * env * burst

    # 4. broadband noise floor
    noise = 0.01 * rng.standard_normal(n)

    x = thump + voice + hihat + noise
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.9  # normalize, leave headroom
    return x, fs


def _adsr_envelope(n_samples, fs, attack=0.02, decay=0.08, sustain_level=0.6, release=0.15):
    """Simple Attack-Decay-Sustain-Release envelope, length n_samples."""
    a = int(attack * fs)
    d = int(decay * fs)
    r = int(release * fs)
    s = max(n_samples - a - d - r, 0)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False) if a > 0 else np.array([]),
        np.linspace(1, sustain_level, d, endpoint=False) if d > 0 else np.array([]),
        np.full(s, sustain_level),
        np.linspace(sustain_level, 0, r) if r > 0 else np.array([]),
    ])
    if len(env) < n_samples:
        env = np.concatenate([env, np.zeros(n_samples - len(env))])
    return env[:n_samples]


def _note(freq, duration, fs, harmonics=(1.0, 0.5, 0.25, 0.12), **adsr_kwargs):
    """Render one note as a sum of harmonically-related sines under an ADSR envelope
    (a crude but pleasant additive-synthesis 'instrument')."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    wave = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        wave += amp * np.sin(2 * np.pi * freq * k * t)
    wave /= sum(harmonics)
    env = _adsr_envelope(n, fs, **adsr_kwargs)
    return wave * env


def make_demo_music_signal(fs=44100, bpm=100, bars=8, seed=1):
    """
    A more 'musical' synthetic demo track than make_synthetic_test_signal():
    a simple bass line, a mid-range chord/melody pad, and a rhythmic
    hi-hat layer, all built from scratch with additive synthesis (no
    external samples needed -- keeps the project self-contained and
    copyright-free). Energy is spread across low/mid/high bands on
    purpose so every graphic-EQ band and every parametric preset has an
    audible, obvious effect for a demo.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration = total_beats * beat
    n_total = int(duration * fs) + fs  # a little tail room
    mix = np.zeros(n_total)

    # --- Bass line: walking pattern over a simple I-vi-IV-V-ish loop (low octave) ---
    bass_pattern_hz = [65.41, 65.41, 87.31, 87.31, 73.42, 73.42, 98.00, 98.00]  # C2 C2 F2 F2 D2 D2 G2 G2
    for beat_idx in range(total_beats):
        freq = bass_pattern_hz[beat_idx % len(bass_pattern_hz)]
        note = _note(freq, beat * 0.95, fs, harmonics=(1.0, 0.6, 0.3, 0.15, 0.08),
                     attack=0.005, decay=0.1, sustain_level=0.7, release=0.25)
        start = int(beat_idx * beat * fs)
        end = start + len(note)
        if end <= n_total:
            mix[start:end] += 0.5 * note

    # --- Mid pad/melody: a chord tone changes every 2 beats, plus a simple melody on top ---
    chord_roots = [130.81, 130.81, 174.61, 174.61, 146.83, 146.83, 196.00, 196.00]  # one octave above bass
    melody_offsets = [0, 4, 7, 4, 0, 7, 4, 0, 3, 7, 5, 3, 0, 4, 7, 12]  # semitone-ish steps (approx, musical flavor)
    for beat_idx in range(total_beats):
        root = chord_roots[beat_idx % len(chord_roots)]
        # sustained chord tone (root + fifth) every 2 beats
        if beat_idx % 2 == 0:
            chord = _note(root, beat * 2 * 0.95, fs, harmonics=(1.0, 0.4, 0.2),
                          attack=0.05, decay=0.3, sustain_level=0.5, release=0.4)
            chord += _note(root * 1.5, beat * 2 * 0.95, fs, harmonics=(1.0, 0.3),
                           attack=0.05, decay=0.3, sustain_level=0.4, release=0.4)
            start = int(beat_idx * beat * fs)
            end = start + len(chord)
            if end <= n_total:
                mix[start:end] += 0.18 * chord

        # short melody note on the offbeat, up an octave, using the scale offsets
        semitone = melody_offsets[beat_idx % len(melody_offsets)]
        mel_freq = root * 2 * (2 ** (semitone / 12))
        mel = _note(mel_freq, beat * 0.4, fs, harmonics=(1.0, 0.5, 0.2, 0.1),
                    attack=0.01, decay=0.05, sustain_level=0.5, release=0.15)
        start = int((beat_idx * beat + beat * 0.5) * fs)
        end = start + len(mel)
        if end <= n_total:
            mix[start:end] += 0.14 * mel

    # --- Hi-hat rhythm: eighth notes, filtered noise bursts (high-frequency energy) ---
    eighth = beat / 2
    n_eighths = total_beats * 2
    for i in range(n_eighths):
        seg_len = int(0.04 * fs)
        idx = int(i * eighth * fs)
        seg_len = min(seg_len, n_total - idx)
        if seg_len <= 0:
            continue
        burst = rng.standard_normal(seg_len)
        burst = np.diff(burst, prepend=0)
        burst = np.diff(burst, prepend=0)  # double-differencing = stronger high-pass emphasis
        accent = 0.28 if i % 4 == 0 else 0.14  # accent the downbeats
        env = np.exp(-np.arange(seg_len) / fs * 70)
        mix[idx:idx + seg_len] += accent * env * burst

    # --- a little broadband noise floor for realism ---
    mix += 0.004 * rng.standard_normal(n_total)

    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9
    return mix, fs



# ---------------------------------------------------------------------------
# Voice-like signals for the Voice Separation demos
# ---------------------------------------------------------------------------
# Classic formant frequencies (F1, F2, F3 in Hz) for a handful of vowel
# sounds -- approximate values from the widely-used Peterson & Barney
# vowel chart. These are the resonant frequencies a human vocal tract
# naturally emphasizes for each vowel; boosting a harmonic "buzz" at
# these three frequencies is the classic *source-filter* model of speech
# synthesis (buzzy vocal-cord source, shaped by vocal-tract resonance),
# and it's what makes two voices sound like genuinely different people
# rather than just two different pitches.
VOWEL_FORMANTS = {
    "ah": (700, 1220, 2600),   # as in "father"
    "ih": (400, 1920, 2560),   # as in "sit"
    "oh": (450, 800, 2830),    # as in "go"
    "oo": (325, 700, 2530),    # as in "boot"
    "eh": (530, 1840, 2480),   # as in "bed"
    "ee": (270, 2300, 3000),   # as in "see"
}


def _apply_formants(x, fs, formants, gains=(9.0, 6.0, 4.0), q=9.0):
    """
    Color a raw harmonic buzz into something much more vowel-like by
    boosting three resonant bands (formants) -- reuses this project's own
    peaking-EQ biquad design (core/filters.py), the exact same building
    block the Parametric EQ uses, just applied to a synthesis signal
    instead of a loaded recording.
    """
    y = x.copy()
    for f0, gain in zip(formants, gains):
        b, a = peaking_eq_coeffs(f0, gain, q, fs)
        y = lfilter(b, a, y)
    return y


def _voice_talker(n, fs, seed, base_f0, wobble_hz, wobble_depth,
                   syllable_period, syllable_on_frac, vowels, n_harmonics=16):
    """
    One synthetic "talker": a wandering-pitch harmonic buzz (mimicking
    natural speech intonation), colored syllable-by-syllable into
    different vowel sounds via formant filtering, and gated on/off in
    speech-like bursts. Two talkers with different (pitch, vowel set,
    rhythm) combinations sound like genuinely different people -- not
    just a different note, but a different *voice* -- which exercises
    both cues (pitch and frequency-amplitude shape) the voice-separation
    algorithm itself relies on.

    Each vowel's coloring is computed by filtering the *entire* buzz
    once, continuously start-to-end, rather than re-running the filter
    fresh on each short syllable -- that avoids the filter-reset "click"
    a from-zero-state filter would otherwise produce at every syllable
    boundary. Which pre-colored version plays is then just a per-syllable
    selection, like switching between two already-running instruments.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    f0 = base_f0 * (1 + wobble_depth * np.sin(2 * np.pi * wobble_hz * t + rng.uniform(0, 2 * np.pi)))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    max_harm = max(2, min(n_harmonics, int((fs / 2) / max(base_f0, 1.0)) - 1))
    buzz = np.zeros(n)
    for k in range(1, max_harm):
        buzz += (1.0 / k) * np.sin(k * phase)  # natural-ish rolloff, like a real glottal buzz

    colored_versions = [_apply_formants(buzz, fs, VOWEL_FORMANTS[v]) for v in vowels]

    out = np.zeros(n)
    pos = 0
    vi = 0
    while pos < n:
        seg_len = int(syllable_period * fs * (0.6 + 0.8 * rng.random()))
        on_len = int(seg_len * syllable_on_frac * (0.7 + 0.6 * rng.random()))
        on_len = min(on_len, n - pos)
        if on_len > 0:
            ramp = min(int(0.02 * fs), on_len // 2)
            e = np.ones(on_len)
            if ramp > 0:
                e[:ramp] = np.linspace(0, 1, ramp)
                e[-ramp:] = np.linspace(1, 0, ramp)
            src = colored_versions[vi % len(colored_versions)]
            out[pos:pos + on_len] = src[pos:pos + on_len] * e
            vi += 1
        pos += seg_len

    out = out / (np.max(np.abs(out)) + 1e-9) * 0.9  # balance this talker's own loudness
    return out


def make_two_voice_test_signal(fs=44100, duration=7.0, seed=2, return_components=False):
    """
    Synthetic "two people talking over each other" demo track. There is
    no bundled multi-speaker recording, so this exists to try the Voice
    Separation feature immediately -- and to actually *sound* like two
    different people, not just two different pitches, each talker gets
    its own pitch register, its own pair of vowel colors, and its own
    speaking rhythm:

      Voice 1: lower register (~105Hz), "ah"/"oh" vowels, slower pace
      Voice 2: higher register (~220Hz), "ee"/"eh" vowels, faster pace

    Their "syllable" bursts are timed independently so they frequently
    overlap -- exactly the "multiple people talking at once" case the
    harmonic-separation method targets.
    """
    n = int(fs * duration)
    voice_1 = _voice_talker(n, fs, seed + 10, base_f0=105, wobble_hz=2.0, wobble_depth=0.05,
                             syllable_period=0.40, syllable_on_frac=0.72,
                             vowels=["ah", "oh"])
    voice_2 = _voice_talker(n, fs, seed + 20, base_f0=220, wobble_hz=2.8, wobble_depth=0.06,
                             syllable_period=0.28, syllable_on_frac=0.75,
                             vowels=["ee", "eh"])

    rng = np.random.default_rng(seed)
    mix = 0.5 * voice_1 + 0.5 * voice_2 + 0.004 * rng.standard_normal(n)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9

    if return_components:
        return mix, fs, voice_1, voice_2
    return mix, fs


def make_three_voice_test_signal(fs=44100, duration=9.0, seed=3, return_components=False):
    """
    Synthetic "three people talking over each other" demo track, for
    trying Voice Separation with n_voices=3. Same idea as the two-voice
    demo, extended: three talkers, each with a clearly different pitch
    register, vowel-color pair, and rhythm, so a human listener can tell
    them apart by ear -- which is also exactly what gives the pitch and
    frequency-amplitude cues something to work with.

      Voice 1: low register (~95Hz), "ah"/"oo" vowels, slowest pace
      Voice 2: mid register (~165Hz), "eh"/"oh" vowels, medium pace
      Voice 3: high register (~250Hz), "ee"/"ih" vowels, fastest pace
    """
    n = int(fs * duration)
    voice_1 = _voice_talker(n, fs, seed + 10, base_f0=95, wobble_hz=1.8, wobble_depth=0.05,
                             syllable_period=0.45, syllable_on_frac=0.70,
                             vowels=["ah", "oo"])
    voice_2 = _voice_talker(n, fs, seed + 20, base_f0=165, wobble_hz=2.4, wobble_depth=0.05,
                             syllable_period=0.32, syllable_on_frac=0.72,
                             vowels=["eh", "oh"])
    voice_3 = _voice_talker(n, fs, seed + 30, base_f0=250, wobble_hz=3.0, wobble_depth=0.06,
                             syllable_period=0.24, syllable_on_frac=0.75,
                             vowels=["ee", "ih"])

    rng = np.random.default_rng(seed)
    mix = (voice_1 + voice_2 + voice_3) / 3 + 0.004 * rng.standard_normal(n)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9

    if return_components:
        return mix, fs, voice_1, voice_2, voice_3
    return mix, fs


# ---------------------------------------------------------------------------
# Noise reduction demo signal
# ---------------------------------------------------------------------------
def make_noise_demo_signal(fs=44100, seed=5, return_components=False):
    """
    A short, clean, recognizable melodic phrase (C-E-G-E-D-C, repeated
    three times with clear silence in between) buried under obvious
    broadband hiss and 50Hz+100Hz mains hum -- built specifically to make
    noise reduction audibly obvious.

    Two things about this signal are deliberate, not incidental:
      1. The clear SILENCE between notes gives spectral gating (which
         estimates a per-frequency noise floor from the *median* level
         across time -- see core/noise_reduction.py) genuinely quiet
         material to measure that floor from. A wall-to-wall signal with
         no gaps would defeat that method entirely.
      2. The hum is a strong, *continuous* tone -- present in every frame,
         not just the gaps -- so spectral gating's floor-based approach
         alone will NOT remove it well (its median magnitude simply looks
         like "the signal" at that frequency). That's not a bug to work
         around here; it's the honest reason a dedicated notch filter
         (the "Remove 50Hz hum" checkbox) exists as a *separate* tool.
         Loading this demo turns both on, so Play Equalized removes the
         hiss (via spectral gating) and the hum (via the notch) together.
    """
    notes_hz = [261.63, 329.63, 392.00, 329.63, 293.66, 261.63]  # C4 E4 G4 E4 D4 C4
    note_dur, gap_dur, phrase_gap, n_repeats = 0.35, 0.15, 0.6, 3

    clean_notes = [_note(f, note_dur, fs, harmonics=(1.0, 0.5, 0.25, 0.12),
                          attack=0.01, decay=0.08, sustain_level=0.6, release=0.12)
                   for f in notes_hz]
    gap = np.zeros(int(gap_dur * fs))
    phrase = np.concatenate(sum(([note, gap] for note in clean_notes), []))
    phrase_gap_arr = np.zeros(int(phrase_gap * fs))
    clean = np.concatenate(sum(([phrase, phrase_gap_arr] for _ in range(n_repeats)), []))
    edge = np.zeros(int(0.3 * fs))
    clean = np.concatenate([edge, clean, edge])

    n = len(clean)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    hiss = 0.05 * rng.standard_normal(n)
    hum = 0.07 * np.sin(2 * np.pi * 50 * t) + 0.03 * np.sin(2 * np.pi * 100 * t)

    noisy = 0.6 * clean + hiss + hum
    noisy = noisy / (np.max(np.abs(noisy)) + 1e-9) * 0.9

    if return_components:
        return noisy, fs, clean
    return noisy, fs

