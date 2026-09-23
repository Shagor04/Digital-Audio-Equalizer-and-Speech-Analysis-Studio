"""
core/audio_io.py
=================
WAV read/write (via scipy.io.wavfile, no extra native deps needed) plus a
synthetic test-signal generator for developing/demoing the EQ without a
real audio file.
"""

import numpy as np
from scipy.io import wavfile


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


def _synthetic_talker(n, fs, seed, base_f0, wobble_hz, wobble_depth,
                       syllable_period, syllable_on_frac, harmonics):
    """
    One synthetic "talker": a wandering-pitch tone (mimicking natural
    speech intonation) built from a fundamental + harmonics, gated on
    and off in random speech-like "syllable" bursts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    f0 = base_f0 * (1 + wobble_depth * np.sin(2 * np.pi * wobble_hz * t + rng.uniform(0, 2 * np.pi)))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    tone = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        tone += amp * np.sin(k * phase)

    env = np.zeros(n)
    pos = 0
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
            env[pos:pos + on_len] = e
        pos += seg_len

    return tone * env


def make_two_voice_test_signal(fs=44100, duration=6.0, seed=2, return_components=False):
    """
    Synthetic "two people talking over each other" demo track, built
    from scratch (no external samples, so it stays copyright-free and
    the project stays self-contained) -- there's no bundled multi-
    speaker recording, so this exists to try the Voice Separation
    feature immediately.

    Voice A sits in a lower register (~110 Hz), Voice B in a higher one
    (~210 Hz), and their "syllable" bursts are timed independently so
    they frequently overlap -- exactly the "multiple people talking at
    once" case the harmonic-separation method targets.
    """
    n = int(fs * duration)
    voice_a = _synthetic_talker(n, fs, seed + 10, base_f0=110, wobble_hz=2.2, wobble_depth=0.05,
                                 syllable_period=0.35, syllable_on_frac=0.7,
                                 harmonics=[1.0, 0.6, 0.35, 0.2, 0.1, 0.05])
    voice_b = _synthetic_talker(n, fs, seed + 20, base_f0=210, wobble_hz=2.6, wobble_depth=0.06,
                                 syllable_period=0.3, syllable_on_frac=0.75,
                                 harmonics=[1.0, 0.5, 0.25, 0.12, 0.06])

    rng = np.random.default_rng(seed)
    mix = 0.5 * voice_a + 0.5 * voice_b + 0.005 * rng.standard_normal(n)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.9

    if return_components:
        return mix, fs, voice_a, voice_b
    return mix, fs
