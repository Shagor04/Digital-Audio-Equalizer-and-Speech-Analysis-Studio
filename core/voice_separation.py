"""
core/voice_separation.py
=========================
Separate two (or three) overlapping talkers from a single mono
recording using classic pitch + frequency-amplitude analysis -- no
machine learning, no downloaded models, just Fourier analysis and
clustering.

Two cues are combined to tell voices apart:

  1. PITCH (fundamental frequency) -- different people speak in
     different pitch registers. Detected per-frame with the Harmonic
     Product Spectrum (HPS), a classic pitch-detection trick going back
     to Parsons (1976), "Separation of speech from interfering speech
     by means of harmonic selection": downsample the magnitude spectrum
     by 2x, 3x, 4x... and multiply the copies together. A true
     fundamental's harmonics line up under every downsampling, so it
     lights up in the product; noise doesn't.

  2. FREQUENCY AMPLITUDE PROFILE -- how a person's vocal tract shapes
     their voice's energy across frequency bands (formants) also
     differs from person to person, independently of pitch. For every
     frame we measure how much energy sits in each of several
     frequency bands (bass, low-mid, mid, ... treble) -- a coarse
     spectral "fingerprint" -- the same kind of feature used in classic
     (pre-ML) speaker/timbre analysis.

Pipeline:
  1. STFT the mixture into short overlapping frames.
  2. In every frame: find up to `n_voices` pitch candidates via HPS,
     AND compute that frame's amplitude-by-frequency-band profile.
  3. Cluster all detected pitches across the whole recording into
     `n_voices` registers with k-means (pitch alone -- it's the
     strongest, most reliable cue, so it isn't diluted by other
     features here). Assign each frame's pitch candidate(s) to its
     nearest register and build a harmonic-comb mask there
     (raised-cosine bumps on the pitch and its harmonics --
     "harmonic selection").
  4. Learn each voice's amplitude fingerprint by averaging the
     amplitude profile of every frame that voice was directly,
     confidently detected in via pitch.
  5. Anything a frame's harmonic combs don't claim -- including entire
     frames with no detected pitch at all, like unvoiced consonants --
     is split among the voices in proportion to how well each voice's
     amplitude fingerprint matches *that frame's own* amplitude
     profile. This is the frequency-amplitude segregation step: even
     with no pitch to go on, a frame that "sounds like" voice 2's
     usual spectral shape gets assigned mostly to voice 2.
  6. Multiply the mixture's STFT by each voice's mask and inverse-STFT
     back to the time domain -- one approximate, isolated stream per voice.

Honest limitations (worth knowing before trusting the output):
  - Pitch is still the strongest, most precise cue -- voices with very
    similar pitch AND similar amplitude fingerprints will still bleed
    into each other.
  - The amplitude fingerprint is a coarse, whole-recording average, not
    a moment-to-moment measurement of "who is louder right now" -- it
    answers "whose usual frequency shape does this frame resemble",
    not "is this bin louder for voice A than voice B".
  - This is meant for listening / per-voice EQ tweaking on a course
    project, not studio-grade isolation. Some bleed-through is normal.
"""

import numpy as np
from scipy.signal import stft, istft


def _kmeans_nd(X, k, n_iter=25):
    """
    Bare-bones n-dimensional k-means (Lloyd's algorithm) -- the same
    kind of vector quantization used in classic speech coding, plain
    numerical clustering, not a trained model. X has shape
    (n_samples, n_features); works for n_features == 1 too.
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # Deterministic init: spread initial centroids evenly along whichever
    # feature has the most variance (usually pitch), so runs are repeatable.
    # Using the 10th-90th percentile range (not the raw min/max) keeps a
    # handful of outlier detections from being picked as a starting seed
    # and dragging the whole clustering off course.
    axis = int(np.argmax(X.var(axis=0)))
    order = np.argsort(X[:, axis])
    percentiles = np.linspace(10, 90, k)
    pick = (percentiles / 100.0 * (n_samples - 1)).astype(int)
    centroids = X[order[pick]].copy()

    for _ in range(n_iter):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            in_cluster = X[labels == c]
            if len(in_cluster) > 0:
                new_centroids[c] = in_cluster.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels


def _hps_pitch_peaks(mag_spectrum, freqs, fmin, fmax, n_peaks, n_harmonics=5, min_separation_hz=15.0):
    """
    Harmonic Product Spectrum: multiply the magnitude spectrum by
    downsampled copies of itself, which reinforces true fundamental
    frequencies (their harmonics all line up under downsampling) while
    washing out everything else. Returns up to `n_peaks` (freq, strength)
    candidates found in [fmin, fmax], at least min_separation_hz apart.
    """
    hps = mag_spectrum.copy()
    for h in range(2, n_harmonics + 1):
        downsampled = mag_spectrum[::h]
        hps[: len(downsampled)] *= downsampled

    valid = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(valid) < 3:
        return []
    hps_v = hps[valid]
    freqs_v = freqs[valid]

    peaks = []
    for i in range(1, len(hps_v) - 1):
        if hps_v[i] > hps_v[i - 1] and hps_v[i] >= hps_v[i + 1]:
            peaks.append((freqs_v[i], hps_v[i]))
    if not peaks:
        return []
    peaks.sort(key=lambda p: -p[1])

    chosen = []
    for f, s in peaks:
        if all(abs(f - cf) >= min_separation_hz for cf, _ in chosen):
            chosen.append((f, s))
        if len(chosen) >= n_peaks:
            break
    return chosen


def _harmonic_bump_mask(freqs, pitch_hz, nyquist_hz, max_harmonics=20,
                         rel_bandwidth=0.06, min_bandwidth_hz=25.0):
    """
    Build a 0..1 mask over `freqs` that is 1.0 right on top of pitch_hz
    and its first `max_harmonics` harmonics, tapering to 0 with a
    raised-cosine (Hann-shaped) bump around each one -- this is the
    "harmonic selection" filter from Parsons' method.
    """
    mask = np.zeros_like(freqs)
    k = 1
    while pitch_hz * k < nyquist_hz and k <= max_harmonics:
        center = pitch_hz * k
        half_width = max(min_bandwidth_hz, rel_bandwidth * center)
        diff = np.abs(freqs - center)
        bump = np.where(diff < half_width, 0.5 * (1.0 + np.cos(np.pi * diff / half_width)), 0.0)
        mask = np.maximum(mask, bump)
        k += 1
    return mask


def _make_band_edges(n_bands, fmin, fmax):
    """Log-spaced frequency band edges, e.g. 8 bands from 100Hz to 8kHz --
    roughly how a person's formants spread their voice's energy out."""
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    return list(zip(edges[:-1], edges[1:]))


def _band_amplitude_profile(mag_col, freqs, band_edges):
    """
    The "frequency amplitude" fingerprint of one STFT frame: log-energy
    inside each frequency band in band_edges. Independent of pitch --
    this is about *how loud each part of the spectrum is*, which tends
    to differ between people because everyone's vocal tract resonates
    (forms formants) a little differently.
    """
    profile = np.zeros(len(band_edges))
    for i, (lo, hi) in enumerate(band_edges):
        sel = (freqs >= lo) & (freqs < hi)
        energy = np.sum(mag_col[sel] ** 2) if np.any(sel) else 0.0
        profile[i] = np.log10(energy + 1e-10)
    return profile


def _standardize(X):
    """Z-score each column of X (so different-scale features, like Hz vs
    log-energy, contribute comparably to a distance calculation). Returns
    (standardized_X, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (X - mean) / std, mean, std


def _softmax_weights(distances, temperature=1.5):
    """Turn a list of distances (smaller = more similar) into positive
    weights that sum to 1, favoring the closest match(es)."""
    d = np.asarray(distances, dtype=float) / temperature
    d = d - d.min()  # for numerical stability
    w = np.exp(-d)
    return w / w.sum()


def separate_voices(x, fs, n_voices=2, frame_size=2048, hop=512,
                     fmin=70.0, fmax=400.0, n_harmonics=5, pitch_nfft=8192,
                     amp_bands=8, amp_fmin=100.0, amp_fmax=8000.0):
    """
    Separate `x` into `n_voices` approximate, isolated streams using
    pitch AND frequency-amplitude (spectral fingerprint) segregation.

    Returns (voices, info):
      voices -- list of n_voices numpy arrays, each the same length as x
      info   -- dict with diagnostics for the UI:
                  "pitch_hz"          : each voice's detected pitch register (low to high)
                  "active_fraction"   : fraction of frames each voice had a directly-detected pitch in
                  "amplitude_profile" : each voice's amplitude-by-band fingerprint (for display/debugging)
    """
    x = np.asarray(x, dtype=float)
    noverlap = frame_size - hop
    nyquist = fs / 2
    amp_fmax = min(amp_fmax, nyquist - 1.0)
    band_edges = _make_band_edges(amp_bands, amp_fmin, amp_fmax)

    # A zero-padded (finer-resolution) spectrum just for pitch detection --
    # HPS needs closely-spaced bins to tell two similar pitches apart.
    f_fine, _, Z_fine = stft(x, fs=fs, nperseg=frame_size, noverlap=noverlap, nfft=pitch_nfft)
    mag_fine = np.abs(Z_fine)

    # The native-resolution spectrum actually used for masking + resynthesis.
    f, _, Z = stft(x, fs=fs, nperseg=frame_size, noverlap=noverlap)
    n_bins, n_frames = Z.shape

    # Step 1: per-frame pitch candidates (HPS) AND amplitude-band profile
    frame_candidates = []
    frame_amp_profile = np.zeros((n_frames, amp_bands))
    for i in range(n_frames):
        col = mag_fine[:, i]
        frame_amp_profile[i] = _band_amplitude_profile(col, f_fine, band_edges)

        frame_max = col.max() if col.size else 0.0
        if frame_max < 1e-9:
            frame_candidates.append([])
            continue
        cands = _hps_pitch_peaks(col, f_fine, fmin, fmax, n_voices, n_harmonics)
        # throw out candidates that are weak relative to this frame's
        # loudest bin -- likely noise, not a real voiced pitch
        strong = []
        for freq_hz, strength in cands:
            bin_idx = np.argmin(np.abs(f_fine - freq_hz))
            if col[bin_idx] > 0.12 * frame_max:
                strong.append((freq_hz, strength))
        frame_candidates.append(strong)

    # Step 2: cluster the pooled pitch values ALONE into n_voices pitch
    # registers first. (Folding the 8-dimensional amplitude profile
    # directly into this clustering was tried and made things worse --
    # with many more amplitude dimensions than the single pitch
    # dimension, amplitude ends up numerically dominating the distance
    # metric even after standardization, dragging pitch registers off
    # target. Pitch alone is the strongest, most reliable cue, so it
    # gets clustered on its own.)
    pooled_pitch = [freq_hz for cands in frame_candidates for freq_hz, _ in cands]
    if len(pooled_pitch) < n_voices * 5:
        # not enough clearly-pitched material to tell voices apart --
        # fall back to evenly-spaced registers so the app still returns something
        pitch_centroids = np.linspace(fmin, fmax, n_voices)
    else:
        pitch_z, p_mean, p_std = _standardize(np.array(pooled_pitch)[:, None])
        centroids_z, _ = _kmeans_nd(pitch_z, n_voices)
        pitch_centroids = (centroids_z * p_std + p_mean)[:, 0]
    pitch_centroids = np.sort(pitch_centroids)  # voice 0 = lowest register, ...

    # Step 3: assign each frame's pitch candidate(s) to the nearest register
    assigned_pitch = np.zeros((n_voices, n_frames))
    for i, cands in enumerate(frame_candidates):
        used = set()
        for freq_hz, _ in cands:
            for v in np.argsort(np.abs(pitch_centroids - freq_hz)):
                if v not in used:
                    assigned_pitch[v, i] = freq_hz
                    used.add(v)
                    break

    # Step 4: harmonic-comb mask wherever a voice got a pitch this frame
    masks = np.zeros((n_voices, n_bins, n_frames))
    for i in range(n_frames):
        for v in range(n_voices):
            pitch = assigned_pitch[v, i]
            if pitch > 0:
                masks[v, :, i] = _harmonic_bump_mask(f, pitch, nyquist)

    # Step 5: FREQUENCY-AMPLITUDE SEGREGATION. First, learn each voice's
    # characteristic amplitude-by-frequency-band fingerprint by averaging
    # the amplitude profile of every frame that voice was confidently,
    # directly detected in (via pitch, above). Then, whatever energy the
    # harmonic combs didn't claim in a frame -- including entire frames
    # with no detected pitch at all, like unvoiced consonants -- is split
    # among the voices in proportion to how well each voice's amplitude
    # fingerprint matches *that frame's own* amplitude profile, instead
    # of guessing or splitting evenly.
    amp_centroids = np.zeros((n_voices, amp_bands))
    for v in range(n_voices):
        own_frames = assigned_pitch[v] > 0
        if np.any(own_frames):
            amp_centroids[v] = frame_amp_profile[own_frames].mean(axis=0)

    amp_z_all, amp_mean, amp_std = _standardize(frame_amp_profile)
    amp_centroids_z = (amp_centroids - amp_mean) / amp_std

    total = masks.sum(axis=0)                       # shape (n_bins, n_frames)
    leftover = np.clip(1.0 - total, 0.0, None)
    for i in range(n_frames):
        dists = np.linalg.norm(amp_centroids_z - amp_z_all[i][None, :], axis=1)
        weights = _softmax_weights(dists)
        for v in range(n_voices):
            masks[v, :, i] += leftover[:, i] * weights[v]

    # Step 6: never amplify above the original level
    total2 = masks.sum(axis=0)
    scale = np.where(total2 > 1.0, 1.0 / total2, 1.0)
    masks = masks * scale[None, :, :]

    # Step 7: apply each mask and inverse-STFT back to a waveform
    voices = []
    for v in range(n_voices):
        Zv = Z * masks[v]
        _, yv = istft(Zv, fs=fs, nperseg=frame_size, noverlap=noverlap)
        if len(yv) > len(x):
            yv = yv[: len(x)]
        elif len(yv) < len(x):
            yv = np.pad(yv, (0, len(x) - len(yv)))
        voices.append(yv)

    info = {
        "pitch_hz": pitch_centroids.tolist(),
        "active_fraction": [(assigned_pitch[v] > 0).mean() for v in range(n_voices)],
        "amplitude_profile": amp_centroids.tolist(),
    }
    return voices, info
