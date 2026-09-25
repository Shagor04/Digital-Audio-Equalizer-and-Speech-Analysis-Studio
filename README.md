# Digital Audio Equalizer — CSE220 Project

A software audio equalizer built from first principles for the Signals &
Systems course: every filter is derived from the theory (sinc-based FIR
design, Fourier convolution theorem, IIR difference equations, bilinear
transform) rather than treated as a black box.

## Project status (built step-by-step)

- [x] **Step 1 — Core DSP engine** (`core/filters.py`, `core/eq_engine.py`)
  - FIR band-pass filter bank via windowed-sinc design → `GraphicEQ`
  - IIR biquad peaking/shelving filters (RBJ cookbook, bilinear transform) → `ParametricEQ`
  - FFT-based convolution (Convolution Theorem) for production-speed processing
  - From-scratch direct convolution + from-scratch IIR recurrence, kept as
    verification references against `scipy`
- [x] **Step 2 — Presets + audio file I/O + CLI** (`main.py`, `presets/`, `core/audio_io.py`)
  - `.wav` load/save (16-bit PCM, mono-downmixed)
  - Synthetic test-signal generator (thump + vibrato tone + hi-hat bursts + noise floor) so the project runs without needing a sample file
  - 8 graphic presets (bass_boost, treble_boost, vocal_boost, v_shape, ...) and 5 parametric presets (vocal_presence, de_ess, telephone, ...)
  - Shared-color-scale before/after spectrograms (so EQ changes are actually visible, not hidden by independent auto-normalization)
- [x] **Step 3 — GUI with live sliders** (`gui/app.py`)
  - Tkinter interface with a Graphic EQ tab (10 vertical sliders) and a Parametric EQ tab (4 configurable bands: type/freq/gain/Q, each toggleable) via a tab switcher
  - Live embedded matplotlib frequency-response plot that updates instantly as sliders move (cheap — recomputed from filter coefficients, not by reprocessing audio)
  - Presets dropdown per mode, Load/Export WAV, Play Original / Play Equalized / Stop via `sounddevice`
  - Screenshot-tested headlessly (Xvfb) to confirm layout renders correctly before delivery
- [ ] Step 4 — (stretch) Real-time microphone processing
- [x] **Step 5 — DSP Analysis & extra processing** (`core/analysis.py`, `core/noise_reduction.py`)
  - Quantitative metrics: RMS, peak, energy, spectral centroid, dominant frequency, zero-crossing rate — computed before/after and printed as a comparison table (CLI) or shown live in a metrics table (GUI)
  - Before/after magnitude-spectrum overlay plot, and a combined analysis dashboard image (waveform + spectrum + spectrogram + metrics table in one figure) for the report
  - 50 Hz mains-hum notch filter (+ 2nd harmonic), tuned so it removes the hum tone by ~20 dB while leaving nearby content within 0.01 dB — verified with a synthetic-hum test
  - Basic noise reduction: low-pass, high-pass, band-stop, and spectral gating (STFT-based noise gate, tuned to cut noise-floor RMS by >20 dB while barely touching real signal)
  - New "DSP Analysis" tab in the GUI: live metrics table + spectrum plot for whatever's currently loaded/processed
  - New "Post-processing" row in the GUI: toggle 50 Hz hum removal and pick a noise-reduction mode, applied after the EQ stage
- [x] **Step 6 — Filter theory analysis & report generation** (`core/filter_analysis.py`, `core/report.py`)
  - Phase response and group delay for both EQ types, computed uniformly by collapsing each into a single (b, a): GraphicEQ's parallel FIR bank sums to one composite kernel; ParametricEQ's biquad cascade multiplies out via polynomial convolution
  - Verified: the FIR graphic EQ has a **provably linear phase** (group delay constant to 1e-12 ms) while the IIR parametric EQ shows genuinely nonlinear phase/varying group delay near resonances — a clean textbook FIR-vs-IIR contrast
  - IIR stability check via pole magnitudes (|z| < 1)
  - Filter performance metrics: measured gain vs design target, gain error, measured peak frequency, -3dB bandwidth
  - FIR vs IIR structural + **measured** comparison (processing time, coefficient count, stability, phase) — also surfaced a real perf bug (non-FFT-friendly convolution length was making FIR processing ~9x slower than necessary; fixed via `scipy.fft.next_fast_len`)
  - Frequency-band energy distribution (Bass/Low-mid/Mid/High-mid/Treble), before vs after, as a horizontal bar chart
  - Clipping protection: detects `|y[n]| > 1` after EQ and automatically normalizes with configurable headroom, reporting peak-before/peak-after
  - Automatic multi-page PDF report generator (`--report` flag, or "PDF Report..." button in the GUI): title/metrics page, waveform+spectrum+spectrogram page, phase & group delay page, band-energy page, filter-performance + FIR-vs-IIR page
  - New "View" selector in the GUI's DSP Analysis tab: switch between Spectrum / Phase & Group Delay / Band Energy without leaving the tab
- [x] **Step 7 — Testing, portability, and controlled experiments** (`tests/`, `core/experiments.py`, `core/run_experiments.py`)
  - Fixed a real portability bug: test scripts previously had a hard-coded absolute path, which would break on any other machine/clone — all paths now derive from `os.path.dirname(os.path.abspath(__file__))`
  - Converted print-only test scripts into real `assert`-backed automated tests with pass/fail status and non-zero exit codes on failure
  - New `tests/test_end_to_end.py`: full pipeline test (WAV → EQ → hum removal → noise reduction → clipping-protection normalize → WAV), checking sample rate/duration preservation, no NaN/Inf, amplitude bounds, and that each stage's claimed effect is independently measurable — this test caught a real interaction (spectral gating can attenuate sustained bass content it mistakes for steady noise) which is now documented rather than silently wrong
  - Auto-detected filter-performance targets: the CLI and GUI now automatically pick the target frequency/gain from whichever band has the largest |gain|, instead of requiring manually-typed values that can drift out of sync with the actual EQ settings
  - Five controlled experiments (`--experiments` CLI flag, or the GUI's "Experiments" tab): (A) bass-boost gain sweep, (B) Q-factor sweep showing bandwidth narrows as Q increases, (C) a matched FIR-vs-IIR comparison (same center frequency/gain, two implementations), (D) 50Hz hum removal, (E) noise reduction — each produces its own plot plus a combined 10-page PDF report
  - Corrected README wording: "from first principles" was imprecise since production processing uses `scipy.signal.lfilter`/`stft` — now states plainly that the theory/design is original and verified from scratch, while execution uses optimized, vetted libraries (a stronger and more honest engineering story)
- [x] **Step 8 — Voice separation + live waveform (GUI)** (`core/voice_separation.py`, `gui/app.py`)
  - **Overlapping-voice separation**, no ML/downloaded models, using two combined
    classic-DSP cues:
    - **Pitch** -- STFT the mixture, detect up to N simultaneous pitches per
      frame via the Harmonic Product Spectrum (after Parsons 1976's "harmonic
      selection"), cluster all detected pitches across the recording into N
      registers with a from-scratch k-means, then build a soft harmonic-comb
      mask per voice per frame.
    - **Frequency amplitude profile** -- each voice's energy-by-frequency-band
      "fingerprint" (bass/low-mid/mid/.../treble, learned from the frames that
      voice was directly pitch-detected in) is used to segregate everything
      the harmonic combs can't reach on pitch alone -- including entire
      unvoiced frames (consonants, fricatives, silence) that have no pitch to
      track at all. Those frames are split among voices by comparing that
      frame's own amplitude-by-band profile against each voice's fingerprint,
      a measurement rather than a guess.
    New **"Voice Separation"** GUI tab: pick 2 or 3 voices, hit **Separate
    Voices**, then each detected voice gets its own 5-band graphic EQ, a
    **Solo** button, and a **Mute** toggle; **Play Mixed** / **Export Mixed
    WAV...** re-combine the (optionally per-voice-EQ'd, non-muted) streams.
    New **"🗣 Two-Voice Demo"** loads a synthetic two-talker
    signal (two independent synthesized voices in different pitch registers,
    overlapping) since there's no bundled multi-speaker recording to test on
    otherwise. (Step 9 below replaced the original buzzy-tone version of this
    demo with proper formant-colored voices, and added a three-voice demo.)
    Automated ground-truth test: `tests/verify_voice_separation.py` builds the
    two-voice demo (whose two source voices are known, since they're
    synthesized), separates it, and checks that each separated stream
    correlates strongly with the *correct* source voice (0.9+ typical).
    **Note on combining the two cues**: folding the amplitude profile directly
    into the *pitch* clustering was tried first and made things worse -- 8
    amplitude dimensions numerically overpowered the single pitch dimension
    even after standardization, dragging pitch registers off target by
    30-100Hz. The two cues are kept in separate stages instead: pitch is
    clustered alone (it's the strongest, most precise cue), and the amplitude
    fingerprint is a secondary measurement used only to resolve what pitch
    can't reach.
    **Honest limits** (also documented in `core/voice_separation.py`): this
    only works when voices sit in noticeably different pitch ranges -- two
    very similar-pitched voices with similar amplitude fingerprints will still
    bleed together. It's built for listening / per-voice EQ tweaking on a
    course project, not studio-grade isolation.
  - **Live waveform**: a "Now Playing" panel, visible on every tab, with a
    scrolling oscilloscope-style trace of whatever's currently playing
    (original, equalized, a solo'd voice, or the separated mix) plus a static
    overview strip of the whole track with a moving playhead and an
    elapsed/total time readout. Purely a Tkinter `Canvas` (no extra matplotlib
    redraw cost), driven by elapsed wall-clock time since Play was pressed, so
    it animates correctly regardless of whether this machine actually has
    audio hardware to hear the result on.

- [x] **Step 9 — Voice-like demo signals, three-voice separation, and an audible noise-reduction demo** (`core/audio_io.py`, `gui/app.py`, `tests/`)
  - **The old two-voice demo didn't sound like two people** -- it was two
    plain harmonic tones differing only in pitch, buzzy and hard to tell
    apart by ear even after separation. Fixed with a proper **source-filter
    voice synthesizer**: each talker is a harmonic "buzz" (the glottal
    source) shaped by three **formant** resonances (the vocal tract filter)
    pulled from the classic Peterson & Barney vowel chart, reusing this
    project's own `peaking_eq_coeffs` biquad (`core/filters.py`) as the
    formant filter -- the exact building block the Parametric EQ uses,
    repurposed to *synthesize* a voice instead of *equalizing* one. Each
    talker also gets its own speaking rhythm (syllable timing), so two or
    three talkers differ in pitch, timbre (vowel color), *and* rhythm --
    three independent cues a human listener can use to tell them apart,
    not just one.
  - **Three-Voice Demo** (new): a third synthetic talker (low/mid/high
    registers at ~95/165/250Hz), for trying `n_voices=3` separation.
    Ground-truth tested exactly like the two-voice case (see below) --
    matched correlations of 0.83-0.91 against the true source voices, a
    genuinely hard 3-way separation problem solved convincingly.
  - **Noise Reduction Demo** (new): a clean, recognizable melody (C-E-G-E-D-C
    played three times with clear silence between repeats) buried under
    obvious broadband hiss and 50Hz+100Hz hum. The silence gaps are
    deliberate -- spectral gating estimates its noise floor from the
    *median* level across time, so it needs genuinely quiet material to
    measure (see `core/noise_reduction.py`); the continuous hum is
    deliberate too, since it demonstrates *why* hum removal has to be a
    separate notch-filter tool rather than something spectral gating alone
    can catch (a tone that's always on looks like "signal", not "noise", to
    a floor-based method). Loading this demo automatically ticks **Remove
    50Hz hum** and sets **Noise reduction** to `spectral_gate`, so Play
    Equalized immediately demonstrates both tools together. Measured
    (`tests/verify_noise_demo.py`): 19.5dB noise-floor reduction in the
    silent gaps, 36.7dB reduction of the 50Hz hum specifically, and 0.997
    correlation between the processed and clean melody (i.e. the real
    content survives essentially intact).
  - **One combined "Demo" picker** replaces the old one-button-per-demo
    toolbar row: a single dropdown (Synthetic Test Tone / Music Track /
    Two-Voice Demo / Three-Voice Demo / Noise Reduction Demo) keeps the
    toolbar from growing a button per feature, with a small `loaded: ...`
    label next to it confirming what's active. Selecting a voice demo
    **automatically switches to the Voice Separation tab and runs
    Separate Voices**; selecting the noise demo **automatically switches to
    the DSP Analysis tab** so the improvement is visible (spectrum + before/
    after metrics), not just audible -- the same "show, don't just tell"
    idea applied consistently everywhere a demo has a point to prove.
  - **Bug found and fixed while wiring this up**: switching demos while the
    Voice Separation tab was showing a previous result left stale numbers
    on screen (e.g. loading the Noise Reduction Demo while "Voice 1:
    ~99Hz..." from an earlier separation was still displayed, describing
    audio that wasn't loaded anymore). `_clear_voice_separation()` now
    resets the summary text to its idle instructions too, not just the
    voice data -- a reminder that "clearing state" means clearing
    everything derived from it, including labels, not just the arrays.
  - Both new signal types are covered by the same ground-truth testing
    approach as Step 8: `tests/verify_voice_separation.py` now checks both
    the two-voice *and* three-voice cases (pitch accuracy + matched-vs-cross
    correlation for every voice), and the new `tests/verify_noise_demo.py`
    checks the noise-reduction demo's measured before/after numbers
    directly, the same "build a signal whose correct answer is known by
    construction" strategy used throughout this project's test suite.

## How each part maps to CSE220 theory

| Code | Theory |
|---|---|
| `_sinc_lowpass_kernel` | Ideal low-pass filter = sinc in time domain; windowing to make it realizable (finite, causal) |
| `design_bandpass_fir` | Linearity of the Fourier Transform: H_bp = H_lp2 − H_lp1 |
| `convolve_fft` | Convolution Theorem: x*h ⟷ X(f)·H(f) |
| `GraphicEQ.process` | Parallel bank-of-filters LTI system |
| `peaking_eq_coeffs` etc. | Analog RLC resonator → bilinear transform → discrete IIR |
| `apply_iir` | Direct-Form-I difference equation (the recurrence relation for an LTI system) |
| `ParametricEQ.process` | Cascade (series) connection of LTI systems → transfer functions multiply |

## Directory layout

```
audio_equalizer/
├── core/
│   ├── filters.py      # filter design + convolution primitives
│   ├── eq_engine.py     # GraphicEQ and ParametricEQ classes
│   └── audio_io.py       # wav load/save + synthetic test-signal generator
├── visualization/
│   └── plots.py         # frequency response / waveform / spectrogram plots
├── presets/
│   └── presets.py        # named graphic + parametric EQ curves
├── gui/                   # (step 3) Tkinter interface
├── tests/
│   └── verify_step1.py   # sanity checks: flatness, boost accuracy, IIR correctness
├── output/                 # generated wav files + plots land here
└── main.py                  # CLI entry point
```

## Running the Step 1 verification

```bash
cd audio_equalizer
python3 tests/verify_step1.py
```

Expected output includes a flat (0.00 dB) composite response with all
graphic-EQ gains at 0 dB, an exact +8 dB bump at 1 kHz for a parametric
peak band set to gain=8, and a from-scratch-vs-scipy IIR error near
machine precision (~1e-14).

## Running the CLI (Step 2)

```bash
cd audio_equalizer

# list all presets
python3 main.py --list-presets

# no wav file needed -- generates a synthetic test signal automatically
python3 main.py --synthetic --mode graphic --preset bass_boost
python3 main.py --synthetic --mode parametric --preset vocal_presence

# custom graphic gains (10 comma-separated dB values, matching
# TEN_BAND_CENTERS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
python3 main.py --synthetic --mode graphic --gains 6,5,3,1,0,0,0,0,0,0

# process your own file once you have one
python3 main.py --input audio/song.wav --mode graphic --preset vocal_boost
```

Each run writes to `output/`:
- `output_<mode>_<preset>.wav` — the processed audio
- `eq_response.png` — the filter's frequency-response curve
- `waveform_compare.png` — before/after waveform
- `spectrogram_compare.png` — before/after spectrogram, shared color scale

## Running the GUI (Step 3 + 5)

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, sounddevice
python3 gui/app.py
```

(On Linux you may also need `sudo apt install python3-tk` for Tkinter and
`sudo apt install libportaudio2` for `sounddevice` to find PortAudio.)

The app opens with a synthetic test signal already loaded. Switch between
the **Graphic EQ** and **Parametric EQ** tabs, drag sliders or pick a
preset, and the frequency-response plot updates live. Click **Play
Equalized** to hear it, or **Export Processed WAV...** to save it.
`Load WAV...` lets you use your own audio file instead, and the **Demo**
dropdown next to it switches between five built-in signals (Synthetic Test
Tone, Music Track, Two-Voice Demo, Three-Voice Demo, Noise Reduction Demo)
without cluttering the toolbar with one button per demo -- see Step 9 for
what each one is built to demonstrate.

This project was developed and screenshot-tested in a sandbox with no
audio hardware — the DSP, sliders, presets, and plots were all verified
there, but actual sound playback needs to be tried on your own machine.

The GUI also has a **"DSP Analysis" tab** (metrics table + before/after
spectrum plot) and a **post-processing row** above the tabs (50 Hz hum
removal, noise reduction) — these apply on top of whichever EQ mode is
active when you hit Play/Export.

A **"Now Playing"** panel sits above the tabs at all times, showing a
live scrolling waveform and a progress strip for whatever is currently
playing.

The **"Voice Separation"** tab lets you pull apart a recording of
multiple overlapping talkers (try **"Two-Voice Demo"** or **"Three-Voice
Demo"** from the Demo dropdown first, which separates automatically) into
2-3 approximate, individually-equalizable voices — see
Step 8 (and Step 9 for the demo signals themselves) below for how it works
and its honest limitations.

## Extra CLI flags (Step 5: analysis + hum/noise removal)

```bash
# remove 50Hz mains hum + its 2nd harmonic
python3 main.py --demo-music --mode graphic --preset flat --notch50

# apply a noise-reduction stage after EQ (none|lowpass|highpass|bandstop|spectral_gate|hum_removal)
python3 main.py --demo-music --mode graphic --preset bass_boost --noise-reduction spectral_gate

# combine both
python3 main.py --demo-music --mode parametric --preset warm --notch50 --noise-reduction lowpass
```

Every run now also prints a quantitative before/after metrics table
(RMS, peak, energy, spectral centroid, dominant frequency, zero-crossing
rate) and saves two extra report-ready images to `output/`:
- `spectrum_compare.png` — overlaid magnitude spectrum, original vs processed
- `analysis_dashboard.png` — waveform + spectrum + spectrogram + metrics
  table combined into one figure

## What's intentionally NOT included (and why)

Following the standard advice on scoping a course DSP project, a few
"nice to have" ideas were deliberately left out to keep the project
focused on signals-and-systems fundamentals rather than becoming a
software-engineering exercise:
- **Live microphone streaming** — a real feature, but a separate one
  (buffering/latency/threading), noted as a possible Step 4 if there's
  time; ask if you want it built.
- **ML/AI-based noise reduction, genre classification, cloud features** —
  explicitly out of scope for a Signals & Systems course; the spectral
  gating implementation here is intentionally classic DSP (STFT + gain
  mask), not learned. This is also why voice separation (Step 8) is
  built on pitch/harmonic clustering rather than a trained
  diarization/source-separation model -- same reasoning, same result:
  no downloaded weights, everything traceable back to Fourier analysis.
- **"Filter Playground" / standalone experiment-mode UI** — the underlying
  building blocks (phase, group delay, band energy, filter performance)
  are all implemented and scriptable; a dedicated experiment-sweep UI
  (Q sweep, gain sweep, etc.) wasn't built but would be straightforward
  to add as a small script using the existing functions.

## Filter theory analysis + PDF report (Step 6)

```bash
# phase response, group delay, band energy, and FIR-vs-IIR are computed
# and printed/plotted on every run automatically:
python3 main.py --demo-music --mode graphic --preset bass_boost

# check a specific band's measured performance against its design target:
python3 main.py --demo-music --mode parametric --preset vocal_presence \
    --target-freq 3000 --target-gain 4

# generate the full multi-page PDF report:
python3 main.py --demo-music --mode graphic --preset bass_boost --report
```

Every run now also saves:
- `phase_group_delay.png` — phase response + group delay, stacked
- `band_energy.png` — Bass/Low-mid/Mid/High-mid/Treble energy, before vs after
- `dsp_report.pdf` (only with `--report`) — the full combined report

In the GUI, the **DSP Analysis** tab has a **View** dropdown to switch
between Spectrum / Phase & Group Delay / Band Energy, and there's a
**"PDF Report..."** button in the toolbar that saves the same report
using whatever's currently loaded and however the sliders are set. Both
the CLI and GUI auto-detect a sensible filter-performance target (the
band with the largest |gain|) if you don't specify one.

## Controlled experiments (Step 7)

```bash
python3 main.py --experiments
```

Runs all 5 experiments and saves `output/experiments/experiments_report.pdf`
(10 pages: a text summary + plot per experiment) plus individual PNGs:

- **A. Bass boost sweep** — gain in {0, +3, +6, +9} dB, showing spectrum, band energy, RMS, and peak all increase together
- **B. Q factor sweep** — Q in {0.5, 1, 2, 5} at a fixed center frequency, showing the measured -3dB bandwidth narrows in an inverse relationship with Q
- **C. FIR vs IIR** — a graphic-EQ band and a parametric peak matched at the same center frequency/gain, comparing magnitude, phase, group delay, and measured processing time side by side
- **D. 50Hz hum removal** — before/after spectrum with measured attenuation at the fundamental and 2nd harmonic
- **E. Noise reduction** — before/after spectrum with measured noise-floor reduction

In the GUI, the **Experiments** tab has one button per experiment (runs
it and shows the plot inline) plus a "Save All (PDF)" button for the
combined report.

## Voice separation (Step 8)

```bash
python3 tests/verify_voice_separation.py
```

Builds the synthetic two-talker demo (a low-register ~105 Hz voice and a
high-register ~220 Hz voice, overlapping throughout, each colored with real
vowel formants -- see Step 9) and a three-talker demo (~95/165/250Hz),
separates each, and checks both (a) the detected pitch registers land close
to the true values and (b) each separated stream actually correlates with
the *matching* ground-truth voice rather than any other one -- i.e. it
checks the separation is *correct*, not just that something came out.

Separation uses two combined cues: pitch (clustered on its own, since it's
the strongest signal) and a frequency-amplitude fingerprint per voice
(learned from that voice's pitch-detected frames, then used to classify
whatever the pitch-based harmonic masks can't reach -- mainly unvoiced
consonants and silence, which have no pitch to track).

In the GUI: pick **"Two-Voice Demo"** or **"Three-Voice Demo"** from the
Demo dropdown (this separates automatically), or load your own multi-talker
audio, open the **Voice Separation** tab, choose 2 or 3
voices, and click **Separate Voices**. Each detected voice gets its own
5-band EQ, a Solo button, and a Mute toggle; **Play Mixed** / **Export Mixed
WAV...** recombine whichever voices aren't muted with their EQ applied.

## Noise reduction demo (Step 9)

```bash
python3 tests/verify_noise_demo.py
```

Builds the Noise Reduction Demo signal (a clean C-E-G-E-D-C melody, played
three times with silence between repeats, buried under broadband hiss and
50Hz+100Hz hum), processes it exactly as the GUI auto-configures it (hum
removal, then spectral gating), and checks the measured result against the
signal's own known-clean reference: the noise floor during silent gaps
drops by 19.5dB, the 50Hz hum drops by 36.7dB, and the processed melody
still correlates at 0.997 with the clean original -- i.e. the noise is
demonstrably gone and the real content is demonstrably intact, not just
"probably fine".

In the GUI: pick **"Noise Reduction Demo"** from the Demo dropdown -- it
loads the signal, ticks **Remove 50Hz hum**, sets **Noise reduction** to
`spectral_gate`, and jumps to the **DSP Analysis** tab so you can see the
before/after spectrum and metrics immediately. Click **Play Original** then
**Play Equalized** to hear the difference.
