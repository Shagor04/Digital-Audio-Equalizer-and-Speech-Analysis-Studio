"""
gui/app.py
==========
Tkinter GUI for the CSE220 Audio Equalizer -- dark, modern theme.

Run locally (needs a display, speakers, and PortAudio for playback):
    python3 gui/app.py

Note: this sandbox environment has no audio hardware, so playback
buttons will report an error here -- run it on your own machine to
actually hear it. Every other feature (loading/generating signals,
moving sliders, presets, live frequency-response plot, export) works
regardless of audio hardware.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_io import (load_wav, save_wav, make_synthetic_test_signal, make_demo_music_signal,
                            make_two_voice_test_signal, make_three_voice_test_signal,
                            make_noise_demo_signal, check_and_normalize)
from core.eq_engine import GraphicEQ, ParametricEQ, TEN_BAND_CENTERS, FIVE_BAND_CENTERS
from core.analysis import summarize, compute_band_energy
from core.noise_reduction import NOISE_REDUCTION_METHODS, remove_hum
from core.filter_analysis import compute_phase_response, compute_group_delay
from core.report import generate_pdf_report
from core.voice_separation import separate_voices
from presets.presets import GRAPHIC_PRESETS, PARAMETRIC_PRESETS

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
    SD_ERROR = None
except Exception as e:  # pragma: no cover - depends on host machine
    SOUNDDEVICE_OK = False
    SD_ERROR = str(e)


# label -> internal key, single source of truth for the toolbar's Demo
# dropdown and its dispatcher (on_load_demo)
DEMO_OPTIONS = [
    ("\U0001F3B9 Synthetic Test Tone", "synthetic"),
    ("\U0001F3B5 Music Track", "music"),
    ("\U0001F5E3 Two-Voice Demo", "two_voice"),
    ("\U0001F465 Three-Voice Demo", "three_voice"),
    ("\U0001F507 Noise Reduction Demo", "noise"),
]

PARAM_BAND_DEFAULTS = [
    {"kind": "lowshelf", "f0": 120.0, "gain_db": 0.0, "Q": 0.7, "enabled": True},
    {"kind": "peak", "f0": 500.0, "gain_db": 0.0, "Q": 1.0, "enabled": True},
    {"kind": "peak", "f0": 3000.0, "gain_db": 0.0, "Q": 1.0, "enabled": True},
    {"kind": "highshelf", "f0": 8000.0, "gain_db": 0.0, "Q": 0.7, "enabled": True},
]

# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------
BG = "#12141c"          # window background
PANEL = "#1a1d29"        # toolbar / notebook background
CARD = "#20232f"         # card / slider-column background
CARD_BORDER = "#2e3244"
TEXT = "#e7e9f0"
SUBTEXT = "#8b90a3"
ACCENT = "#8b7cf6"        # violet -- primary accent (tabs, headers)
ACCENT_2 = "#22d3c9"      # teal -- secondary accent
PLAY_COLOR = "#2fbf71"    # green -- play buttons
STOP_COLOR = "#ef5a6f"    # red -- stop button
PLOT_BG = "#1a1d29"
PLOT_GRID = "#343850"
PLOT_LINE = "#8b7cf6"

# gradient endpoints for band-slider accent colors: bass -> mid -> treble
BASS_COLOR = (251, 146, 60)    # orange
MID_COLOR = (34, 211, 201)     # teal
TREBLE_COLOR = (192, 132, 252)  # violet

# live-waveform panel colors
WAVE_PLAYED = "#8b7cf6"     # accent -- portion already played
WAVE_UNPLAYED = "#343850"   # dim -- portion not played yet
WAVE_SCOPE = "#22d3c9"      # scrolling oscilloscope trace
WAVE_PLAYHEAD = "#ef5a6f"

# one accent color per separated voice (cycles if more voices than colors)
VOICE_COLORS = ["#22d3c9", "#fb923c", "#c084fc", "#60a5fa"]


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _band_color(index, total):
    """Return a hex color that sweeps bass(orange) -> mid(teal) -> treble(violet)
    across the band index, purely for a nicer, more informative slider row."""
    t = index / max(total - 1, 1)
    if t < 0.5:
        c = _lerp_color(BASS_COLOR, MID_COLOR, t / 0.5)
    else:
        c = _lerp_color(MID_COLOR, TREBLE_COLOR, (t - 0.5) / 0.5)
    return "#{:02x}{:02x}{:02x}".format(*c)


def freq_label(hz: float) -> str:
    return f"{hz/1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"


class EqualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSE220 Digital Audio Equalizer")
        self.root.geometry("1080x1010")
        self.root.configure(bg=BG)
        self.root.minsize(940, 900)

        self.fs = 44100
        self.x = None      # original signal currently loaded
        self.y = None      # cached processed signal
        self.dirty = True  # True => self.y needs recomputation before playing/exporting

        self.graphic_eq = GraphicEQ(self.fs)
        self.current_eq_mode = "graphic"

        # -- live waveform playback state --
        self._playing = False
        self._play_buffer = None
        self._play_len = 0
        self._play_start_time = 0.0
        self.overview_env = None   # cached min/max envelope of self.x, for the overview strip

        # -- voice separation state --
        self.voice_streams = None   # list of raw separated waveforms
        self.voice_info = None      # diagnostics dict from separate_voices()
        self.voice_eqs = []         # one GraphicEQ (5-band) per separated voice
        self.voice_mute_vars = []   # BooleanVar per voice
        self.voice_widgets = []     # slider/label widgets per voice, for reading current gains

        self._init_style()
        self._build_layout()
        self._load_synthetic()  # start with something loaded so the app is usable immediately

    # ------------------------------------------------------------------ styling
    def _init_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=PANEL, foreground=TEXT,
                         fieldbackground=CARD, font=("Segoe UI", 10))

        style.configure("TFrame", background=PANEL)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Toolbar.TFrame", background=PANEL)
        style.configure("Header.TFrame", background=BG)

        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("ToolbarMuted.TLabel", background=PANEL, foreground=SUBTEXT, font=("Segoe UI", 9, "italic"))
        style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Sub.TLabel", background=CARD, foreground=SUBTEXT, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=SUBTEXT, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=BG, foreground=SUBTEXT, font=("Segoe UI", 9))
        style.configure("BandVal.TLabel", background=CARD, foreground=ACCENT_2, font=("Segoe UI", 10, "bold"))

        style.configure("TButton", background=CARD, foreground=TEXT, borderwidth=0,
                         focusthickness=0, padding=(10, 7), font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", CARD_BORDER)])

        style.configure("Play.TButton", background=PLAY_COLOR, foreground="#08210f",
                         borderwidth=0, padding=(12, 7), font=("Segoe UI", 9, "bold"))
        style.map("Play.TButton", background=[("active", "#28a862")])

        style.configure("Stop.TButton", background=STOP_COLOR, foreground="#2a0508",
                         borderwidth=0, padding=(12, 7), font=("Segoe UI", 9, "bold"))
        style.map("Stop.TButton", background=[("active", "#d94d60")])

        style.configure("TNotebook", background=PANEL, borderwidth=0, tabmargins=(4, 6, 4, 0))
        style.configure("TNotebook.Tab", background=CARD, foreground=SUBTEXT,
                         padding=(18, 8), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#0d0a1f")])

        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT,
                         arrowcolor=TEXT, borderwidth=0)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)])

        style.configure("TCheckbutton", background=CARD, foreground=TEXT, font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", CARD)])

        style.configure("TLabelframe", background=CARD, bordercolor=CARD_BORDER, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=CARD, foreground=ACCENT_2, font=("Segoe UI", 9, "bold"))

        style.configure("TSeparator", background=CARD_BORDER)

        style.configure("Vertical.TScale", background=CARD, troughcolor=BG, borderwidth=0)
        style.configure("Horizontal.TScale", background=CARD, troughcolor=BG, borderwidth=0)

    # ------------------------------------------------------------------ layout
    def _build_layout(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 14, 18, 6))
        header.pack(side="top", fill="x")
        ttk.Label(header, text="\U0001F39A  CSE220 Digital Audio Equalizer",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="FIR filter-bank graphic EQ  \u2022  IIR biquad parametric EQ  \u2014  Signals & Systems project",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        top = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(14, 8))
        top.pack(side="top", fill="x")

        ttk.Button(top, text="\U0001F4C2 Load WAV...", command=self.on_load_wav).pack(side="left", padx=3)
        ttk.Label(top, text="Demo:", style="TLabel").pack(side="left", padx=(10, 4))
        self.demo_var = tk.StringVar(value=DEMO_OPTIONS[0][0])
        demo_combo = ttk.Combobox(
            top, textvariable=self.demo_var, state="readonly", width=22,
            values=[label for label, _ in DEMO_OPTIONS])
        demo_combo.pack(side="left", padx=3)
        demo_combo.bind("<<ComboboxSelected>>", self.on_load_demo)

        self.loaded_info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.loaded_info_var, style="ToolbarMuted.TLabel").pack(
            side="left", padx=(10, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)

        ttk.Button(top, text="\u25b6 Play Original", style="Play.TButton",
                   command=self.on_play_original).pack(side="left", padx=3)
        ttk.Button(top, text="\u25b6 Play Equalized", style="Play.TButton",
                   command=self.on_play_equalized).pack(side="left", padx=3)
        ttk.Button(top, text="\u25a0 Stop", style="Stop.TButton", command=self.on_stop).pack(side="left", padx=3)

        # --- post-processing row: 50Hz hum notch + noise reduction, applied after EQ ---
        tools = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(14, 0, 14, 8))
        tools.pack(side="top", fill="x")
        ttk.Label(tools, text="Post-processing:", style="TLabel").pack(side="left", padx=(0, 8))

        self.hum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tools, text="Remove 50Hz hum (+2nd harmonic)", variable=self.hum_var,
                         command=self._on_post_fx_change).pack(side="left", padx=(0, 14))

        ttk.Label(tools, text="Noise reduction:", style="TLabel").pack(side="left", padx=(0, 6))
        self.noise_reduction_var = tk.StringVar(value="none")
        nr_combo = ttk.Combobox(tools, textvariable=self.noise_reduction_var,
                                 values=list(NOISE_REDUCTION_METHODS.keys()),
                                 state="readonly", width=15)
        nr_combo.pack(side="left")
        nr_combo.bind("<<ComboboxSelected>>", lambda e: self._on_post_fx_change())

        ttk.Separator(tools, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)
        ttk.Button(tools, text="\U0001F4BE Export WAV...", command=self.on_export).pack(side="left", padx=3)
        ttk.Button(tools, text="\U0001F4C4 PDF Report...", command=self.on_generate_report).pack(side="left", padx=3)

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 4, 18, 8))
        status_bar.pack(side="bottom", fill="x")
        ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")

        self._build_now_playing_panel(self.root)

        body = ttk.Frame(self.root, style="Toolbar.TFrame")
        body.pack(side="top", fill="both", expand=True)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="top", fill="x", padx=14, pady=(6, 4))

        self.graphic_tab = ttk.Frame(self.notebook, style="Toolbar.TFrame")
        self.parametric_tab = ttk.Frame(self.notebook, style="Toolbar.TFrame")
        self.analysis_tab = ttk.Frame(self.notebook, style="Toolbar.TFrame")
        self.voices_tab = ttk.Frame(self.notebook, style="Toolbar.TFrame")
        self.experiments_tab = ttk.Frame(self.notebook, style="Toolbar.TFrame")
        self.notebook.add(self.graphic_tab, text="  Graphic EQ  ")
        self.notebook.add(self.parametric_tab, text="  Parametric EQ  ")
        self.notebook.add(self.analysis_tab, text="  DSP Analysis  ")
        self.notebook.add(self.voices_tab, text="  Voice Separation  ")
        self.notebook.add(self.experiments_tab, text="  Experiments  ")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self._build_graphic_tab()
        self._build_parametric_tab()
        self._build_analysis_tab()
        self._build_voices_tab()
        self._build_experiments_tab()
        self._build_plot_area(body)

    def on_tab_changed(self, event=None):
        # compare by widget identity rather than a hard-coded index, so
        # tab order can change later without silently breaking this
        selected = self.notebook.nametowidget(self.notebook.select())
        if selected is self.graphic_tab:
            self.current_eq_mode = "graphic"
            self.plot_area.pack(side="top", fill="both", expand=True)
            self.update_plot()
        elif selected is self.parametric_tab:
            self.current_eq_mode = "parametric"
            self.plot_area.pack(side="top", fill="both", expand=True)
            self.update_plot()
        elif selected is self.analysis_tab:
            self.plot_area.pack_forget()
            self.update_analysis()
        else:
            self.plot_area.pack_forget()

    # ------------------------------------------------------------------ live waveform panel
    def _build_now_playing_panel(self, parent):
        """
        A small 'now playing' card, visible on every tab, showing:
          - a scrolling oscilloscope-style trace of whatever is currently
            playing (updates ~30x/sec while Play is active)
          - a static overview strip of the whole track, with a moving
            playhead and an elapsed/total time readout
        The animation is driven purely by elapsed wall-clock time since
        Play was pressed (position = elapsed_seconds * sample_rate), so
        it works the same whether or not this machine actually has audio
        hardware to hear the result on.
        """
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10))
        card.pack(side="top", fill="x", padx=18, pady=(0, 8))

        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(side="top", fill="x")
        ttk.Label(header, text="\U0001F4C8 Now Playing", style="Card.TLabel",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        self.wave_time_var = tk.StringVar(value="0:00 / 0:00")
        ttk.Label(header, textvariable=self.wave_time_var, style="Sub.TLabel").pack(side="right")

        self.wave_canvas = tk.Canvas(card, height=110, bg=CARD, highlightthickness=0)
        self.wave_canvas.pack(side="top", fill="x", pady=(6, 0))
        self.wave_canvas.bind("<Configure>", lambda e: self._redraw_waveform_panel())

        self._reset_waveform_display()

    def _compute_envelope_columns(self, sig, n_cols):
        """Downsample `sig` into n_cols (min, max) pairs -- the standard
        'peak envelope' trick audio editors use to draw a waveform without
        plotting every single sample."""
        n = len(sig)
        if n == 0:
            return np.zeros(n_cols), np.zeros(n_cols)
        chunk = max(1, n // n_cols)
        mins = np.zeros(n_cols)
        maxs = np.zeros(n_cols)
        for c in range(n_cols):
            seg = sig[c * chunk: (c + 1) * chunk]
            if len(seg) == 0:
                continue
            mins[c] = seg.min()
            maxs[c] = seg.max()
        return mins, maxs

    def _reset_waveform_display(self):
        """Recompute the cached overview envelope for whatever is loaded
        as self.x, and redraw the panel with the playhead at the start."""
        if self.x is not None:
            self.overview_env = self._compute_envelope_columns(self.x, 300)
            total = len(self.x) / self.fs
        else:
            self.overview_env = None
            total = 0.0
        self._playing = False
        self.wave_time_var.set(f"0:00 / {self._format_time(total)}")
        self._redraw_waveform_panel(playhead_frac=0.0, scope_window=None)

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _redraw_waveform_panel(self, playhead_frac=0.0, scope_window=None):
        c = self.wave_canvas
        c.delete("all")
        w = c.winfo_width() or 900
        h = c.winfo_height() or 110

        scope_h = int(h * 0.62)
        overview_y = scope_h + 8
        overview_h = h - overview_y - 4

        # --- top: scrolling oscilloscope of the current playback window ---
        mid_y = scope_h / 2
        c.create_line(0, mid_y, w, mid_y, fill=CARD_BORDER, width=1)
        if scope_window is not None and len(scope_window[0]) > 0:
            mins, maxs = scope_window
            n_cols = len(mins)
            col_w = w / n_cols
            for i in range(n_cols):
                x0 = i * col_w
                y_top = mid_y - maxs[i] * (scope_h / 2) * 0.95
                y_bot = mid_y - mins[i] * (scope_h / 2) * 0.95
                c.create_line(x0, y_top, x0, y_bot, fill=WAVE_SCOPE, width=max(1, col_w))

        # --- bottom: static overview strip + moving playhead ---
        if self.overview_env is not None:
            mins, maxs = self.overview_env
            n_cols = len(mins)
            col_w = w / n_cols
            played_cols = int(playhead_frac * n_cols)
            for i in range(n_cols):
                x0 = i * col_w
                y_top = overview_y + overview_h / 2 - maxs[i] * (overview_h / 2) * 0.95
                y_bot = overview_y + overview_h / 2 - mins[i] * (overview_h / 2) * 0.95
                color = WAVE_PLAYED if i < played_cols else WAVE_UNPLAYED
                c.create_line(x0, y_top, x0, y_bot, fill=color, width=max(1, col_w))
            px = playhead_frac * w
            c.create_line(px, overview_y - 4, px, overview_y + overview_h + 4,
                          fill=WAVE_PLAYHEAD, width=2)
        else:
            c.create_text(w / 2, overview_y + overview_h / 2, text="Load or generate a signal to see its waveform",
                           fill=SUBTEXT, font=("Segoe UI", 9))

    def _play_buffer_now(self, data, label):
        """Shared entry point for every 'play this audio' action (original,
        equalized, or a separated voice) -- starts real playback via
        sounddevice AND starts the live waveform animation together, so
        they can never drift out of sync with each other."""
        if data is None:
            return False
        if not SOUNDDEVICE_OK:
            messagebox.showerror("Audio unavailable", f"sounddevice/PortAudio not available:\n{SD_ERROR}")
            return False
        try:
            sd.stop()
            sd.play(data, self.fs)
        except Exception as e:
            messagebox.showerror("Playback error", str(e))
            return False

        self.status_var.set(label)
        self._play_buffer = data
        self._play_len = len(data)
        self._play_start_time = time.time()
        self._playing = True
        self._tick_waveform()
        return True

    def _tick_waveform(self):
        if not self._playing:
            return
        elapsed = time.time() - self._play_start_time
        pos = int(elapsed * self.fs)

        if pos >= self._play_len:
            self._playing = False
            self.status_var.set("Finished playing.")
            total = self._play_len / self.fs
            self.wave_time_var.set(f"{self._format_time(total)} / {self._format_time(total)}")
            self._redraw_waveform_panel(playhead_frac=1.0, scope_window=None)
            return

        window_samples = int(0.25 * self.fs)  # show the trailing ~250ms
        start = max(0, pos - window_samples)
        window = self._play_buffer[start:pos]
        scope = self._compute_envelope_columns(window, 220) if len(window) > 0 else (np.zeros(1), np.zeros(1))

        total = self._play_len / self.fs
        current = pos / self.fs
        self.wave_time_var.set(f"{self._format_time(current)} / {self._format_time(total)}")
        self._redraw_waveform_panel(playhead_frac=pos / self._play_len, scope_window=scope)

        self.root.after(33, self._tick_waveform)

    def _on_post_fx_change(self):
        self.dirty = True


    def _build_graphic_tab(self):
        frame = self.graphic_tab

        preset_row = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 6))
        preset_row.pack(side="top", fill="x")
        ttk.Label(preset_row, text="Preset").pack(side="left", padx=(0, 6))
        self.graphic_preset_var = tk.StringVar(value="flat")
        combo = ttk.Combobox(preset_row, textvariable=self.graphic_preset_var,
                              values=list(GRAPHIC_PRESETS.keys()), state="readonly", width=18)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", self.on_graphic_preset)
        ttk.Button(preset_row, text="Reset to Flat", command=self.on_graphic_reset).pack(side="left", padx=8)

        card = ttk.Frame(frame, style="Card.TFrame", padding=(16, 14))
        card.pack(side="top", fill="x", pady=(2, 0))

        self.graphic_sliders = []
        self.graphic_value_labels = []
        n = len(TEN_BAND_CENTERS)
        for i, c in enumerate(TEN_BAND_CENTERS):
            col = ttk.Frame(card, style="Card.TFrame")
            col.grid(row=0, column=i, padx=8)
            color = _band_color(i, n)

            var = tk.DoubleVar(value=0.0)
            val_lbl = ttk.Label(col, text="0.0", style="BandVal.TLabel")
            val_lbl.pack()
            val_lbl.configure(foreground=color)

            s = ttk.Scale(col, from_=12, to=-12, orient="vertical", length=185,
                          variable=var, command=lambda val, idx=i: self.on_graphic_slider(idx))
            s.pack(pady=2)
            tick = tk.Frame(col, bg=color, height=3, width=34)
            tick.pack(pady=(0, 4))
            ttk.Label(col, text=freq_label(c) + " Hz", style="Sub.TLabel").pack()
            self.graphic_sliders.append(var)
            self.graphic_value_labels.append(val_lbl)

    def _build_parametric_tab(self):
        frame = self.parametric_tab
        preset_row = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 6))
        preset_row.pack(side="top", fill="x")
        ttk.Label(preset_row, text="Preset").pack(side="left", padx=(0, 6))
        self.parametric_preset_var = tk.StringVar(value="flat")
        combo = ttk.Combobox(preset_row, textvariable=self.parametric_preset_var,
                              values=list(PARAMETRIC_PRESETS.keys()), state="readonly", width=18)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", self.on_parametric_preset)

        bands_frame = ttk.Frame(frame, style="Toolbar.TFrame", padding=(8, 6))
        bands_frame.pack(side="top", fill="x")

        band_colors = ["#fb923c", "#22d3c9", "#60a5fa", "#c084fc"]
        self.param_widgets = []
        for i, band in enumerate(PARAM_BAND_DEFAULTS):
            box = ttk.LabelFrame(bands_frame, text=f"\u25cf Band {i+1}", padding=10)
            box.grid(row=0, column=i, padx=8, sticky="n")

            enabled_var = tk.BooleanVar(value=band["enabled"])
            ttk.Checkbutton(box, text="Enabled", variable=enabled_var,
                             command=self.on_param_change).pack(anchor="w", pady=(0, 4))

            type_var = tk.StringVar(value=band["kind"])
            ttk.Combobox(box, textvariable=type_var, values=["peak", "lowshelf", "highshelf"],
                         state="readonly", width=11).pack(pady=3)

            freq_var = tk.DoubleVar(value=band["f0"])
            freq_lbl = ttk.Label(box, text=f"Freq: {freq_label(band['f0'])} Hz", style="Card.TLabel")
            freq_lbl.pack(pady=(8, 0))
            ttk.Scale(box, from_=20, to=20000, orient="horizontal", length=170, variable=freq_var,
                      command=lambda val, i=i: self.on_param_slider(i)).pack()

            gain_var = tk.DoubleVar(value=band["gain_db"])
            gain_lbl = ttk.Label(box, text=f"Gain: {band['gain_db']:.1f} dB", style="Card.TLabel")
            gain_lbl.pack(pady=(6, 0))
            ttk.Scale(box, from_=-12, to=12, orient="horizontal", length=170, variable=gain_var,
                      command=lambda val, i=i: self.on_param_slider(i)).pack()

            q_var = tk.DoubleVar(value=band["Q"])
            q_lbl = ttk.Label(box, text=f"Q: {band['Q']:.2f}", style="Card.TLabel")
            q_lbl.pack(pady=(6, 0))
            ttk.Scale(box, from_=0.3, to=5.0, orient="horizontal", length=170, variable=q_var,
                      command=lambda val, i=i: self.on_param_slider(i)).pack()

            type_var.trace_add("write", lambda *a, i=i: self.on_param_change())

            self.param_widgets.append({
                "enabled": enabled_var, "type": type_var, "freq": freq_var,
                "gain": gain_var, "Q": q_var,
                "freq_label": freq_lbl, "gain_label": gain_lbl, "q_label": q_lbl,
            })

    def _build_analysis_tab(self):
        frame = self.analysis_tab

        top_row = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 6))
        top_row.pack(side="top", fill="x")
        ttk.Button(top_row, text="\u21bb Refresh Analysis", command=self.update_analysis).pack(side="left")
        ttk.Label(top_row, text="View:").pack(side="left", padx=(14, 4))
        self.analysis_view_var = tk.StringVar(value="Spectrum")
        view_combo = ttk.Combobox(top_row, textvariable=self.analysis_view_var,
                                   values=["Spectrum", "Phase & Group Delay", "Band Energy"],
                                   state="readonly", width=20)
        view_combo.pack(side="left")
        view_combo.bind("<<ComboboxSelected>>", lambda e: self.update_analysis())

        content = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 4))
        content.pack(side="top", fill="both", expand=True)

        # metrics table (left)
        table_card = ttk.Frame(content, style="Card.TFrame", padding=12)
        table_card.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(table_card, text="Quantitative Metrics", style="Card.TLabel",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        headers = ["Metric", "Original", "Processed"]
        for c, h in enumerate(headers):
            ttk.Label(table_card, text=h, style="Card.TLabel",
                      font=("Segoe UI", 9, "bold")).grid(row=1, column=c, sticky="w", padx=6, pady=2)

        self.metric_rows = {}
        metric_names = ["Duration (s)", "RMS", "Peak", "Energy",
                         "Spectral Centroid (Hz)", "Dominant Freq (Hz)", "Zero-Crossing Rate"]
        for r, name in enumerate(metric_names, start=2):
            ttk.Label(table_card, text=name, style="Card.TLabel").grid(row=r, column=0, sticky="w", padx=6, pady=2)
            v_before = ttk.Label(table_card, text="--", style="Card.TLabel")
            v_before.grid(row=r, column=1, sticky="e", padx=6, pady=2)
            v_after = ttk.Label(table_card, text="--", style="BandVal.TLabel")
            v_after.grid(row=r, column=2, sticky="e", padx=6, pady=2)
            self.metric_rows[name] = (v_before, v_after)

        # spectrum comparison plot (right)
        plot_card = ttk.Frame(content, style="Toolbar.TFrame")
        plot_card.pack(side="left", fill="both", expand=True)
        self.analysis_fig = Figure(figsize=(6.4, 3.6), dpi=100, facecolor=PLOT_BG)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, master=plot_card)
        self.analysis_canvas.get_tk_widget().configure(bg=PLOT_BG, highlightthickness=0)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_analysis(self):
        if self.x is None:
            return
        if self.dirty:
            self._process_current()

        before = summarize(self.x, self.fs)
        after = summarize(self.y, self.fs)

        display_map = {
            "Duration (s)": ("duration_s", "{:.2f}"),
            "RMS": ("rms", "{:.4f}"),
            "Peak": ("peak", "{:.4f}"),
            "Energy": ("energy", "{:.1f}"),
            "Spectral Centroid (Hz)": ("spectral_centroid_hz", "{:.1f}"),
            "Dominant Freq (Hz)": ("dominant_freq_hz", "{:.1f}"),
            "Zero-Crossing Rate": ("zero_crossing_rate", "{:.4f}"),
        }
        for name, (key, fmt) in display_map.items():
            v_before, v_after = self.metric_rows[name]
            v_before.config(text=fmt.format(before[key]))
            v_after.config(text=fmt.format(after[key]))

        view = self.analysis_view_var.get()
        self.analysis_fig.clear()

        if view == "Spectrum":
            from core.analysis import compute_magnitude_spectrum
            freqs_x, mag_x = compute_magnitude_spectrum(self.x, self.fs)
            freqs_y, mag_y = compute_magnitude_spectrum(self.y, self.fs)

            ax = self.analysis_fig.add_subplot(111)
            ax.set_facecolor(PLOT_BG)
            ax.plot(freqs_x, mag_x, color="#60a5fa", linewidth=0.8, alpha=0.85, label="Original")
            ax.plot(freqs_y, mag_y, color="#ef5a6f", linewidth=0.8, alpha=0.85, label="Processed")
            ax.set_xscale("log")
            ax.set_xlim(20, self.fs / 2)
            ax.set_xlabel("Frequency (Hz)", color=SUBTEXT, fontsize=9)
            ax.set_ylabel("Magnitude (dB)", color=SUBTEXT, fontsize=9)
            ax.set_title("Before / After spectrum", color=TEXT, fontsize=11, fontweight="bold", loc="left")
            ax.grid(True, which="both", color=PLOT_GRID, alpha=0.6, linewidth=0.5)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=CARD_BORDER)
            for spine in ax.spines.values():
                spine.set_color(CARD_BORDER)

        elif view == "Phase & Group Delay":
            mode = self._active_mode()
            if mode == "graphic":
                b, a = self.graphic_eq.composite_kernel(), [1.0]
            else:
                b, a = self._current_parametric_eq().combined_coeffs()
            fp, phase = compute_phase_response(b, a, self.fs)
            fg, gd = compute_group_delay(b, a, self.fs)

            ax1 = self.analysis_fig.add_subplot(211)
            ax2 = self.analysis_fig.add_subplot(212, sharex=ax1)
            for ax in (ax1, ax2):
                ax.set_facecolor(PLOT_BG)
                ax.grid(True, color=PLOT_GRID, alpha=0.6, linewidth=0.5)
                ax.tick_params(colors=SUBTEXT, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(CARD_BORDER)
            ax1.plot(fp, phase, color="#8b7cf6", linewidth=1.5)
            ax1.set_ylabel("Phase (rad)", color=SUBTEXT, fontsize=9)
            ax1.set_title(f"{mode.title()} EQ \u2014 Phase response", color=TEXT, fontsize=11,
                          fontweight="bold", loc="left")
            ax2.plot(fg, gd, color="#22d3c9", linewidth=1.5)
            ax2.set_ylabel("Group delay (ms)", color=SUBTEXT, fontsize=9)
            ax2.set_xlabel("Frequency (Hz)", color=SUBTEXT, fontsize=9)
            ax2.set_title("Group delay", color=TEXT, fontsize=10, fontweight="bold", loc="left")

        elif view == "Band Energy":
            band_before = compute_band_energy(self.x, self.fs)
            band_after = compute_band_energy(self.y, self.fs)
            names = list(band_before.keys())
            yb = np.arange(len(names))
            h = 0.35

            ax = self.analysis_fig.add_subplot(111)
            ax.set_facecolor(PLOT_BG)
            ax.barh(yb + h / 2, [band_before[n] for n in names], height=h, label="Original", color="#60a5fa")
            ax.barh(yb - h / 2, [band_after[n] for n in names], height=h, label="Processed", color="#ef5a6f")
            ax.set_yticks(yb)
            ax.set_yticklabels(names, color=SUBTEXT, fontsize=9)
            ax.set_xlabel("Share of total spectral energy (%)", color=SUBTEXT, fontsize=9)
            ax.set_title("Frequency-band energy distribution", color=TEXT, fontsize=11,
                         fontweight="bold", loc="left")
            ax.grid(True, axis="x", color=PLOT_GRID, alpha=0.6, linewidth=0.5)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=CARD_BORDER)
            for spine in ax.spines.values():
                spine.set_color(CARD_BORDER)

        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

    # ------------------------------------------------------------------ Experiments tab
    def _build_experiments_tab(self):
        frame = self.experiments_tab

        top_row = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 6))
        top_row.pack(side="top", fill="x")
        ttk.Label(top_row, text="Run a controlled experiment:").pack(side="left", padx=(0, 8))
        for label, key in [("Bass Boost", "bass_boost"), ("Q Factor", "q_factor"),
                            ("FIR vs IIR", "fir_vs_iir"), ("50Hz Hum", "hum"),
                            ("Noise Reduction", "noise")]:
            ttk.Button(top_row, text=label, command=lambda k=key: self.run_experiment(k)).pack(side="left", padx=3)
        ttk.Button(top_row, text="\U0001F4C4 Save All (PDF)...",
                   command=self.on_save_experiments_pdf).pack(side="left", padx=(14, 3))

        self.experiment_summary_var = tk.StringVar(value="Choose an experiment above to run it.")
        ttk.Label(frame, textvariable=self.experiment_summary_var, wraplength=1000,
                  font=("Segoe UI", 9)).pack(side="top", fill="x", padx=8, pady=(2, 4))

        self.experiment_plot_frame = ttk.Frame(frame, style="Toolbar.TFrame")
        self.experiment_plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))
        self.experiment_canvas_widget = None

    def run_experiment(self, key):
        from core import experiments as exp_mod
        from visualization import plots as plot_mod

        self.status_var.set(f"Running experiment: {key}...")
        self.root.update_idletasks()
        try:
            if key == "bass_boost":
                data = exp_mod.experiment_bass_boost(x=self.x, fs=self.fs) if self.x is not None \
                    else exp_mod.experiment_bass_boost()
                fig = plot_mod.plot_experiment_bass_boost(data)
                parts = [f"{r['gain_db']:+.0f}dB: bass={r['band_energy']['Bass']:.1f}% RMS={r['rms']:.4f}"
                         for r in data["results"]]
                summary = "Bass Boost Sweep \u2014 " + "   ".join(parts)
            elif key == "q_factor":
                data = exp_mod.experiment_q_factor()
                fig = plot_mod.plot_experiment_q_factor(data)
                parts = [f"Q={r['Q']}: BW={r['bandwidth_hz']:.0f}Hz" for r in data["results"]]
                summary = "Q Factor Sweep \u2014 " + "   ".join(parts)
            elif key == "fir_vs_iir":
                data = exp_mod.experiment_fir_vs_iir(x=self.x, fs=self.fs) if self.x is not None \
                    else exp_mod.experiment_fir_vs_iir()
                fig = plot_mod.plot_experiment_fir_vs_iir(data)
                summary = (f"FIR vs IIR @ ~{data['f0']:.0f}Hz \u2014 FIR: {data['fir']['processing_time_ms']:.1f}ms "
                           f"({data['fir']['num_coeffs']} coeffs)   IIR: {data['iir']['processing_time_ms']:.1f}ms "
                           f"({data['iir']['num_coeffs']} coeffs)")
            elif key == "hum":
                data = exp_mod.experiment_hum_removal()
                fig = plot_mod.plot_experiment_hum_removal(data)
                summary = (f"50Hz Hum Removal \u2014 fundamental attenuation: "
                           f"{data['attenuation_fundamental_db']:.1f}dB, "
                           f"2nd harmonic: {data['attenuation_2nd_harmonic_db']:.1f}dB")
            elif key == "noise":
                data = exp_mod.experiment_noise_reduction()
                fig = plot_mod.plot_experiment_noise_reduction(data)
                summary = f"Noise Reduction \u2014 noise floor reduced {data['noise_reduction_db']:.1f}dB"
            else:
                return
        except Exception as e:
            messagebox.showerror("Experiment failed", str(e))
            self.status_var.set("Ready.")
            return

        self.experiment_summary_var.set(summary)
        self._show_experiment_figure(fig)
        self.status_var.set(f"Experiment '{key}' complete.")

    def _show_experiment_figure(self, fig):
        """Embed a freshly-generated matplotlib Figure (from core.run_experiments'
        plot functions, which build their own standalone figures) into the
        Experiments tab, restyling it to match the app's dark theme, and
        swapping out whichever figure was shown previously."""
        fig.patch.set_facecolor(PLOT_BG)
        for ax in fig.get_axes():
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=SUBTEXT, labelsize=8)
            ax.xaxis.label.set_color(SUBTEXT)
            ax.yaxis.label.set_color(SUBTEXT)
            ax.title.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_color(CARD_BORDER)
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(PANEL)
                legend.get_frame().set_edgecolor(CARD_BORDER)
                for text in legend.get_texts():
                    text.set_color(TEXT)

        if self.experiment_canvas_widget is not None:
            self.experiment_canvas_widget.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.experiment_plot_frame)
        canvas.get_tk_widget().configure(bg=PLOT_BG, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.experiment_canvas_widget = canvas

    def on_save_experiments_pdf(self):
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")],
                                             initialfile="experiments_report.pdf")
        if not path:
            return
        self.status_var.set("Running all 5 experiments (this takes a few seconds)...")
        self.root.update_idletasks()
        try:
            from core.run_experiments import run_all
            import shutil
            out_dir = os.path.dirname(path) or "."
            generated = run_all(out_dir)
            if os.path.abspath(generated) != os.path.abspath(path):
                shutil.move(generated, path)
            self.status_var.set(f"Saved experiments report to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Experiments report failed", str(e))
            self.status_var.set("Ready.")

    # ------------------------------------------------------------------ Voice Separation tab
    def _build_voices_tab(self):
        frame = self.voices_tab

        top_row = ttk.Frame(frame, style="Toolbar.TFrame", padding=(4, 6))
        top_row.pack(side="top", fill="x")
        ttk.Label(top_row, text="Voices to detect:").pack(side="left", padx=(0, 6))
        self.n_voices_var = tk.IntVar(value=2)
        ttk.Combobox(top_row, textvariable=self.n_voices_var, values=[2, 3], state="readonly",
                     width=4).pack(side="left")
        ttk.Button(top_row, text="\U0001F399 Separate Voices",
                   command=self.on_separate_voices).pack(side="left", padx=(10, 3))
        ttk.Button(top_row, text="\u25b6 Play Mixed", style="Play.TButton",
                   command=self.on_play_voices_mixed).pack(side="left", padx=3)
        ttk.Button(top_row, text="\U0001F4BE Export Mixed WAV...",
                   command=self.on_export_voices_mixed).pack(side="left", padx=3)

        self.voices_summary_var = tk.StringVar(value=self.VOICES_IDLE_TEXT)
        ttk.Label(frame, textvariable=self.voices_summary_var, wraplength=1000,
                  font=("Segoe UI", 9), foreground=SUBTEXT).pack(side="top", fill="x", padx=8, pady=(2, 6))

        self.voices_cards_frame = ttk.Frame(frame, style="Toolbar.TFrame")
        self.voices_cards_frame.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

    # single source of truth for the voice-tab's idle instructional text,
    # so it can be restored whenever separation results become stale
    VOICES_IDLE_TEXT = ("Pick \"Two-Voice Demo\" or \"Three-Voice Demo\" from the Demo dropdown above "
                         "(it will separate automatically), or load your own multi-talker audio and click "
                         "Separate Voices. This uses pitch/formant clustering (no AI models), so it works "
                         "best when the voices sit in noticeably different pitch ranges.")

    def _clear_voice_separation(self):
        """Forget any previously separated voices (called whenever a new
        signal is loaded, since old separated streams belong to the old
        signal) -- and reset the summary text too, so switching to a
        different demo never leaves a stale "Voice 1: ~99Hz..." result
        on screen describing audio that isn't loaded anymore."""
        self.voice_streams = None
        self.voice_info = None
        self.voice_eqs = []
        self.voice_mute_vars = []
        self.voice_widgets = []
        for child in self.voices_cards_frame.winfo_children():
            child.destroy()
        self.voices_summary_var.set(self.VOICES_IDLE_TEXT)

    def on_separate_voices(self):
        if self.x is None:
            return
        n_voices = self.n_voices_var.get()
        self.status_var.set(f"Separating {n_voices} voices... this can take a few seconds.")
        self.root.update_idletasks()
        try:
            voices, info = separate_voices(self.x, self.fs, n_voices=n_voices)
        except Exception as e:
            messagebox.showerror("Voice separation failed", str(e))
            self.status_var.set("Ready.")
            return

        self.voice_streams = voices
        self.voice_info = info
        self.voice_eqs = [GraphicEQ(self.fs, centers=FIVE_BAND_CENTERS) for _ in voices]
        self.voice_mute_vars = [tk.BooleanVar(value=False) for _ in voices]
        self._build_voice_cards()

        parts = [f"Voice {i+1}: ~{p:.0f}Hz, detected in {a*100:.0f}% of frames"
                 for i, (p, a) in enumerate(zip(info["pitch_hz"], info["active_fraction"]))]
        self.voices_summary_var.set(
            "Separated (approximate, harmonic-based -- see card notes below). " + "   |   ".join(parts))
        self.status_var.set(f"Separated {n_voices} voices.")

    def _build_voice_cards(self):
        for child in self.voices_cards_frame.winfo_children():
            child.destroy()
        self.voice_widgets = []

        for v, y_v in enumerate(self.voice_streams):
            color = VOICE_COLORS[v % len(VOICE_COLORS)]
            pitch = self.voice_info["pitch_hz"][v]
            box = ttk.LabelFrame(self.voices_cards_frame, text=f"\U0001F3A4 Voice {v + 1}  (~{pitch:.0f} Hz)",
                                  padding=10)
            box.grid(row=0, column=v, padx=8, sticky="n")

            btn_row = ttk.Frame(box, style="Card.TFrame")
            btn_row.pack(side="top", fill="x", pady=(0, 8))
            ttk.Button(btn_row, text="\u25b6 Solo", style="Play.TButton",
                       command=lambda v=v: self.on_play_voice_solo(v)).pack(side="left")
            mute_chk = ttk.Checkbutton(btn_row, text="Mute", variable=self.voice_mute_vars[v])
            mute_chk.pack(side="left", padx=(8, 0))

            sliders_row = ttk.Frame(box, style="Card.TFrame")
            sliders_row.pack(side="top")
            sliders = []
            labels = []
            for b, center in enumerate(FIVE_BAND_CENTERS):
                col = ttk.Frame(sliders_row, style="Card.TFrame")
                col.grid(row=0, column=b, padx=6)
                var = tk.DoubleVar(value=0.0)
                lbl = ttk.Label(col, text="0.0", style="BandVal.TLabel")
                lbl.pack()
                lbl.configure(foreground=color)
                s = ttk.Scale(col, from_=12, to=-12, orient="vertical", length=130, variable=var,
                              command=lambda val, v=v, b=b: self.on_voice_slider(v, b))
                s.pack(pady=2)
                tick = tk.Frame(col, bg=color, height=3, width=26)
                tick.pack(pady=(0, 4))
                ttk.Label(col, text=freq_label(center) + " Hz", style="Sub.TLabel").pack()
                sliders.append(var)
                labels.append(lbl)
            self.voice_widgets.append({"gains": sliders, "labels": labels})

    def on_voice_slider(self, v, b):
        val = self.voice_widgets[v]["gains"][b].get()
        self.voice_widgets[v]["labels"][b].config(text=f"{val:.1f}")

    def _current_voice_output(self, v):
        """Apply voice v's 5-band EQ to its separated stream and return
        the result -- computed on demand (short streams, cheap to redo)
        rather than cached, to keep the slider logic simple."""
        gains = [g.get() for g in self.voice_widgets[v]["gains"]]
        self.voice_eqs[v].set_gains(gains)
        return self.voice_eqs[v].process(self.voice_streams[v])

    def _mixed_voices_output(self):
        outputs = [self._current_voice_output(v) for v in range(len(self.voice_streams))
                   if not self.voice_mute_vars[v].get()]
        if not outputs:
            return np.zeros_like(self.voice_streams[0])
        mix = np.sum(outputs, axis=0)
        mix, _ = check_and_normalize(mix)
        return mix

    def on_play_voice_solo(self, v):
        if not self.voice_streams:
            return
        self._play_buffer_now(self._current_voice_output(v), f"Playing Voice {v + 1} solo...")

    def on_play_voices_mixed(self):
        if not self.voice_streams:
            messagebox.showinfo("No separated voices", "Click 'Separate Voices' first.")
            return
        self._play_buffer_now(self._mixed_voices_output(), "Playing separated mix (per-voice EQ applied)...")

    def on_export_voices_mixed(self):
        if not self.voice_streams:
            messagebox.showinfo("No separated voices", "Click 'Separate Voices' first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")],
                                             initialfile="voices_mixed.wav")
        if not path:
            return
        save_wav(path, self._mixed_voices_output(), self.fs)
        self.status_var.set(f"Exported separated/re-mixed voices to {os.path.basename(path)}")

    def _build_plot_area(self, parent):
        plot_frame = ttk.Frame(parent, style="Toolbar.TFrame", padding=(14, 6, 14, 12))
        plot_frame.pack(side="top", fill="both", expand=True)
        self.plot_area = plot_frame
        self.fig = Figure(figsize=(9, 3.4), dpi=100, facecolor=PLOT_BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().configure(bg=PLOT_BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot()

    # ------------------------------------------------------------------ signal loading
    def _load_synthetic(self):
        self.x, self.fs = make_synthetic_test_signal()
        self.dirty = True
        self._clear_voice_separation()
        self.demo_var.set(DEMO_OPTIONS[0][0])
        self.loaded_info_var.set(f"loaded: {len(self.x)/self.fs:.1f}s tone")
        self.status_var.set(f"Loaded synthetic test signal ({len(self.x)/self.fs:.1f}s @ {self.fs}Hz).")
        self.update_plot()
        self._reset_waveform_display()

    def _load_demo_music(self):
        self.x, self.fs = make_demo_music_signal()
        self.dirty = True
        self._clear_voice_separation()
        self.demo_var.set(DEMO_OPTIONS[1][0])
        self.loaded_info_var.set(f"loaded: {len(self.x)/self.fs:.1f}s music")
        self.status_var.set(f"Loaded demo music track ({len(self.x)/self.fs:.1f}s @ {self.fs}Hz) "
                             f"-- bass + melody + hi-hats, good for hearing EQ changes.")
        self.update_plot()
        self._reset_waveform_display()

    def on_load_demo(self, event=None):
        key = dict(DEMO_OPTIONS).get(self.demo_var.get())
        if key == "synthetic":
            self._load_synthetic()
        elif key == "music":
            self._load_demo_music()
        elif key == "two_voice":
            self._load_voice_demo(make_two_voice_test_signal, n_voices=2,
                                   label="two-voice demo (a lower-register and a higher-register talker, "
                                         "different vowel colors, different speaking rhythms)")
        elif key == "three_voice":
            self._load_voice_demo(make_three_voice_test_signal, n_voices=3,
                                   label="three-voice demo (low, mid, and high register talkers)")
        elif key == "noise":
            self._load_noise_demo()

    def _load_voice_demo(self, loader_fn, n_voices, label):
        self.x, self.fs = loader_fn()
        self.dirty = True
        self._clear_voice_separation()
        self.demo_var.set(DEMO_OPTIONS[2][0] if n_voices == 2 else DEMO_OPTIONS[3][0])
        self.loaded_info_var.set(f"loaded: {len(self.x)/self.fs:.1f}s, {n_voices} talkers")
        self.status_var.set(f"Loaded {label} ({len(self.x)/self.fs:.1f}s @ {self.fs}Hz). Separating voices...")
        self.update_plot()
        self._reset_waveform_display()
        self.root.update_idletasks()
        self.notebook.select(self.voices_tab)
        self.n_voices_var.set(n_voices)
        self.on_separate_voices()
        # give each voice a little EQ lift on its own card so Play Solo /
        # Play Mixed has an audible EQ effect too, not just the raw separation
        for v in range(len(self.voice_widgets)):
            self.voice_widgets[v]["gains"][0].set(3.0)   # a touch more low end
            self.on_voice_slider(v, 0)

    def _load_noise_demo(self):
        self.x, self.fs = make_noise_demo_signal()
        self.dirty = True
        self._clear_voice_separation()
        self.demo_var.set(DEMO_OPTIONS[4][0])
        self.loaded_info_var.set(f"loaded: {len(self.x)/self.fs:.1f}s, hum+hiss")
        self.hum_var.set(True)
        self.noise_reduction_var.set("spectral_gate")
        self.status_var.set(
            f"Loaded noise reduction demo ({len(self.x)/self.fs:.1f}s @ {self.fs}Hz) -- a clean melody "
            f"buried under hiss and 50Hz hum. Hum removal + spectral gating are now both turned ON -- "
            f"click Play Original, then Play Equalized, to hear the difference (and see it below).")
        self.update_plot()
        self._reset_waveform_display()
        # jump straight to the Before/After metrics + spectrum view, so the
        # noise reduction is immediately visible, not just audible -- the
        # same "show, don't just tell" treatment the voice demos get above
        self.analysis_view_var.set("Spectrum")
        self.notebook.select(self.analysis_tab)  # on_tab_changed() runs update_analysis() for us

    def on_load_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self.x, self.fs = load_wav(path)
        self.dirty = True
        self.graphic_eq = GraphicEQ(self.fs)  # rebuild filter bank at the new sample rate
        self._clear_voice_separation()
        self.demo_var.set("")  # not one of the built-in demos
        self.loaded_info_var.set(f"loaded: {os.path.basename(path)}")
        self.status_var.set(f"Loaded {os.path.basename(path)} ({len(self.x)/self.fs:.1f}s @ {self.fs}Hz).")
        self.update_plot()
        self._reset_waveform_display()

    # ------------------------------------------------------------------ graphic tab callbacks
    def on_graphic_slider(self, idx):
        val = self.graphic_sliders[idx].get()
        self.graphic_value_labels[idx].config(text=f"{val:.1f}")
        self.dirty = True
        self.update_plot()

    def on_graphic_preset(self, event=None):
        name = self.graphic_preset_var.get()
        gains = GRAPHIC_PRESETS[name]
        for var, g, lbl in zip(self.graphic_sliders, gains, self.graphic_value_labels):
            var.set(g)
            lbl.config(text=f"{g:.1f}")
        self.dirty = True
        self.update_plot()

    def on_graphic_reset(self):
        self.graphic_preset_var.set("flat")
        self.on_graphic_preset()

    def _current_graphic_gains(self):
        return [v.get() for v in self.graphic_sliders]

    # ------------------------------------------------------------------ parametric tab callbacks
    def on_param_slider(self, i):
        w = self.param_widgets[i]
        w["freq_label"].config(text=f"Freq: {freq_label(w['freq'].get())} Hz")
        w["gain_label"].config(text=f"Gain: {w['gain'].get():.1f} dB")
        w["q_label"].config(text=f"Q: {w['Q'].get():.2f}")
        self.dirty = True
        self.update_plot()

    def on_param_change(self):
        self.dirty = True
        self.update_plot()

    def on_parametric_preset(self, event=None):
        name = self.parametric_preset_var.get()
        bands = PARAMETRIC_PRESETS[name]
        for i, w in enumerate(self.param_widgets):
            if i < len(bands):
                b = bands[i]
                w["enabled"].set(True)
                w["type"].set(b["kind"])
                w["freq"].set(b["f0"])
                w["gain"].set(b["gain_db"])
                w["Q"].set(b["Q"])
            else:
                w["enabled"].set(False)
            self.on_param_slider(i)
        self.dirty = True
        self.update_plot()

    def _current_parametric_eq(self):
        eq = ParametricEQ(self.fs)
        for w in self.param_widgets:
            if w["enabled"].get():
                eq.add_band(f0=w["freq"].get(), gain_db=w["gain"].get(),
                            Q=w["Q"].get(), kind=w["type"].get())
        return eq

    # ------------------------------------------------------------------ processing
    def _active_mode(self):
        return self.current_eq_mode

    def _process_current(self):
        if self.x is None:
            return
        mode = self._active_mode()
        if mode == "graphic":
            self.graphic_eq.set_gains(self._current_graphic_gains())
            y = self.graphic_eq.process(self.x)
        else:
            eq = self._current_parametric_eq()
            y = eq.process(self.x)

        if self.hum_var.get():
            y = remove_hum(y, self.fs, f0=50.0, Q=30.0)

        nr = self.noise_reduction_var.get()
        if nr != "none":
            y = NOISE_REDUCTION_METHODS[nr](y, self.fs)

        y, clip_info = check_and_normalize(y)
        self.last_clip_info = clip_info

        self.y = y
        self.dirty = False

    # ------------------------------------------------------------------ plotting
    def update_plot(self):
        self.ax.clear()
        mode = self._active_mode()
        if mode == "graphic":
            self.graphic_eq.set_gains(self._current_graphic_gains())
            freqs, mag = self.graphic_eq.composite_response()
            title = "Graphic EQ \u2014 composite frequency response"
        else:
            eq = self._current_parametric_eq()
            if len(eq.bands) == 0:
                freqs = np.logspace(np.log10(20), np.log10(self.fs / 2), 512)
                mag = np.zeros_like(freqs)
            else:
                freqs, mag = eq.composite_response()
            title = "Parametric EQ \u2014 composite frequency response"

        self.ax.set_facecolor(PLOT_BG)
        self.ax.plot(freqs, mag, color=PLOT_LINE, linewidth=2.2)
        self.ax.fill_between(freqs, mag, 0, color=PLOT_LINE, alpha=0.15)
        self.ax.axhline(0, color=SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.6)
        self.ax.set_xscale("log")
        self.ax.set_xlim(20, self.fs / 2)
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlabel("Frequency (Hz)", color=SUBTEXT, fontsize=9)
        self.ax.set_ylabel("Gain (dB)", color=SUBTEXT, fontsize=9)
        self.ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", loc="left")
        self.ax.grid(True, which="both", color=PLOT_GRID, alpha=0.6, linewidth=0.6)
        self.ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(CARD_BORDER)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ playback / export
    def on_play_original(self):
        if self.x is None:
            return
        self._play_buffer_now(self.x, "Playing original...")

    def on_play_equalized(self):
        if self.x is None:
            return
        if self.dirty:
            self._process_current()
        self._play_buffer_now(self.y, "Playing equalized...")

    def on_stop(self):
        if SOUNDDEVICE_OK:
            sd.stop()
        self._playing = False
        self.status_var.set("Stopped.")
        self._redraw_waveform_panel(playhead_frac=0.0, scope_window=None)

    def on_export(self):
        if self.x is None:
            return
        self._process_current()
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        save_wav(path, self.y, self.fs)
        rms_in = np.sqrt(np.mean(self.x ** 2))
        rms_out = np.sqrt(np.mean(self.y ** 2))
        change_db = 20 * np.log10(rms_out / (rms_in + 1e-12))
        clip_note = ""
        if getattr(self, "last_clip_info", None) and self.last_clip_info["clipping_detected"]:
            clip_note = (f"  [clipping auto-normalized: peak {self.last_clip_info['peak_before']:.3f} "
                         f"-> {self.last_clip_info['peak_after']:.3f}]")
        self.status_var.set(f"Exported to {os.path.basename(path)}  (RMS change: {change_db:+.2f} dB){clip_note}")

    def _auto_detect_target(self, mode):
        """
        Pick a sensible (target_freq, target_gain) pair for the filter
        performance page automatically, from whichever band currently has
        the largest |gain| -- so the report's "measured vs target" check
        always matches a band that's actually configured, instead of
        requiring the user to separately type in matching values (which
        is easy to get out of sync with the sliders).
        """
        if mode == "graphic":
            gains = self._current_graphic_gains()
            if all(abs(g) < 1e-6 for g in gains):
                return None, None
            idx = int(np.argmax([abs(g) for g in gains]))
            return TEN_BAND_CENTERS[idx], gains[idx]
        else:
            eq = self._current_parametric_eq()
            if not eq.bands:
                return None, None
            band = max(eq.bands, key=lambda b: abs(b["gain_db"]))
            if abs(band["gain_db"]) < 1e-6:
                return None, None
            return band["f0"], band["gain_db"]

    def on_generate_report(self):
        if self.x is None:
            return
        self._process_current()
        mode = self._active_mode()
        if mode == "graphic":
            eq_desc = f"graphic gains={self._current_graphic_gains()}"
            graphic_obj = self.graphic_eq
            parametric_obj = self._current_parametric_eq()
            if len(parametric_obj.bands) == 0:
                parametric_obj = ParametricEQ(self.fs)
                parametric_obj.add_band(f0=1000, gain_db=0, Q=1.0, kind="peak")
        else:
            eq_desc = f"parametric bands={[b for b in self._current_parametric_eq().bands]}"
            parametric_obj = self._current_parametric_eq()
            graphic_obj = self.graphic_eq

        target_freq, target_gain = self._auto_detect_target(mode)

        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")],
                                             initialfile="dsp_report.pdf")
        if not path:
            return
        try:
            generate_pdf_report(self.x, self.y, self.fs, mode=mode, eq_info=eq_desc,
                                 graphic_eq=graphic_obj, parametric_eq=parametric_obj,
                                 target_freq=target_freq, target_gain=target_gain,
                                 out_path=path)
            self.status_var.set(f"Saved DSP analysis report to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Report generation failed", str(e))


def main():
    root = tk.Tk()
    app = EqualizerApp(root)
    if not SOUNDDEVICE_OK:
        app.status_var.set(f"Note: audio playback unavailable on this machine ({SD_ERROR}). "
                            f"Sliders, plots, and export still work.")
    root.mainloop()


if __name__ == "__main__":
    main()
