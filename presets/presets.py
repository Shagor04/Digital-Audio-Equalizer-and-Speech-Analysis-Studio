"""
presets/presets.py
===================
Named EQ curves. Graphic presets are gain lists matching
core.eq_engine.TEN_BAND_CENTERS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000].
Parametric presets are lists of band dicts consumable by ParametricEQ.add_band(**band).
"""

# --- Graphic EQ presets (10 bands, gains in dB) ---
GRAPHIC_PRESETS = {
    "flat":        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "bass_boost":  [7, 6, 4, 2, 0, 0, 0, 0, 0, 0],
    "treble_boost":[0, 0, 0, 0, 0, 0, 2, 4, 6, 7],
    "vocal_boost": [-2, -2, -1, 2, 4, 4, 3, 1, 0, -1],
    "v_shape":     [6, 5, 2, -1, -3, -3, -1, 2, 5, 6],   # "loudness" smiley curve
    "bass_cut":    [-6, -5, -3, -1, 0, 0, 0, 0, 0, 0],
    "treble_cut":  [0, 0, 0, 0, 0, 0, -2, -4, -6, -7],
    "podcast_voice": [-4, -3, -1, 1, 3, 3, 2, 0, -2, -4],
}

# --- Parametric EQ presets (list of band definitions) ---
PARAMETRIC_PRESETS = {
    "flat": [],
    "bass_boost": [
        {"kind": "lowshelf", "f0": 120, "gain_db": 6, "Q": 0.7},
    ],
    "vocal_presence": [
        {"kind": "peak", "f0": 250, "gain_db": -3, "Q": 1.0},   # remove "mud"
        {"kind": "peak", "f0": 3000, "gain_db": 4, "Q": 1.2},   # add presence/clarity
    ],
    "de_ess": [
        {"kind": "peak", "f0": 6500, "gain_db": -6, "Q": 3.0},  # tame harsh sibilance
    ],
    "telephone": [
        {"kind": "lowshelf", "f0": 400, "gain_db": -12, "Q": 0.7},
        {"kind": "highshelf", "f0": 3000, "gain_db": -12, "Q": 0.7},
    ],
    "warm": [
        {"kind": "lowshelf", "f0": 200, "gain_db": 4, "Q": 0.7},
        {"kind": "highshelf", "f0": 8000, "gain_db": -3, "Q": 0.7},
    ],
}


def list_presets():
    print("Graphic EQ presets:", ", ".join(GRAPHIC_PRESETS.keys()))
    print("Parametric EQ presets:", ", ".join(PARAMETRIC_PRESETS.keys()))
