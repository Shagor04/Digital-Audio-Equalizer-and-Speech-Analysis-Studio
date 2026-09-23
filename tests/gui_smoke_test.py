"""
tests/gui_smoke_test.py
========================
Headless GUI test (run under Xvfb): exercises both EQ tabs, all three
Analysis views, post-processing toggles, and PDF report generation
without crashing. All paths are portable (derived from this file's
location), and takes screenshots into tests/ and output/ for visual
review -- but the pass/fail signal is "did it crash", asserted at the
end for CI-style use.
"""
import os, sys, time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from gui.app import EqualizerApp

root = tk.Tk()
app = EqualizerApp(root)
root.update()

# Exercise graphic tab: apply a preset via the slider variables directly
app.graphic_preset_var.set("bass_boost")
app.on_graphic_preset()
root.update()
root.update_idletasks()
root.geometry("+0+0")
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_graphic_tab.png")}"')

# switch to parametric tab
app.notebook.select(1)
app.parametric_preset_var.set("vocal_presence")
app.on_parametric_preset()
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_parametric_tab.png")}"')

# switch to Analysis tab
app.notebook.select(2)
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_analysis_tab.png")}"')

# Band Energy view
app.analysis_view_var.set("Band Energy")
app.update_analysis()
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_analysis_band_energy.png")}"')

# Phase & Group Delay view
app.analysis_view_var.set("Phase & Group Delay")
app.update_analysis()
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_analysis_phase.png")}"')

# back to spectrum view for postfx screenshot
app.analysis_view_var.set("Spectrum")

# test 50Hz hum removal + noise reduction toggles
app.hum_var.set(True)
app.noise_reduction_var.set("spectral_gate")
app._on_post_fx_change()
app.update_analysis()
root.update()
time.sleep(0.3)
root.update()
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_analysis_tab_postfx.png")}"')

# switch to Voice Separation tab, load the two-voice demo, and separate it
app.notebook.select(3)
root.update()
app._load_two_voice_demo()
app.n_voices_var.set(2)
app.on_separate_voices()
root.update()
time.sleep(0.3)
root.update()
assert app.voice_streams is not None and len(app.voice_streams) == 2, "voice separation did not produce 2 streams"
os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_voices_tab.png")}"')
print("Voice separation summary:", app.voices_summary_var.get()[:120], "...")

# exercise a per-voice slider and the mixed-output helper (no audio hardware needed)
app.voice_widgets[0]["gains"][0].set(6.0)
app.on_voice_slider(0, 0)
mixed = app._mixed_voices_output()
assert mixed.shape == app.x.shape, "mixed voice output shape mismatch"
print("Mixed voice output OK, shape:", mixed.shape)

# reload the regular demo music track before the Experiments tab (which
# expects a normal single-track signal) and continue the smoke test
app._load_demo_music()
root.update()

# switch to Experiments tab and run each experiment
app.notebook.select(4)
root.update()
time.sleep(0.2)
root.update()

for key in ["bass_boost", "q_factor", "fir_vs_iir", "hum", "noise"]:
    app.run_experiment(key)
    root.update()
    time.sleep(0.2)
    root.update()
    print(f"Experiment '{key}' summary: {app.experiment_summary_var.get()[:80]}...")

os.system(f'import -window root "{os.path.join(TEST_DIR, "gui_experiments_tab.png")}"')

exp_pdf_path = os.path.join(OUTPUT_DIR, "gui_experiments_test_dir")
from core.run_experiments import run_all
run_all(exp_pdf_path)
print("Experiments PDF batch generation OK")

# test PDF report generation (headless, bypass the save dialog directly)
from core.report import generate_pdf_report
app._process_current()
report_path = os.path.join(OUTPUT_DIR, "gui_report_test.pdf")
generate_pdf_report(app.x, app.y, app.fs, mode=app._active_mode(), eq_info="smoke test",
                     graphic_eq=app.graphic_eq, parametric_eq=app._current_parametric_eq(),
                     out_path=report_path)
assert os.path.exists(report_path), "PDF report was not created"
print("PDF report generated OK")

# Test processing pipeline (non-audio parts) works without crashing
app._process_current()
assert app.y is not None, "processing did not produce output"
assert app.y.shape == app.x.shape, "output shape does not match input shape"
print("y shape:", app.y.shape)
print("dirty:", app.dirty)
print("SOUNDDEVICE_OK:", __import__("gui.app", fromlist=["SOUNDDEVICE_OK"]).SOUNDDEVICE_OK)

root.destroy()
print("GUI smoke test completed without crashing.")
