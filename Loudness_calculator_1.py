import json
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mosqito.sq_metrics import loudness_zwst, sharpness_din_from_loudness
import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydub import AudioSegment
from scipy.signal import bilinear, find_peaks, lfilter, welch

try:
    import sottek_hearing_model as shm

    HAS_SHM = True
except ImportError:
    HAS_SHM = False

CONFIG_FILE = Path("config.json")
LEGACY_CALIBRATION_FILE = Path("calibration.json")
LEGACY_STANDARDS_FILE = Path("standards.json")

# =====================================================
# DIN45681 TONALITY
# =====================================================


def masking_index_av(freq):
    f_pts = np.array([
        90,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
    ])

    av_pts = np.array([
        8.0,
        7.5,
        7.0,
        6.0,
        5.0,
        4.0,
        3.5,
        3.0,
        2.5,
        2.0,
        1.5,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    return np.interp(freq, f_pts, av_pts, left=8.0, right=0.0)


def bark_bandwidth(freq):
    return 25 + 75 * (1 + 1.4 * (freq / 1000.0) ** 2) ** 0.69


def din45681_kt(delta_l):
    if delta_l <= 0:
        return 0
    elif delta_l <= 2:
        return 1
    elif delta_l <= 4:
        return 2
    elif delta_l <= 6:
        return 3
    elif delta_l <= 9:
        return 4
    elif delta_l <= 12:
        return 5
    else:
        return 6


def din45681_analysis(signal, fs, fmin=90):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(16384, len(signal)),
        scaling="spectrum",
    )

    psd_db = 10 * np.log10(psd + 1e-20)
    peaks, _ = find_peaks(psd_db, prominence=3.0)

    results = []

    for peak_idx in peaks:
        ft = frequencies[peak_idx]
        if ft < fmin:
            continue

        bw = bark_bandwidth(ft)
        band_idx = np.where(
            (frequencies >= ft - bw / 2) & (frequencies <= ft + bw / 2)
        )[0]

        if len(band_idx) < 5:
            continue

        Lt = psd_db[peak_idx]
        exclude = np.arange(
            max(0, peak_idx - 1), min(len(frequencies), peak_idx + 2)
        )
        noise_idx = np.setdiff1d(band_idx, exclude)

        if len(noise_idx) == 0:
            continue

        noise_energy = np.sum(10 ** (psd_db[noise_idx] / 10))
        Lg = 10 * np.log10(noise_energy + 1e-30)
        av = masking_index_av(ft)
        delta_l = Lt - Lg - av
        kt = din45681_kt(delta_l)

        results.append({
            "frequency_hz": float(ft),
            "delta_l": float(delta_l),
            "kt": float(kt),
            "Lt": float(Lt),
            "Lg": float(Lg),
            "av": float(av),
        })

    return sorted(results, key=lambda x: x["delta_l"], reverse=True)


def tonality_vs_time(signal, fs, window_sec=3.0, overlap=0.5):
    frame_size = int(window_sec * fs)
    hop_size = int(frame_size * (1 - overlap))

    times = []
    kt_values = []
    tone_freqs = []
    delta_values = []

    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start : start + frame_size]
        tones = din45681_analysis(frame, fs)
        center_time = (start + frame_size / 2) / fs
        times.append(center_time)

        if len(tones):
            strongest = tones[0]
            kt_values.append(strongest["kt"])
            tone_freqs.append(strongest["frequency_hz"])
            delta_values.append(strongest["delta_l"])
        else:
            kt_values.append(0)
            tone_freqs.append(np.nan)
            delta_values.append(0)

    return (
        np.array(times),
        np.array(kt_values),
        np.array(tone_freqs),
        np.array(delta_values),
    )


# =====================================================
# ECMA-418 / HMS TONALITY (tuHMS) & LOUDNESS (soneHMS)
# =====================================================


def _extract_shm_series(
    obj,
    value_keys=("tonality", "tonality_t", "T_t", "T", "values"),
    hop_sec=0.05,
):
    if isinstance(obj, dict):
        t_val = None
        for k in ("t", "time", "time_axis", "times"):
            if k in obj and obj[k] is not None:
                t_val = np.asarray(obj[k], dtype=np.float64).ravel()
                break

        y_val = None
        for k in value_keys:
            if k in obj and obj[k] is not None:
                y_val = np.asarray(obj[k], dtype=np.float64)
                break

        if y_val is None:
            for v in obj.values():
                arr = np.asarray(v, dtype=np.float64)
                if arr.ndim >= 1 and arr.size > 1:
                    y_val = arr
                    break

        if y_val is None:
            return np.array([0.0]), np.array([0.0])

        if y_val.ndim > 1:
            axis = 1 if (t_val is not None and y_val.shape[0] == len(t_val)) else 0
            y_1d = np.max(np.maximum(y_val, 0.0), axis=axis)
        else:
            y_1d = np.maximum(y_val.ravel(), 0.0)

        if t_val is None or len(t_val) != len(y_1d):
            t_val = np.arange(len(y_1d), dtype=np.float64) * hop_sec

        return t_val, y_1d

    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 0:
        return np.array([0.0]), np.array([max(0.0, float(arr))])
    if arr.ndim > 1:
        y_1d = np.max(np.maximum(arr, 0.0), axis=-1)
    else:
        y_1d = np.maximum(arr.ravel(), 0.0)
    t_val = np.arange(len(y_1d), dtype=np.float64) * hop_sec
    return t_val, y_1d


def ecma418_tonality_analysis(signal, fs):
    """
    Computes ECMA-418-2 / Sottek Hearing Model tonality (tuHMS).
    """
    signal = np.asarray(signal, dtype=np.float64)
    duration = len(signal) / fs if fs > 0 else 0.0

    if not HAS_SHM:
        return np.nan, np.linspace(0.0, duration, 1), np.array([np.nan])

    tonality_fn = None
    for attr_name in (
        "shm_tonality_ecma",
        "shm_tonality",
        "tonality_hms",
        "tonality",
    ):
        candidate = getattr(shm, attr_name, None)
        if callable(candidate):
            tonality_fn = candidate
            break

    if tonality_fn is None:
        for name in dir(shm):
            if "tonality" in name.lower():
                candidate = getattr(shm, name, None)
                if callable(candidate):
                    tonality_fn = candidate
                    break

    if tonality_fn is None:
        return np.nan, np.linspace(0.0, duration, 1), np.array([np.nan])

    try:
        res = tonality_fn(signal, fs)
    except TypeError:
        res = tonality_fn(signal, sample_rate=fs)

    overall_tuhms = np.nan
    times = np.array([0.0])
    tuhms_t = np.array([np.nan])

    if isinstance(res, dict):
        for k in ("tonality_avg", "T", "tonality", "tonality_overall", "overall"):
            if k in res and np.ndim(res[k]) == 0:
                overall_tuhms = float(res[k])
                break
        times, tuhms_t = _extract_shm_series(
            res,
            value_keys=("tonality_t", "tonality_t_avg", "tonality", "T_t", "T"),
        )
        if np.isnan(overall_tuhms) and len(tuhms_t):
            overall_tuhms = float(np.nanmax(tuhms_t))
    elif isinstance(res, tuple):
        if len(res) >= 1 and np.ndim(res[0]) == 0:
            overall_tuhms = float(res[0])
        if len(res) >= 2:
            times, tuhms_t = _extract_shm_series(res[1])
        elif len(res) >= 1 and np.ndim(res[0]) >= 1:
            times, tuhms_t = _extract_shm_series(res[0])
            overall_tuhms = float(np.nanmax(tuhms_t)) if len(tuhms_t) else np.nan
    else:
        times, tuhms_t = _extract_shm_series(res)
        overall_tuhms = float(np.nanmax(tuhms_t)) if len(tuhms_t) else np.nan

    if (
        np.isnan(overall_tuhms)
        and len(tuhms_t)
        and not np.all(np.isnan(tuhms_t))
    ):
        overall_tuhms = float(np.nanmax(tuhms_t))

    return overall_tuhms, times, tuhms_t


def ecma418_loudness_analysis(signal, fs):
    """
    Computes ECMA-418-2 / Sottek Hearing Model loudness (soneHMS).
    """
    signal = np.asarray(signal, dtype=np.float64)
    duration = len(signal) / fs if fs > 0 else 0.0
    # ECMA-418-2 uses 53 critical bands spanning 0.5 to 26.5 Bark (0.5 Bark steps)
    default_bark = np.arange(0.5, 27.0, 0.5)

    if not HAS_SHM:
        return (
            np.nan,
            np.linspace(0.0, duration, 1),
            np.array([np.nan]),
            default_bark,
            np.full_like(default_bark, np.nan),
        )

    loudness_fn = None
    for attr_name in (
        "shm_loudness_ecma",
        "shm_loudness",
        "loudness_hms",
        "loudness",
    ):
        candidate = getattr(shm, attr_name, None)
        if callable(candidate):
            loudness_fn = candidate
            break

    if loudness_fn is None:
        for name in dir(shm):
            if "loudness" in name.lower():
                candidate = getattr(shm, name, None)
                if callable(candidate):
                    loudness_fn = candidate
                    break

    if loudness_fn is None:
        return (
            np.nan,
            np.linspace(0.0, duration, 1),
            np.array([np.nan]),
            default_bark,
            np.full_like(default_bark, np.nan),
        )

    try:
        res = loudness_fn(signal, fs)
    except TypeError:
        res = loudness_fn(signal, sample_rate=fs)

    overall_N = np.nan
    times = np.array([0.0])
    loudness_t = np.array([np.nan])
    bark_axis = None
    specific_N = None
    loudness_keys = (
        "loudness_t",
        "loudness_t_avg",
        "loudness",
        "N_t",
        "N",
        "values",
    )

    if isinstance(res, dict):
        # 1. Overall loudness scalar
        for k in (
            "loudness_powavg",
            "loudness_avg",
            "loudness_N5",
            "loudness",
            "N",
            "overall",
        ):
            if k in res and res[k] is not None and np.ndim(res[k]) == 0:
                overall_N = float(res[k])
                break

        # 2. Time-varying loudness
        times, loudness_t = _extract_shm_series(res, value_keys=loudness_keys)

        # 3. Critical band axis (Hz -> Bark if band_centre_freqs is returned)
        for k in ("bark", "bark_axis", "critical_band_rate", "z"):
            if k in res and res[k] is not None:
                bark_axis = np.asarray(res[k], dtype=np.float64).ravel()
                break

        if bark_axis is None and "band_centre_freqs" in res and res["band_centre_freqs"] is not None:
            freqs_hz = np.asarray(res["band_centre_freqs"], dtype=np.float64).ravel()
            # Convert centre frequencies (Hz) to Bark scale
            bark_axis = (
                13.0 * np.arctan(0.00076 * freqs_hz)
                + 3.5 * np.arctan((freqs_hz / 7500.0) ** 2)
            )

        # 4. Specific loudness (matches `spec_loudness` in mlotinga/sottek-hearing-model)
        for k in (
            "spec_loudness",
            "spec_loudness_avg",
            "specific_loudness",
            "N_specific",
            "N_prime",
            "specific",
        ):
            if k in res and res[k] is not None:
                arr = np.asarray(res[k], dtype=np.float64)
                if arr.ndim == 3:
                    # Shape: (time, bands, channels) -> average across time & channels
                    specific_N = np.nanmean(arr, axis=(0, 2))
                elif arr.ndim == 2:
                    # Average along the time axis (whichever axis matches len(times))
                    if len(times) > 1 and arr.shape[0] == len(times):
                        specific_N = np.nanmean(arr, axis=0)
                    elif len(times) > 1 and arr.shape[1] == len(times):
                        specific_N = np.nanmean(arr, axis=1)
                    else:
                        # Standard SHM has 53 bands
                        time_axis = 1 if arr.shape[0] == 53 else 0
                        specific_N = np.nanmean(arr, axis=time_axis)
                else:
                    specific_N = arr.ravel()
                break

        # Fallback: scan any remaining 2D array in dict that isn't time-only
        if specific_N is None:
            for k, v in res.items():
                if v is None:
                    continue
                arr = np.asarray(v, dtype=np.float64)
                if arr.ndim >= 2 and 53 in arr.shape:
                    time_axis = 1 if arr.shape[0] == 53 else 0
                    specific_N = np.nanmean(arr, axis=time_axis).ravel()
                    break

    elif isinstance(res, tuple):
        if len(res) >= 1 and np.ndim(res[0]) == 0:
            overall_N = float(res[0])
        if len(res) >= 2:
            times, loudness_t = _extract_shm_series(
                res[1], value_keys=loudness_keys
            )
        elif len(res) >= 1 and np.ndim(res[0]) >= 1:
            times, loudness_t = _extract_shm_series(
                res[0], value_keys=loudness_keys
            )

        if len(res) >= 3:
            arr2 = np.asarray(res[2], dtype=np.float64)
            specific_N = np.nanmean(arr2, axis=0) if arr2.ndim > 1 else arr2.ravel()
        if len(res) >= 4:
            bark_axis = np.asarray(res[3], dtype=np.float64).ravel()

    else:
        times, loudness_t = _extract_shm_series(res, value_keys=loudness_keys)

    if (
        np.isnan(overall_N)
        and len(loudness_t)
        and not np.all(np.isnan(loudness_t))
    ):
        overall_N = float(np.nanmean(loudness_t))

    if specific_N is None:
        bark_axis = default_bark
        specific_N = np.full_like(default_bark, np.nan)
    elif bark_axis is None or len(bark_axis) != len(specific_N):
        bark_axis = np.arange(1, len(specific_N) + 1, dtype=np.float64) * 0.5

    return overall_N, times, loudness_t, bark_axis, specific_N


# =====================================================
# A-WEIGHTING FILTER
# =====================================================


def a_weighting(fs):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    nums = [
        (2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)),
        0,
        0,
        0,
        0,
    ]

    dens = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
    )

    dens = np.polymul(
        np.polymul(dens, [1, 2 * np.pi * f3]),
        [1, 2 * np.pi * f2],
    )

    return bilinear(nums, dens, fs)


def calculate_dba(signal, fs):
    b, a = a_weighting(fs)
    weighted = lfilter(b, a, signal)
    rms = np.sqrt(np.mean(weighted**2))

    if rms <= 0:
        return 0

    return 20 * np.log10(rms / 20e-6)


def sone_to_phon(sone):
    if sone <= 0 or np.isnan(sone):
        return 0.0

    return 40 + 10 * np.log2(sone)


# =====================================================
# AUDIO LOADER
# =====================================================


def load_audio(filename):
    audio = AudioSegment.from_file(filename)
    signal = np.array(audio.get_array_of_samples(), dtype=np.float64)

    if audio.channels > 1:
        signal = signal.reshape((-1, audio.channels)).mean(axis=1)

    signal /= 2 ** (8 * audio.sample_width - 1)
    fs = audio.frame_rate

    return signal, fs


# =====================================================
# GUI
# =====================================================

LOUDNESS_ISO = "ISO 532-1 (Zwicker)"
LOUDNESS_ECMA = "ECMA-418 (HMS)"
LOUDNESS_OPTIONS = [LOUDNESS_ISO, LOUDNESS_ECMA]

TONALITY_DIN = "DIN 45681"
TONALITY_ECMA = "ECMA-418 (HMS)"
TONALITY_OPTIONS = [TONALITY_DIN, TONALITY_ECMA]


class LoudnessAnalyzer:

    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Analyzer")
        self.root.geometry("1000x880")

        self.filename = None
        self.calibration_factor = 1.0
        self.peak_frequency = 0.0

        # Loudness & Tonality state attributes
        self.loudness_mode = LOUDNESS_ISO
        self.tonality_mode = TONALITY_DIN
        self.tonality_axis = np.array([])
        self.tonality_time = np.array([])
        self.delta_l_time = np.array([])
        self.tone_frequency_time = np.array([])
        self.tonality_overall = np.nan

        self.load_config()

        # ================= TOP TITLE =================
        tk.Label(
            root,
            text="Loudness & Tonality Analyzer",
            font=("Arial", 16, "bold"),
        ).pack(pady=(15, 10))

        # ================= CONTROL PANEL FRAME (3 Columns) =================
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5, fill="x", padx=20)

        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)

        # --- LEFT COLUMN: Calibration, Loudness & Tonality Methods ---
        left_frame = tk.Frame(control_frame)
        left_frame.grid(row=0, column=0, sticky="w")

        tk.Button(
            left_frame,
            text="Calibrate From SPL Meter",
            width=22,
            command=self.calibrate_from_spl,
        ).pack(anchor="w", pady=(0, 5))

        self.cal_label = tk.Label(
            left_frame,
            text=f"Calibration Factor: {self.calibration_factor:.4f}",
        )
        self.cal_label.pack(anchor="w", pady=(0, 8))

        tk.Label(
            left_frame,
            text="Loudness Standard:",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w")

        self.loudness_method_var = tk.StringVar(value=self.loudness_mode)
        self.loudness_combo = ttk.Combobox(
            left_frame,
            textvariable=self.loudness_method_var,
            values=LOUDNESS_OPTIONS,
            state="readonly",
            width=22,
        )
        self.loudness_combo.pack(anchor="w", pady=(2, 6))
        self.loudness_combo.bind(
            "<<ComboboxSelected>>", self.on_standard_selected
        )

        tk.Label(
            left_frame,
            text="Tonality Standard:",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w")

        self.tonality_method_var = tk.StringVar(value=self.tonality_mode)
        self.tonality_combo = ttk.Combobox(
            left_frame,
            textvariable=self.tonality_method_var,
            values=TONALITY_OPTIONS,
            state="readonly",
            width=22,
        )
        self.tonality_combo.pack(anchor="w", pady=(2, 0))
        self.tonality_combo.bind(
            "<<ComboboxSelected>>", self.on_standard_selected
        )

        # --- CENTER COLUMN: File Selection, Run & Progress Bar ---
        center_frame = tk.Frame(control_frame)
        center_frame.grid(row=0, column=1)

        self.file_label = tk.Label(center_frame, text="No file selected", fg="blue")
        self.file_label.pack(pady=(0, 3))

        self.select_btn = tk.Button(
            center_frame,
            text="Select Audio File",
            width=22,
            command=self.select_file,
        )
        self.select_btn.pack(pady=(0, 3))

        self.run_btn = tk.Button(
            center_frame,
            text="Run Analysis",
            width=22,
            bg="#2ecc71",
            fg="white",
            command=self.start_analysis,
        )
        self.run_btn.pack()

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            center_frame,
            variable=self.progress_var,
            maximum=100.0,
            length=180,
            mode="determinate",
        )
        self.progress_bar.pack(pady=(8, 2))

        self.progress_label = tk.Label(
            center_frame,
            text="Idle (0%)",
            font=("Arial", 8),
            fg="#555555",
        )
        self.progress_label.pack()

        # --- RIGHT COLUMN: Checkboxes ---
        right_frame = tk.Frame(control_frame)
        right_frame.grid(row=0, column=2, sticky="e")

        self.save_xlsx_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            right_frame,
            text="Save XLSX Summary Report",
            variable=self.save_xlsx_var,
        ).pack(anchor="w", pady=1)

        self.save_csv_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            right_frame,
            text="Save Loudness CSV Profile",
            variable=self.save_csv_var,
        ).pack(anchor="w", pady=1)

        self.save_png_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            right_frame,
            text="Save Plot PNG Files (Cleanup if unchecked)",
            variable=self.save_png_var,
        ).pack(anchor="w", pady=1)

        # ================= RESULTS TEXT AREA =================
        self.result_text = tk.Text(root, height=28)
        self.result_text.pack(fill="both", expand=True, padx=20, pady=15)

    # =================================================
    def set_progress(self, value, status_text=None, running=None):
        clamped = max(0.0, min(100.0, float(value)))

        def _update():
            self.progress_var.set(clamped)
            if status_text is not None:
                self.progress_label.config(text=f"{status_text} ({clamped:.0f}%)")
            else:
                self.progress_label.config(text=f"{clamped:.0f}%")

            if running is not None:
                state = "disabled" if running else "normal"
                self.run_btn.config(state=state)
                self.select_btn.config(state=state)

        self.root.after(0, _update)

    def load_config(self):
        loaded_from_config = False
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "factor" in data and data["factor"] is not None:
                        self.calibration_factor = float(data["factor"])
                    saved_loudness = data.get("loudness_standard")
                    saved_tonality = data.get("tonality_standard")
                    if saved_loudness in LOUDNESS_OPTIONS:
                        self.loudness_mode = saved_loudness
                    if saved_tonality in TONALITY_OPTIONS:
                        self.tonality_mode = saved_tonality
                    loaded_from_config = True
        except Exception:
            pass

        if loaded_from_config:
            return

        # Fallback to legacy files if config.json does not exist yet
        try:
            if LEGACY_CALIBRATION_FILE.exists():
                with open(LEGACY_CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    cal_data = json.load(f)
                if isinstance(cal_data, dict) and "factor" in cal_data:
                    self.calibration_factor = float(cal_data["factor"])
        except Exception:
            pass

        try:
            if LEGACY_STANDARDS_FILE.exists():
                with open(LEGACY_STANDARDS_FILE, "r", encoding="utf-8") as f:
                    std_data = json.load(f)
                if isinstance(std_data, dict):
                    saved_loudness = std_data.get("loudness_standard")
                    saved_tonality = std_data.get("tonality_standard")
                    if saved_loudness in LOUDNESS_OPTIONS:
                        self.loudness_mode = saved_loudness
                    if saved_tonality in TONALITY_OPTIONS:
                        self.tonality_mode = saved_tonality
        except Exception:
            pass

    def save_config(
        self,
        factor=None,
        loudness_standard=None,
        tonality_standard=None,
    ):
        factor_val = (
            float(factor)
            if factor is not None
            else float(self.calibration_factor)
        )
        loudness_val = (
            loudness_standard
            if loudness_standard is not None
            else self.loudness_method_var.get()
        )
        tonality_val = (
            tonality_standard
            if tonality_standard is not None
            else self.tonality_method_var.get()
        )

        if loudness_val not in LOUDNESS_OPTIONS:
            loudness_val = LOUDNESS_ISO
        if tonality_val not in TONALITY_OPTIONS:
            tonality_val = TONALITY_DIN

        self.calibration_factor = factor_val
        self.loudness_mode = loudness_val
        self.tonality_mode = tonality_val

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "factor": factor_val,
                        "loudness_standard": loudness_val,
                        "tonality_standard": tonality_val,
                    },
                    f,
                    indent=4,
                )
        except Exception:
            pass

    def on_standard_selected(self, _event=None):
        self.save_config()

    def calibrate_from_spl(self):
        cal_file = filedialog.askopenfilename(
            title="Select Recording Used For SPL Measurement",
            filetypes=[(
                "Audio Files",
                "*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.mp4 *.aac",
            )],
        )

        if not cal_file:
            return

        spl = simpledialog.askfloat(
            "Reference SPL",
            "Enter measured SPL (dB):\n\nExample: 35.2",
        )

        if spl is None:
            return

        try:
            signal, fs = load_audio(cal_file)
            rms = np.sqrt(np.mean(signal**2))

            if rms <= 0:
                raise Exception("Invalid recording")

            pressure_pa = 20e-6 * 10 ** (spl / 20)
            factor = float(pressure_pa / rms)
            self.save_config(factor=factor)

            self.cal_label.config(
                text=f"Calibration Factor: {self.calibration_factor:.6f}"
            )

            result = (
                f"Measured SPL : {spl:.2f} dB\n"
                f"Recording RMS: {rms:.6f}\n"
                f"Pressure     : {pressure_pa:.8f} Pa\n\n"
                f"Calibration Factor:\n{self.calibration_factor:.6f}"
            )

            messagebox.showinfo("Calibration Complete", result)

        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))

    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[(
                "Audio Files",
                "*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.mp4 *.aac",
            )]
        )

        if self.filename:
            self.file_label.config(text=Path(self.filename).name)

    def start_analysis(self):
        if not self.filename:
            messagebox.showerror("Error", "Please select a file.")
            return

        self.save_config(
            loudness_standard=self.loudness_method_var.get(),
            tonality_standard=self.tonality_method_var.get(),
        )
        self.set_progress(0.0, "Starting analysis", running=True)
        threading.Thread(target=self.analyze_file, daemon=True).start()

    def analyze_file(self):
        try:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Loading file...\n\n")
            self.set_progress(5.0, "Loading audio")

            signal, fs = load_audio(self.filename)
            signal = signal * self.calibration_factor
            duration = len(signal) / fs
            self.set_progress(12.0, "Audio loaded")

            self.result_text.insert(tk.END, f"File: {Path(self.filename).name}\n")
            self.result_text.insert(tk.END, f"Sample Rate: {fs} Hz\n")
            self.result_text.insert(tk.END, f"Duration: {duration:.2f} sec\n")
            self.result_text.insert(
                tk.END, f"Loudness Standard: {self.loudness_mode}\n"
            )
            self.result_text.insert(
                tk.END, f"Tonality Standard: {self.tonality_mode}\n\n"
            )

            is_ecma_loudness = self.loudness_mode == LOUDNESS_ECMA
            sone_unit = "soneHMS" if is_ecma_loudness else "sones"

            # Always compute Zwicker loudness for DIN Sharpness compatibility
            self.set_progress(18.0, "Calculating loudness & sharpness")
            N_zw, N_spec_zw, bark_zw = loudness_zwst(signal, fs)
            sharpness = sharpness_din_from_loudness(N_zw, N_spec_zw)

            if is_ecma_loudness:
                self.result_text.insert(
                    tk.END, "Generating ECMA-418 (HMS) loudness profile...\n"
                )
                self.set_progress(25.0, "Computing ECMA-418 loudness")
                N, times_hms, sones_hms, bark_axis, N_specific = (
                    ecma418_loudness_analysis(signal, fs)
                )
                self.bark_axis = bark_axis
                self.N_specific = N_specific
                self.set_progress(40.0, "Plotting loudness profile")
                self.save_and_plot_profile(times_hms, sones_hms, sone_unit)
            else:
                N = float(N_zw)
                self.bark_axis = bark_zw
                self.N_specific = N_spec_zw
                self.compute_zwicker_profile(signal, fs, sone_unit)

            self.set_progress(46.0, "Calculating phon & SPL(A)")
            phon = sone_to_phon(float(N))
            dba = calculate_dba(signal, fs)

            self.set_progress(52.0, "Analyzing tonality")
            tonality_png = self.plot_tonality(signal, fs)

            self.set_progress(62.0, "Plotting specific loudness")
            self.plot_specific_loudness(
                self.bark_axis, self.N_specific, sone_unit
            )

            self.set_progress(70.0, "Plotting frequency spectrum")
            spectrum_png = self.plot_frequency_spectrum(signal, fs)

            if self.tonality_mode == TONALITY_ECMA:
                tonality_summary_line = (
                    f"Tonality (ECMA-418): {self.tonality_overall:.3f} tuHMS\n"
                    if not np.isnan(self.tonality_overall)
                    else "Tonality (ECMA-418): N/A (install sottek-hearing-model)\n"
                )
            else:
                max_kt = (
                    float(np.nanmax(self.tonality_time))
                    if len(self.tonality_time)
                    else 0.0
                )
                tonality_summary_line = (
                    f"Tonality (DIN45681): {max_kt:.1f} dB (max Kt)\n"
                )

            if is_ecma_loudness and np.isnan(N):
                loudness_summary_line = (
                    "Loudness      : N/A (install sottek-hearing-model)\n"
                )
            else:
                loudness_summary_line = (
                    f"Loudness      : {float(N):.2f} {sone_unit}\n"
                )

            self.result_text.insert(tk.END, "\n===== OVERALL RESULTS =====\n\n")
            self.result_text.insert(tk.END, loudness_summary_line)
            self.result_text.insert(tk.END, f"Phon Level    : {phon:.2f} phon\n")
            self.result_text.insert(tk.END, f"Sharpness     : {sharpness:.2f} acum\n")
            self.result_text.insert(tk.END, tonality_summary_line)
            self.result_text.insert(tk.END, f"SPL(A)        : {dba:.2f} dB(A)\n\n")
            self.result_text.insert(tk.END, "Analysis Complete.\n")

            profile_png = Path(self.filename).with_suffix(".loudness.png")
            specific_png = Path(self.filename).with_suffix(".specific_loudness.png")
            spectrum_png_path = Path(self.filename).with_suffix(
                ".frequency_spectrum.png"
            )
            tonality_png_path = Path(tonality_png)

            if self.save_xlsx_var.get():
                self.set_progress(78.0, "Saving XLSX summary")
                summary_file = Path(self.filename).with_suffix(".summary.xlsx")
                wb = Workbook()
                ws = wb.active
                ws.title = "Summary"

                loudness_header = (
                    "Loudness ECMA-418 (soneHMS)"
                    if is_ecma_loudness
                    else "Loudness ISO 532-1 (sone)"
                )
                loudness_val = (
                    round(float(N), 2) if not np.isnan(N) else "N/A"
                )

                if self.tonality_mode == TONALITY_ECMA:
                    tonality_header = "Tonality ECMA-418 (tuHMS)"
                    tonality_val = (
                        round(float(self.tonality_overall), 3)
                        if not np.isnan(self.tonality_overall)
                        else "N/A"
                    )
                else:
                    tonality_header = "Tonality DIN45681 Max Kt (dB)"
                    tonality_val = (
                        round(float(np.nanmax(self.tonality_time)), 1)
                        if len(self.tonality_time)
                        else 0.0
                    )

                ws["A1"] = "File"
                ws["B1"] = loudness_header
                ws["C1"] = "Phon"
                ws["D1"] = "Sharpness (Acum)"
                ws["E1"] = "SPL dB(A)"
                ws["F1"] = "Peak Frequency (Hz)"
                ws["G1"] = tonality_header

                ws["A2"] = Path(self.filename).name
                ws["B2"] = loudness_val
                ws["C2"] = round(float(phon), 2)
                ws["D2"] = round(float(sharpness), 2)
                ws["E2"] = round(float(dba), 2)
                ws["F2"] = round(float(self.peak_frequency), 1)
                ws["G2"] = tonality_val

                if profile_png.exists():
                    img1 = Image(str(profile_png))
                    img1.width = 800
                    img1.height = 400
                    ws.add_image(img1, "A5")

                if specific_png.exists():
                    img2 = Image(str(specific_png))
                    img2.width = 800
                    img2.height = 400
                    ws.add_image(img2, "A30")

                if spectrum_png_path.exists():
                    img3 = Image(str(spectrum_png_path))
                    img3.width = 800
                    img3.height = 400
                    ws.add_image(img3, "A55")

                if tonality_png_path.exists():
                    img4 = Image(str(tonality_png_path))
                    img4.width = 800
                    img4.height = 400
                    ws.add_image(img4, "A80")

                tonality_ws = wb.create_sheet("Tonality")
                if self.tonality_mode == TONALITY_ECMA:
                    tonality_ws.append(["Time (s)", "Tonality (tuHMS)"])
                    for t, val in zip(self.tonality_axis, self.tonality_time):
                        tonality_ws.append([
                            float(t),
                            float(val) if not np.isnan(val) else None,
                        ])
                else:
                    tonality_ws.append([
                        "Time (s)",
                        "Kt (dB)",
                        "Delta L (dB)",
                        "Tone Frequency (Hz)",
                    ])
                    for t, k, d, f in zip(
                        self.tonality_axis,
                        self.tonality_time,
                        self.delta_l_time,
                        self.tone_frequency_time,
                    ):
                        tonality_ws.append([
                            float(t),
                            float(k),
                            float(d),
                            float(f) if not np.isnan(f) else None,
                        ])

                wb.save(summary_file)
                self.result_text.insert(
                    tk.END, f"Summary saved:\n{summary_file}\n\n"
                )

            self.set_progress(88.0, "Generating HTML report")
            html_file = Path(self.filename).with_suffix(".summary.html")
            self.save_html_report(
                html_file,
                self.filename,
                float(N),
                float(phon),
                float(sharpness),
                float(dba),
                float(self.peak_frequency),
                self.loudness_time,
                self.time_axis,
                self.bark_axis,
                self.N_specific,
                self.frequencies,
                self.psd_db,
            )

            if not self.save_png_var.get():
                self.set_progress(95.0, "Cleaning up PNG files")
                for p_file in [
                    profile_png,
                    specific_png,
                    spectrum_png_path,
                    tonality_png_path,
                ]:
                    if p_file.exists():
                        try:
                            p_file.unlink()
                        except Exception:
                            pass
                self.result_text.insert(
                    tk.END, "Intermediate PNG plots cleaned up.\n\n"
                )

            self.set_progress(100.0, "Complete", running=False)
            os.startfile(html_file)

        except Exception as e:
            self.set_progress(0.0, "Error", running=False)
            messagebox.showerror("Analysis Error", str(e))

    def compute_zwicker_profile(self, signal, fs, unit_label="sones"):
        window = int(0.5 * fs)
        hop = int(0.1 * fs)
        times = []
        sones = []

        self.result_text.insert(
            tk.END, "Generating ISO 532-1 (Zwicker) loudness profile...\n"
        )

        starts = list(range(0, max(0, len(signal) - window), hop))
        total_steps = max(1, len(starts))

        for idx, start in enumerate(starts):
            try:
                frame = signal[start : start + window]
                result = loudness_zwst(frame, fs)
                loudness = float(result[0])
                times.append(start / fs)
                sones.append(loudness)
            except Exception:
                pass

            if idx % 10 == 0 or idx == total_steps - 1:
                pct = 20.0 + (idx + 1) / total_steps * 22.0
                self.set_progress(pct, "Computing Zwicker profile")

        self.save_and_plot_profile(times, sones, unit_label)

    def save_and_plot_profile(self, times, sones, unit_label="sones"):
        sones_arr = np.asarray(sones, dtype=np.float64)
        valid_sones = (
            sones_arr[~np.isnan(sones_arr)] if len(sones_arr) else np.array([])
        )

        if len(valid_sones):
            avg_sone = float(np.mean(valid_sones))
            min_sone = float(np.min(valid_sones))
            max_sone = float(np.max(valid_sones))

            self.result_text.insert(
                tk.END, f"Average Loudness : {avg_sone:.2f} {unit_label}\n"
            )
            self.result_text.insert(
                tk.END, f"Peak Loudness    : {max_sone:.2f} {unit_label}\n"
            )
            self.result_text.insert(
                tk.END, f"Minimum Loudness : {min_sone:.2f} {unit_label}\n\n"
            )

        if self.save_csv_var.get():
            col_name = (
                "Loudness_soneHMS"
                if self.loudness_mode == LOUDNESS_ECMA
                else "Loudness_Sone"
            )
            profile_df = pd.DataFrame({"Time_s": times, col_name: sones})
            csv_file = Path(self.filename).with_suffix(".loudness.csv")
            profile_df.to_csv(csv_file, index=False)
            self.result_text.insert(tk.END, f"Profile saved:\n{csv_file}\n\n")

        self.time_axis = times
        self.loudness_time = sones

        plt.figure(figsize=(10, 5))
        if len(valid_sones):
            plt.plot(times, sones, color="blue")
        else:
            plt.text(
                0.5,
                0.5,
                "sottek-hearing-model not installed\n(pip install sottek-hearing-model)",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
        plt.title(f"Loudness vs Time ({self.loudness_mode})")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Loudness ({unit_label})")
        plt.grid(True)
        plt.tight_layout()

        profile_png = Path(self.filename).with_suffix(".loudness.png")
        plt.savefig(profile_png, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_specific_loudness(
        self, bark_axis, specific_loudness, unit_label="sone"
    ):
        spec_arr = np.asarray(specific_loudness, dtype=np.float64)
        plt.figure(figsize=(10, 5))
        if len(spec_arr) and not np.all(np.isnan(spec_arr)):
            plt.plot(bark_axis, specific_loudness, color="red", linewidth=2)
        else:
            plt.text(
                0.5,
                0.5,
                "Specific loudness unavailable\n(install sottek-hearing-model)",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
        plt.title(f"Specific Loudness Distribution ({self.loudness_mode})")
        plt.xlabel("Critical Band Rate (Bark)")
        plt.ylabel(f"Specific Loudness ({unit_label}/Bark)")
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.grid(True)
        plt.tight_layout()

        specific_png = Path(self.filename).with_suffix(".specific_loudness.png")
        plt.savefig(specific_png, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_frequency_spectrum(self, signal, fs):
        self.result_text.insert(tk.END, "Generating frequency spectrum...\n")
        signal = signal - np.mean(signal)

        frequencies, psd = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=min(8192, len(signal)),
            noverlap=None,
            scaling="spectrum",
        )

        valid = frequencies >= 20
        frequencies = frequencies[valid]
        psd = psd[valid]
        psd_db = 10 * np.log10(psd + 1e-20)

        peak_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_idx]
        self.peak_frequency = float(peak_frequency)
        self.frequencies = frequencies
        self.psd_db = psd_db

        peaks, _ = find_peaks(psd_db, prominence=3.0)

        plt.figure(figsize=(10, 5))
        plt.semilogx(frequencies, psd_db, color="green", linewidth=1.5)
        plt.scatter(frequencies[peaks], psd_db[peaks], color="red", s=12)
        plt.title("Frequency Spectrum (Welch PSD)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB)")
        plt.grid(True, which="both")
        plt.tight_layout()

        spectrum_png = Path(self.filename).with_suffix(".frequency_spectrum.png")
        plt.savefig(spectrum_png, dpi=300, bbox_inches="tight")
        plt.close()

        self.result_text.insert(tk.END, f"Peak Frequency : {peak_frequency:.1f} Hz\n")
        return spectrum_png

    def plot_tonality(self, signal, fs):
        tonality_png = Path(self.filename).with_suffix(".tonality.png")

        if self.tonality_mode == TONALITY_ECMA:
            self.result_text.insert(
                tk.END, "Generating ECMA-418 (HMS) tonality profile...\n"
            )
            overall, times, tuhms_t = ecma418_tonality_analysis(signal, fs)
            self.tonality_overall = overall
            self.tonality_axis = times
            self.tonality_time = tuhms_t
            self.delta_l_time = np.array([])
            self.tone_frequency_time = np.array([])

            plt.figure(figsize=(10, 5))
            if len(tuhms_t) and not np.all(np.isnan(tuhms_t)):
                plt.plot(times, tuhms_t, color="purple", linewidth=2)
            else:
                plt.text(
                    0.5,
                    0.5,
                    "sottek-hearing-model not installed\n(pip install sottek-hearing-model)",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
            plt.xlabel("Time (s)")
            plt.ylabel("Tonality (tuHMS)")
            plt.title("ECMA-418 (HMS) Tonality vs Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tonality_png, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            self.result_text.insert(
                tk.END, "Generating DIN45681 tonality profile...\n"
            )
            times, kt, tone_freq, delta_l = tonality_vs_time(signal, fs)
            self.tonality_overall = (
                float(np.nanmax(kt)) if len(kt) else np.nan
            )
            self.tonality_time = kt
            self.tonality_axis = times
            self.tone_frequency_time = tone_freq
            self.delta_l_time = delta_l

            plt.figure(figsize=(10, 5))
            plt.plot(times, kt, color="red", linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel("Kt (dB)")
            plt.title("DIN45681 Tonality vs Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tonality_png, dpi=300, bbox_inches="tight")
            plt.close()

        return tonality_png

    def save_html_report(
        self,
        output_html,
        filename,
        loudness,
        phon,
        sharpness,
        dba,
        peak_frequency,
        loudness_time,
        time_axis,
        bark_axis,
        specific_loudness,
        frequencies,
        psd_db,
    ):
        is_ecma_loudness = self.loudness_mode == LOUDNESS_ECMA
        loudness_unit = "soneHMS" if is_ecma_loudness else "sone"
        loudness_display = (
            "N/A" if np.isnan(loudness) else f"{loudness:.2f} {loudness_unit}"
        )
        loudness_subplot_title = f"Loudness vs Time ({self.loudness_mode})"
        specific_subplot_title = f"Specific Loudness ({self.loudness_mode})"

        is_ecma = self.tonality_mode == TONALITY_ECMA
        tonality_subplot_title = (
            "ECMA-418 (HMS) Tonality vs Time"
            if is_ecma
            else "DIN45681 Tonality vs Time"
        )
        tonality_y_label = "Tonality (tuHMS)" if is_ecma else "Kt (dB)"

        if is_ecma:
            tonality_badge = (
                f"ECMA-418 Tonality={self.tonality_overall:.3f} tuHMS"
                if not np.isnan(self.tonality_overall)
                else "ECMA-418 Tonality=N/A"
            )
            summary_row_html = f"""
            <tr><td><strong>Tonality (ECMA-418)</strong></td><td>{
                'N/A'
                if np.isnan(self.tonality_overall)
                else f'{self.tonality_overall:.3f} tuHMS'
            }</td></tr>
            """
            tonality_table = ""
        else:
            max_kt = (
                float(np.nanmax(self.tonality_time))
                if len(self.tonality_time)
                else 0.0
            )
            tonality_badge = f"DIN45681 Max Kt={max_kt:.1f} dB"
            summary_row_html = f"""
            <tr><td><strong>Tonality (DIN45681 Max Kt)</strong></td><td>{max_kt:.1f} dB</td></tr>
            """
            tonality_table = """
            <h3>DIN45681 Tonality Results</h3>
            <table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #2c3e50; color: white;">
                <th>Time (s)</th>
                <th>Kt (dB)</th>
                <th>ΔL (dB)</th>
                <th>Tone Frequency (Hz)</th>
            </tr>
            """
            for t, k, d, f in zip(
                self.tonality_axis,
                self.tonality_time,
                self.delta_l_time,
                self.tone_frequency_time,
            ):
                tonality_table += f"""
                <tr>
                    <td>{t:.2f}</td>
                    <td>{k:.1f}</td>
                    <td>{d:.2f}</td>
                    <td>{'' if np.isnan(f) else f'{f:.1f}'}</td>
                </tr>
                """
            tonality_table += "</table><br>"

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                loudness_subplot_title,
                tonality_subplot_title,
                "Frequency Spectrum (Welch PSD)",
                specific_subplot_title,
            ),
            vertical_spacing=0.10,
        )

        fig.add_trace(
            go.Scatter(x=time_axis, y=loudness_time, mode="lines", name="Loudness"),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text=f"Loudness ({loudness_unit})", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=self.tonality_axis,
                y=self.tonality_time,
                mode="lines",
                name="Tonality",
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text=tonality_y_label, row=2, col=1)

        fig.add_trace(
            go.Scatter(x=frequencies, y=psd_db, mode="lines", name="Spectrum"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bark_axis,
                y=specific_loudness,
                mode="lines",
                name="Specific Loudness",
            ),
            row=4,
            col=1,
        )

        fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=3, col=1)
        fig.update_yaxes(title_text="Level (dB)", row=3, col=1)
        fig.update_xaxes(title_text="Bark", dtick=1, row=4, col=1)
        fig.update_yaxes(title_text=f"{loudness_unit}/Bark", row=4, col=1)

        fig.update_layout(
            height=1800,
            title=(
                f"{Path(filename).name}<br>"
                f"Loudness={loudness_display} | "
                f"Phon={phon:.1f} | "
                f"Sharpness={sharpness:.2f} acum | "
                f"{tonality_badge} | "
                f"SPL(A)={dba:.2f} dB(A) | "
                f"Peak={peak_frequency:.1f} Hz"
            ),
        )

        reference_guide_html = r"""
        <hr style="margin: 30px 0;">
        <h2>Acoustic Metrics: Definitions, Reference Baselines & Calibration</h2>
        <p>This report evaluates audio recordings using standardized psychoacoustic and environmental metrics. Below are their precise definitions, calibration baselines, and real-world references:</p>

        <table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">
            <tr style="background-color: #2c3e50; color: white;">
                <th>Metric & Symbol</th>
                <th>Definition & Calibration Baseline</th>
                <th>Reference Values & Real-Life Scale</th>
            </tr>
            <tr>
                <td><strong>ISO 532-1 Loudness (sone)</strong></td>
                <td>
                    <strong>Definition:</strong> Linear psychoacoustic scale of perceived volume where doubling the sone value doubles the perceived loudness.<br>
                    <strong>Calibration/Baseline:</strong> Calculated via ISO 532-1 (Zwicker method) scaled against a calibration factor derived from reference SPL meter measurements.
                </td>
                <td>
                    <ul>
                        <li><strong>1.0 sone (Reference):</strong> 1 kHz tone at 40 dB SPL (40 phons) / Quiet refrigerator hum</li>
                        <li><strong>0.1 – 0.3 sone:</strong> Rustling leaves / quiet studio</li>
                        <li><strong>4.0 sone:</strong> Normal conversation at 1 meter</li>
                        <li><strong>16.0+ sone:</strong> City traffic / vacuum cleaner</li>
                    </ul>
                </td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td><strong>ECMA-418 Loudness (sone<sub>HMS</sub>)</strong></td>
                <td>
                    <strong>Definition:</strong> Psychoacoustic loudness computed via the Sottek Hearing Model (ECMA-418-2), separating tonal and broadband noise components before nonlinear combination.<br>
                    <strong>Calibration/Baseline:</strong> Aligned with ISO 226:2023 equal-loudness contours (1.0 sone<sub>HMS</sub> corresponds to a 1 kHz tone at 40 dB SPL).
                </td>
                <td>
                    <ul>
                        <li><strong>1.0 sone<sub>HMS</sub> (Reference):</strong> 1 kHz tone at 40 dB SPL</li>
                        <li><strong>Advantage:</strong> Improved correlation with human jury ratings on tonal machinery (EV motors, fans, inverters)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td><strong>Loudness Level (phon)</strong></td>
                <td>
                    <strong>Definition:</strong> Logarithmic measure of loudness level mapped to human equal-loudness contours.<br>
                    <strong>Calibration/Baseline:</strong> Derived mathematically from sone using 40 + 10 log<sub>2</sub>(sone).
                </td>
                <td>
                    <ul>
                        <li><strong>0 phon:</strong> Threshold of human hearing</li>
                        <li><strong>40 phon (Reference):</strong> Matches 1 sone baseline</li>
                        <li><strong>60 phon:</strong> Normal face-to-face conversation</li>
                        <li><strong>80 phon:</strong> Highway vehicle cabin noise / hairdryer</li>
                    </ul>
                </td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td><strong>DIN 45681 Tonality (<i>K</i><sub>t</sub> in dB)</strong></td>
                <td>
                    <strong>Definition:</strong> Evaluates prominent discrete frequency components (tones/hums) against surrounding masking noise.<br>
                    <strong>Calibration/Baseline:</strong> Uses critical band analysis (Bark scale) and DIN 45681 masking indices (<i>a</i><sub>v</sub>) to assign a dB penalty (0 dB to +6 dB).
                </td>
                <td>
                    <ul>
                        <li><strong>0 dB:</strong> Broadband noise (ocean waves, wind, smooth HVAC)</li>
                        <li><strong>1 – 2 dB:</strong> Faint mechanical gear whine or PC fan hum</li>
                        <li><strong>3 – 4 dB:</strong> Outdoor power transformer or heat pump hum</li>
                        <li><strong>6 dB:</strong> Maximum penalty for severe tonal inverter/brake noise</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td><strong>ECMA-418 / HMS Tonality (tuHMS)</strong></td>
                <td>
                    <strong>Definition:</strong> Psychoacoustic tonality based on the Sottek Hearing Model (ECMA-418-2), separating tonal and noise components.<br>
                    <strong>Calibration/Baseline:</strong> Calibrated to reference tonal stimuli (1.0 tuHMS corresponds to a 1 kHz tone at 40 dB SPL).
                </td>
                <td>
                    <ul>
                        <li><strong>&lt; 0.05 tuHMS:</strong> Broadband / non-tonal sound</li>
                        <li><strong>~0.2 – 0.5 tuHMS:</strong> Mild tonal content</li>
                        <li><strong>~1.0+ tuHMS:</strong> Prominent tonal perception</li>
                    </ul>
                </td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td><strong>Sharpness (acum)</strong></td>
                <td>
                    <strong>Definition:</strong> Measures high-frequency content and spectral brightness relative to critical band rate (Bark).<br>
                    <strong>Calibration/Baseline:</strong> Calculated using DIN standard formulas from specific loudness distributions.
                </td>
                <td>
                    <ul>
                        <li><strong>0.5 – 0.8 acum:</strong> Luxury sedan cabin rumble / muffled bass</li>
                        <li><strong>1.0 acum (Reference):</strong> Narrow-band noise centered at 1 kHz (160 Hz width) at 60 dB / steady rainfall</li>
                        <li><strong>1.8 – 2.5 acum:</strong> Standard household vacuum cleaner or inkjet printer</li>
                        <li><strong>3.5+ acum:</strong> Chalk on a blackboard / metallic squeal</li>
                    </ul>
                </td>
            </tr>
        </table>
        <br>
        """

        html = f"""
        <html>
        <head>
            <title>Acoustic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                h2 {{ color: #2c3e50; }}
                table {{ margin-bottom: 20px; }}
                th, td {{ text-align: left; vertical-align: top; }}
            </style>
        </head>
        <body>
        <h2>Acoustic Analysis Report</h2>
        <table border="1" cellpadding="6" style="border-collapse: collapse;">
            <tr><td><strong>File</strong></td><td>{Path(filename).name}</td></tr>
            <tr><td><strong>Loudness Standard</strong></td><td>{self.loudness_mode}</td></tr>
            <tr><td><strong>Tonality Standard</strong></td><td>{self.tonality_mode}</td></tr>
            <tr><td><strong>Loudness</strong></td><td>{loudness_display}</td></tr>
            <tr><td><strong>Phon</strong></td><td>{phon:.1f}</td></tr>
            <tr><td><strong>Sharpness</strong></td><td>{sharpness:.2f} acum</td></tr>
            {summary_row_html}
            <tr><td><strong>SPL(A)</strong></td><td>{dba:.2f} dB(A)</td></tr>
            <tr><td><strong>Peak Frequency</strong></td><td>{peak_frequency:.1f} Hz</td></tr>
        </table>
        <br>
        {tonality_table}
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        {reference_guide_html}
        </body>
        </html>
        """

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = LoudnessAnalyzer(root)
    root.mainloop()