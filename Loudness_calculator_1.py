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
# ECMA-418 / HMS TONALITY (tuHMS)
# =====================================================


def _extract_shm_series(obj, hop_sec=0.05):
    if isinstance(obj, dict):
        t_val = None
        for k in ("t", "time", "time_axis", "times"):
            if k in obj and obj[k] is not None:
                t_val = np.asarray(obj[k], dtype=np.float64).ravel()
                break

        y_val = None
        for k in ("tonality", "tonality_t", "T_t", "T", "values"):
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
    Returns:
        overall_tuhms (float)
        times (np.ndarray)
        tuhms_t (np.ndarray)
    """
    signal = np.asarray(signal, dtype=np.float64)
    duration = len(signal) / fs if fs > 0 else 0.0

    if not HAS_SHM:
        return np.nan, np.linspace(0.0, duration, 1), np.array([np.nan])

    tonality_fn = None
    for attr_name in (
        "shm_tonality",
        "tonality_hms",
        "tonality",
        "acoustic_shm_tonality",
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

    if isinstance(res, tuple):
        if len(res) >= 1 and np.ndim(res[0]) == 0:
            overall_tuhms = float(res[0])
        if len(res) >= 2:
            times, tuhms_t = _extract_shm_series(res[1])
        elif len(res) >= 1 and np.ndim(res[0]) >= 1:
            times, tuhms_t = _extract_shm_series(res[0])
            overall_tuhms = float(np.nanmax(tuhms_t)) if len(tuhms_t) else np.nan
    elif isinstance(res, dict):
        for k in ("T", "tonality", "tonality_overall", "overall"):
            if k in res and np.ndim(res[k]) == 0:
                overall_tuhms = float(res[k])
                break
        times, tuhms_t = _extract_shm_series(res)
        if np.isnan(overall_tuhms) and len(tuhms_t):
            overall_tuhms = float(np.nanmax(tuhms_t))
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
    if sone <= 0:
        return 0

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

TONALITY_DIN = "DIN 45681"
TONALITY_ECMA = "ECMA-418 (HMS)"


class LoudnessAnalyzer:

    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Analyzer")
        self.root.geometry("1000x850")

        self.filename = None
        self.calibration_factor = 1.0
        self.peak_frequency = 0.0

        # Tonality state attributes
        self.tonality_mode = TONALITY_DIN
        self.tonality_axis = np.array([])
        self.tonality_time = np.array([])
        self.delta_l_time = np.array([])
        self.tone_frequency_time = np.array([])
        self.tonality_overall = np.nan

        try:
            with open("calibration.json", "r") as f:
                self.calibration_factor = json.load(f)["factor"]
        except Exception:
            pass

        # ================= TOP TITLE =================
        tk.Label(
            root,
            text="ISO 532 Loudness & Tonality Analyzer",
            font=("Arial", 16, "bold"),
        ).pack(pady=(15, 10))

        # ================= CONTROL PANEL FRAME (3 Columns) =================
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5, fill="x", padx=20)

        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)

        # --- LEFT COLUMN: Calibration & Tonality Method ---
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
            text="Tonality Standard:",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w")

        self.tonality_method_var = tk.StringVar(value=TONALITY_DIN)
        self.tonality_combo = ttk.Combobox(
            left_frame,
            textvariable=self.tonality_method_var,
            values=[TONALITY_DIN, TONALITY_ECMA],
            state="readonly",
            width=20,
        )
        self.tonality_combo.pack(anchor="w", pady=(2, 0))

        # --- CENTER COLUMN: File Selection & Run ---
        center_frame = tk.Frame(control_frame)
        center_frame.grid(row=0, column=1)

        self.file_label = tk.Label(center_frame, text="No file selected", fg="blue")
        self.file_label.pack(pady=(0, 3))

        tk.Button(
            center_frame,
            text="Select Audio File",
            width=22,
            command=self.select_file,
        ).pack(pady=(0, 3))

        run_btn = tk.Button(
            center_frame,
            text="Run Analysis",
            width=22,
            bg="#2ecc71",
            fg="white",
            command=self.start_analysis,
        )
        run_btn.pack()

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
            self.calibration_factor = pressure_pa / rms

            with open("calibration.json", "w") as f:
                json.dump({"factor": self.calibration_factor}, f, indent=4)

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

        self.tonality_mode = self.tonality_method_var.get()
        threading.Thread(target=self.analyze_file, daemon=True).start()

    def analyze_file(self):
        try:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Loading file...\n\n")

            signal, fs = load_audio(self.filename)
            signal = signal * self.calibration_factor
            duration = len(signal) / fs

            self.result_text.insert(tk.END, f"File: {Path(self.filename).name}\n")
            self.result_text.insert(tk.END, f"Sample Rate: {fs} Hz\n")
            self.result_text.insert(tk.END, f"Duration: {duration:.2f} sec\n")
            self.result_text.insert(
                tk.END, f"Tonality Standard: {self.tonality_mode}\n\n"
            )

            N, N_specific, bark_axis = loudness_zwst(signal, fs)
            self.bark_axis = bark_axis
            self.N_specific = N_specific

            phon = sone_to_phon(float(N))
            sharpness = sharpness_din_from_loudness(N, N_specific)
            dba = calculate_dba(signal, fs)

            self.compute_profile(signal, fs)
            tonality_png = self.plot_tonality(signal, fs)
            self.plot_specific_loudness(bark_axis, N_specific)
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

            self.result_text.insert(tk.END, "\n===== OVERALL RESULTS =====\n\n")
            self.result_text.insert(tk.END, f"Loudness      : {N:.2f} sones\n")
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
                summary_file = Path(self.filename).with_suffix(".summary.xlsx")
                wb = Workbook()
                ws = wb.active
                ws.title = "Summary"

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
                ws["B1"] = "Loudness (Sone)"
                ws["C1"] = "Phon"
                ws["D1"] = "Sharpness (Acum)"
                ws["E1"] = "SPL dB(A)"
                ws["F1"] = "Peak Frequency (Hz)"
                ws["G1"] = tonality_header

                ws["A2"] = Path(self.filename).name
                ws["B2"] = round(float(N), 2)
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

            os.startfile(html_file)

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

    def compute_profile(self, signal, fs):
        window = int(0.5 * fs)
        hop = int(0.1 * fs)
        times = []
        sones = []

        self.result_text.insert(tk.END, "Generating loudness profile...\n")

        for start in range(0, len(signal) - window, hop):
            try:
                frame = signal[start : start + window]
                result = loudness_zwst(frame, fs)
                loudness = float(result[0])
                times.append(start / fs)
                sones.append(loudness)
            except Exception:
                pass

        if len(sones):
            avg_sone = np.mean(sones)
            min_sone = np.min(sones)
            max_sone = np.max(sones)

            self.result_text.insert(
                tk.END, f"Average Loudness : {avg_sone:.2f} sones\n"
            )
            self.result_text.insert(
                tk.END, f"Peak Loudness    : {max_sone:.2f} sones\n"
            )
            self.result_text.insert(
                tk.END, f"Minimum Loudness : {min_sone:.2f} sones\n\n"
            )

        if self.save_csv_var.get():
            profile_df = pd.DataFrame({"Time_s": times, "Loudness_Sone": sones})
            csv_file = Path(self.filename).with_suffix(".loudness.csv")
            profile_df.to_csv(csv_file, index=False)
            self.result_text.insert(tk.END, f"Profile saved:\n{csv_file}\n\n")

        self.time_axis = times
        self.loudness_time = sones

        plt.figure(figsize=(10, 5))
        plt.plot(times, sones, color="blue")
        plt.title("Loudness vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Loudness (sone)")
        plt.grid(True)
        plt.tight_layout()

        profile_png = Path(self.filename).with_suffix(".loudness.png")
        plt.savefig(profile_png, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_specific_loudness(self, bark_axis, specific_loudness):
        plt.figure(figsize=(10, 5))
        plt.plot(bark_axis, specific_loudness, color="red", linewidth=2)
        plt.title("Specific Loudness Distribution")
        plt.xlabel("Critical Band Rate (Bark)")
        plt.ylabel("Specific Loudness (sone/Bark)")
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
                "Loudness vs Time",
                tonality_subplot_title,
                "Frequency Spectrum (Welch PSD)",
                "Specific Loudness",
            ),
            vertical_spacing=0.10,
        )

        fig.add_trace(
            go.Scatter(x=time_axis, y=loudness_time, mode="lines", name="Loudness"),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Loudness (Sone)", row=1, col=1)

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
        fig.update_yaxes(title_text="Sone/Bark", row=4, col=1)

        fig.update_layout(
            height=1800,
            title=(
                f"{Path(filename).name}<br>"
                f"Loudness={loudness:.2f} sone | "
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
                <td><strong>Loudness (sone)</strong></td>
                <td>
                    <strong>Definition:</strong> Linear psychoacoustic scale of perceived volume where doubling the sone value doubles the perceived loudness.<br>
                    <strong>Calibration/Baseline:</strong> Calculated via ISO 532 (Zwicker method) scaled against a calibration factor derived from reference SPL meter measurements.
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
            <tr>
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
            <tr><td><strong>Tonality Standard</strong></td><td>{self.tonality_mode}</td></tr>
            <tr><td><strong>Loudness</strong></td><td>{loudness:.2f} sone</td></tr>
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