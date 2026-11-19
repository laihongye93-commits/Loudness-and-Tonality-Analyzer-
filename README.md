# Loudness & Tonality Analyzer

A Tkinter desktop application in Python for evaluating audio recordings across psychoacoustic and environmental acoustic metrics, complete with SPL calibration, standard persistence, and XLSX/CSV/HTML reporting.

## Features

* **Loudness Standards**: Supports **ISO 532-1 (Zwicker)** (`sones` / `sone`) via `mosqito` and **ECMA-418 (HMS)** (`soneHMS`) via `sottek-hearing-model`.
* **Tonality Standards**: Supports **DIN 45681** (`Kt` in dB, `ΔL`, and tone frequency over time) and **ECMA-418 (HMS)** (`tuHMS`) via `sottek-hearing-model`.
* **Additional Acoustic Metrics**: Computes **Loudness Level (`phon`)**, **Sharpness (`acum`, DIN)**, **A-weighted Sound Pressure Level (`dB(A)`)**, and **Welch PSD Frequency Spectrum** (including peak frequency and peak detection).
* **SPL Calibration**: Calibrates audio pressure scaling (`Pa`) from a reference audio recording and a measured SPL value (`dB`).
* **Persistent Configuration (`config.json`)**: Automatically saves and restores `factor`, `loudness_standard`, and `tonality_standard` (with fallback support for legacy `calibration.json` and `standards.json` files).
* **Export & Reporting**:
  * Optional **XLSX Summary Report** (`<audio_name>.summary.xlsx`) with a `Summary` sheet (metrics + embedded plots) and a `Tonality` sheet[cite: 1].
  * Optional **Loudness CSV Profile** (`<audio_name>.loudness.csv`)[cite: 1].
  * Optional **PNG Plots** (`.loudness.png`, `.specific_loudness.png`, `.frequency_spectrum.png`, `.tonality.png`), with automatic cleanup if unchecked[cite: 1].
  * **Interactive HTML Report** (`<audio_name>.summary.html`) with a 4-panel Plotly chart, summary/tonality tables, and an acoustic metrics reference guide, opened automatically via `os.startfile`[cite: 1].

## Requirements

* **Python**: 3.x with `tkinter`[cite: 1]
* **Core Python Packages**:
  * `numpy`[cite: 1]
  * `pandas`[cite: 1]
  * `scipy`[cite: 1]
  * `matplotlib`[cite: 1]
  * `plotly`[cite: 1]
  * `openpyxl`[cite: 1]
  * `pydub`[cite: 1]
  * `mosqito`[cite: 1]
* **Optional Package (for ECMA-418 / HMS features)**:
  * `sottek-hearing-model`[cite: 1]
* **System Dependency**: `ffmpeg` / `ffprobe` available on your system `PATH` for `pydub.AudioSegment.from_file(...)` to decode supported audio formats (`*.wav`, `*.flac`, `*.aiff`, `*.aif`, `*.mp3`, `*.m4a`, `*.mp4`, `*.aac`)[cite: 1].

## Usage

1. Run the application:
   ```bash
   python Loudness_calculator_1.py
