# 🎥 Video Speed Processor

> **Automatically speed up silence in video recordings with a GUI**
> Save editing time — speed up only the silent parts and keep speech natural!

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/philornot/VideoSpeedProcessor)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/AIslop-violet.svg)](https://github.com/philornot/VideoSpeedProcessor)

---

## 🎯 What is this?

**Video Speed Processor** is a tool for content creators that automatically:

* 🎤 **Detects silent parts** in video recordings
* ⚡ **Speeds them up 2-5x** with visible overlay
* 🎬 **Exports a ready timeline** to DaVinci Resolve
* 💾 **Saves time** on tedious editing

### Before / After

| Without this tool                    | With this tool                               |
| ------------------------------------ | -------------------------------------------- |
| ⏱️ 45 min recording → 45 min editing | ⏱️ 45 min recording → **25 min** final video |
| 🔄 Manual trimming of every pause    | 🤖 **Automatic** detection and speeding      |
| 😫 Long, boring pauses               | ⚡ **Dynamic** transitions with overlays      |

---

## ✨ Features

### 🖥️ Simple GUI

* No CLI - everything in one window
* Drag & drop folders
* Live preview of detected files
* Smart progress bar with ETA

### 🧠 Smart speech detection

* **🤖 Whisper AI** - most accurate (multi-language)
* **⚡ WebRTC VAD** - fastest (real-time)
* **📊 Energy analysis** - fallback option (always works)

### 🎨 Professional overlays

* Visible "x3" indicator in corner
* Auto-resizes to video resolution
* Fallback to colored boxes (no ImageMagick required)
* Customizable colors and position

### 🎬 Pro-level export

* **EDL timeline** - native for DaVinci Resolve
* **Ready MP4 files** - with built-in overlays
* **Frame-accurate timecode**
* **Batch processing** - hundreds of files at once

---

## 🚀 Getting Started

### 1. Installation (5 minutes)

```bash
# Clone the repo
git clone https://github.com/philornot/VideoSpeedProcessor
cd video-speed-processor

# Install everything in one command
pip install -r requirements.txt

# DONE! Run the program
python video_speed_processor.py
```

### 2. First Use

1. 📁 Select a folder with MP4 files
2. ⚙️ Set speed (default: 3x is ideal for most)
3. 🚀 Click "Process" and wait
4. 🎬 Import to DaVinci: `File → Import → Timeline → timeline.edl`

### 3. Done!

Your timeline with automatic speed effects is ready for further editing!

---

## 📋 System Requirements

### Minimum ✅

* **Python 3.11+**
* **FFmpeg** (in PATH)
* **4GB RAM**
* **Windows 10/macOS 10.15/Ubuntu 20.04**

### Recommended 🚀

* **Python 3.12**
* **8GB+ RAM**
* **SSD** (faster processing)
* **Dedicated GPU** (for Whisper)

---

## 🎞️ Interface Overview

```
┌─ Video Speed Processor ─────────────────────────────┐
│ Status: ✅ All libraries OK                         │
├─────────────────────────────────────────────────────┤
│ 📁 Input folder: [videos/]              [Browse]     │
│    ✅ Found 15 MP4 files (2.3 GB)                  │
│                                                    │
│ 📤 Output folder: [output/]              [Browse]    │
├─────────────────────────────────────────────────────┤
│ ⚙️ Settings:                                        │
│    Speed:             [●────────] 3.0x              │
│    Min silence:       [●──────] 1.5s                │
│    Detection:         (●) WebRTC  ( ) Whisper       │
│                                                    │
│ 📤 Output: [✓] MP4 Video  [✓] EDL Timeline          │
├─────────────────────────────────────────────────────┤
│ [🚀 Start] [⏹️ Stop] [❓ Help] [📁 Results]        │
├─────────────────────────────────────────────────────┤
│ 📊 Progress: Processing video_05.mp4 (8/15)         │
│ ████████████████░░░░ 75% | ETA: ~3 min              │
├─────────────────────────────────────────────────────┤
│ 📋 Logs:                                   [Clear]   │
│ [10:30] ✅ Done: gameplay_intro.mp4                 │
│ [10:32] ⚡ WebRTC: 23 segments in 2.1s              │
│ [10:35] 🎬 Overlay added: x3 (45 segments)          │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Config Examples

| 🎥 Video Type                | ⚡ Speed | ⏱ Min. Silence | 🧠 Detection | 💡 Why                                |
| ---------------------------- | ------- | -------------- | ------------ | ------------------------------------- |
| **🎮 Gaming + Commentary**   | `3.0x`  | `1.5s`         | WebRTC       | Fast detection of speech pauses       |
| **📚 Tutorial/Presentation** | `2.5x`  | `2.0s`         | Whisper      | Precise detection of technical speech |
| **🎧 Podcast/Interview**     | `2.0x`  | `1.0s`         | Whisper      | Natural pacing for conversations      |
| **🎬 Gameplay no speech**    | `5.0x`  | `3.0s`         | WebRTC       | Aggressive cut of loading/menu parts  |
| **📹 Vlog/Lifestyle**        | `2.0x`  | `1.5s`         | Whisper      | Keeps natural vibe                    |

---

## 🎬 DaVinci Resolve Import

### Method 1: EDL Timeline ⭐ *Recommended*

```bash
1. File → Import → Timeline → Pre-Conform
2. Select: output/timeline.edl
3. ✅ Done! Effects applied automatically
```

### Method 2: MP4 with overlays

```bash
1. Import *_processed.mp4 to Media Pool
2. Drag onto timeline
3. Overlays already baked into the video
```

---

## 🔍 Troubleshooting

<details>
<summary><strong>"No MP4 files found"</strong></summary>

**Fixes:**

* Check folder path and content
* Rename files to lowercase `.mp4`
* Ensure read permissions

</details>

<details>
<summary><strong>"FFmpeg not working"</strong></summary>

**Fixes:**

* Check installation: `ffmpeg -version`
* Add to PATH if needed

</details>

<details>
<summary><strong>"No silent segments detected"</strong></summary>

**Fixes:**

* Lower min silence to `1.0s`
* Try different detection method
* Check logs for debug info

</details>

<details>
<summary><strong>"App freezes or crashes"</strong></summary>

**Causes:**

* Low RAM or large video
* Use WebRTC instead of Whisper
* Ensure enough disk space

</details>

---

## 📊 Performance Tips

| 🎥 Length  | 🧠 WebRTC   | 🤖 Whisper | 💻 System       |
| ---------- | ----------- | ---------- | --------------- |
| 5 minutes  | 30s - 2 min | 2-8 min    | i5 + 8GB RAM    |
| 30 minutes | 3-10 min    | 15-45 min  | i5 + 8GB RAM    |
| 2 hours    | 20-40 min   | 1-3 hrs    | i5 + 8GB RAM    |
| 5 minutes  | 15s - 1 min | 1-4 min    | i7 + 16GB + GPU |

**Speed it up:**

* Prefer WebRTC over Whisper
* Close unused apps
* Use SSD
* More RAM = less swapping
* Process fewer files at a time

---

## 📈 Example Use Cases

### 🎮 Gaming Creator

```
Problem: 45-minute gameplay with long pauses
Solution: Speed 3x, WebRTC, export to YouTube
Result: 25-minute dynamic cut with overlay x3
```

### 📚 Online Instructor

```
Problem: 90-minute training with thinking pauses
Solution: Speed 2x, Whisper, export to LMS
Result: 65-minute tight presentation
```

### 🎧 Podcaster

```
Problem: 120-min interview with pauses
Solution: Speed 1.5x, Whisper, mild trimming
Result: 95-minute natural flow
```

---

## 📄 License

MIT License — Free to use, modify, distribute.

Just keep:

* 📋 Original copyright
* 📋 Copy of license

Uses FFmpeg (LGPL) and OpenAI Whisper (MIT).

---

## 💕 Credits

Made possible thanks to amazing open-source projects:

* 🎬 **MoviePy** - Python video editing
* 🎵 **Librosa** - Audio analysis
* 🤖 **OpenAI Whisper** - Speech recognition
* ⚡ **WebRTC VAD** - Voice detection
* 🔧 **FFmpeg** - Multimedia processing

## 🎥 Setup

```bash
git clone https://github.com/philornot/VideoSpeedProcessor
cd video-speed-processor
pip install -r requirements.txt
python video_speed_processor.py
```

**First launch may take a few minutes (downloads models).**

**Happy editing! ✨**
