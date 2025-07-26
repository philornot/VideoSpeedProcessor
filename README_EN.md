# Video Speed Processor

Automatically process recordings - speed up silent segments while maintaining editability in DaVinci Resolve.

## 🎯 Project Goal

This program automatically detects silent segments in video recordings, speeds them up (e.g., x3) with visible text overlay, and exports to DaVinci Resolve in an editable format. This allows you to quickly prepare recordings for further editing without losing quality and timeline control.

## ✨ Features

- **🎤 Intelligent speech detection** - WebRTC VAD or Whisper AI
- **⚡ Silence acceleration** - configurable speed (x2, x3, x5...)
- **📺 Text overlay** - visible multiplier "x3" in bottom right corner
- **🎬 DaVinci Resolve export** - FCPXML format with preserved editability
- **📁 Batch processing** - process multiple folders simultaneously
- **🎞️ Ready MP4 files** - optionally with built-in overlays

## 📋 Requirements

- **Python 3.11+**
- **FFmpeg** (in PATH)
- **Windows 10/11** (primarily tested)

### FFmpeg Installation
```bash
# Via chocolatey (recommended)
choco install ffmpeg

# Manually: download from https://ffmpeg.org/ and add to PATH
```

## 🚀 Installation

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd video-speed-processor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Whisper for better detection**
   ```bash
   pip install openai-whisper
   ```

## 💻 Usage

### Basic Usage
```bash
# Process all MP4 files from folder, speeding up silence x3
python video_processor.py --input_folder "videos/" --speed_multiplier 3.0

# Also generate ready MP4 files
python video_processor.py --input_folder "clips/" --speed_multiplier 2.5 --generate_video

# Use Whisper for better speech detection
python video_processor.py --input_folder "input/" --speed_multiplier 3.0 --use_whisper
```

### CLI Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--input_folder` | Folder with MP4 files *(required)* | - | `videos/` |
| `--output_folder` | Output folder | `output` | `processed/` |
| `--speed_multiplier` | Silence multiplier *(required)* | - | `3.0` |
| `--min_silence_duration` | Min. silence length (s) | `1.5` | `2.0` |
| `--generate_video` | Generate MP4 files | `False` | `--generate_video` |
| `--generate_timeline` | Generate FCPXML | `True` | `--generate_timeline` |
| `--combine_clips` | Combine all clips | `False` | `--combine_clips` |
| `--use_whisper` | Use Whisper AI | `False` | `--use_whisper` |
| `--debug` | Debug mode | `False` | `--debug` |

### Batch Processing
```bash
# Process all folders with recordings
python batch_processor.py --input_root "D:\Recordings\awesomegameplayclips" --speed_multiplier 3.0

# With automatic confirmation
python batch_processor.py --input_root "recordings/" --speed_multiplier 2.5 --auto_confirm
```

## 📁 Output Structure

```
output/
├── timeline.fcpxml              # Timeline for DaVinci Resolve
├── timeline_data.json           # Technical segment data
├── video1_processed.mp4         # Processed video (optional)
├── video2_processed.mp4         # ...
├── combined_video.mp4           # Combined video (optional)
└── video_processor.log          # Processing logs
```

## 🎬 DaVinci Resolve Import

### Method 1: FCPXML Timeline (recommended) ⭐
1. **File → Import → Timeline**
2. Select `timeline.fcpxml`
3. Timeline will be imported with all segments and speed effects
4. Original MP4 files must be accessible

### Method 2: Ready MP4 Files
1. Import processed `.mp4` files to Media Pool
2. Drag to timeline
3. "x3" overlay is built into the video

### Method 3: Combined Video
1. Import `combined_video.mp4`
2. Ready for further editing

## ⚙️ Configuration for Different Content

| Recording Type | Speed Multiplier | Min Silence | Detection |
|----------------|------------------|-------------|-----------|
| 🎮 Gaming with commentary | 2.5-3.0 | 1.5s | WebRTC |
| 📚 Tutorial/Presentation | 2.0-2.5 | 2.0s | Whisper |
| 🎙️ Podcast/Conversation | 1.5-2.0 | 1.0s | Whisper |
| 🎮 Gameplay without voice | 4.0-5.0 | 3.0s | WebRTC |

## 🔧 Troubleshooting

### FFmpeg not working
```bash
# Check installation
ffmpeg -version

# Install via chocolatey
choco install ffmpeg
```

### Missing libraries
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or basic ones
pip install moviepy librosa numpy
```

### No silence segments
- Decrease `--min_silence_duration` (e.g., 1.0)
- Enable `--debug` for diagnostics
- Try `--use_whisper`

### Memory issues
- Process smaller files
- Close other applications
- Consider increasing RAM

## 📈 Example Workflow

### 1. Preparation
```bash
# Copy recordings to folder
mkdir input
copy "C:\Users\Username\Videos\*.mp4" input\
```

### 2. Processing
```bash
python video_processor.py \
    --input_folder input/ \
    --output_folder processed/ \
    --speed_multiplier 3.0 \
    --generate_video \
    --use_whisper
```

### 3. DaVinci Import
1. **File → Import → Timeline** → `processed/timeline.fcpxml`
2. Edit color grading and effects
3. **Deliver** → Final export

## 🎨 Overlay Customization

In `video_processor.py` you can modify the "x3" text appearance:

```python
# In create_speed_overlay() function
txt_clip = mp.TextClip(
    text,
    fontsize=50,           # Size
    color='yellow',        # Color
    font='Arial-Black',    # Font
    stroke_color='red',    # Stroke color
    stroke_width=3         # Stroke width
)

# Change position (top left corner)
txt_clip = txt_clip.set_position((margin, margin))
```

## 📊 Performance

**Estimated processing time:**
- **5-minute video**: 2-5 min (WebRTC) / 5-15 min (Whisper)
- **30-minute video**: 10-20 min (WebRTC) / 30-60 min (Whisper)

*Depends on: computer power, amount of silence, audio quality*

## 🤝 Support

### Diagnostics
```bash
# Enable debug mode
python video_processor.py --input_folder videos/ --speed_multiplier 3.0 --debug

# Check logs
cat video_processor.log  # Linux/Mac
type video_processor.log  # Windows
```

### Common Issues
1. **FFmpeg** - check `ffmpeg -version`
2. **Libraries** - check `pip list`
3. **Segmentation** - use `--debug`
4. **Memory** - process smaller files

## 📄 License

MIT License - you can freely use and modify.

---

**Happy editing! 🎬✨**