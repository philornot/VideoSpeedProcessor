# Video Speed Processor

Automatyczne przetwarzanie nagrań wideo - przyspiesza fragmenty ciszy zachowując edytowalność w DaVinci Resolve.

## 🎯 Cel projektu

Program automatycznie wykrywa fragmenty ciszy w nagraniach wideo, przyspiesza je (np. x3) z widocznym overlayem tekstowym i eksportuje do DaVinci Resolve w formacie edytowalnym. Dzięki temu możesz szybko przygotować nagrania do dalszego montażu bez utraty jakości i kontroli nad timeline.

## ✨ Funkcje

- **🎤 Inteligentna detekcja mowy** - WebRTC VAD lub Whisper AI
- **⚡ Przyspieszanie ciszy** - konfigurowalne tempo (x2, x3, x5...)
- **📺 Overlay tekstowy** - widoczny mnożnik "x3" w prawym dolnym rogu
- **🎬 Eksport do DaVinci Resolve** - format FCPXML z zachowaniem edytowalności
- **📁 Batch processing** - przetwarzanie wielu folderów jednocześnie
- **🎞️ Gotowe pliki MP4** - opcjonalnie z wbudowanymi overlayami

## 📋 Wymagania

- **Python 3.11+**
- **FFmpeg** (w PATH)
- **ImageMagick** (dla overlayów tekstowych)
- **Windows 10/11** (głównie testowane)

### Instalacja FFmpeg
```bash
# Przez chocolatey (zalecane)
choco install ffmpeg

# Ręcznie: pobierz z https://ffmpeg.org/ i dodaj do PATH
```

### Instalacja ImageMagick
```bash
# Automatycznie (zalecane)
python install_imagemagick.py

# Przez chocolatey
choco install imagemagick

# Ręcznie: pobierz z https://imagemagick.org/script/download.php#windows
# WAŻNE: Podczas instalacji zaznacz "Install development headers"
```

**💡 Bez ImageMagick:** Program będzie działał, ale zamiast tekstu "x3" pokaże kolorowy wskaźnik.

## 🚀 Instalacja

1. **Sklonuj repozytorium**
   ```bash
   git clone <repo-url>
   cd video-speed-processor
   ```

2. **Utwórz środowisko wirtualne**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Zainstaluj zależności**
   ```bash
   pip install -r requirements.txt
   ```

4. **Opcjonalnie: Whisper dla lepszej detekcji**
   ```bash
   pip install openai-whisper
   ```

## 💻 Użytkowanie

### Podstawowe użycie
```bash
# Przetwórz wszystkie MP4 z folderu, przyspieszając ciszę x3
python video_processor.py --input_folder "videos/" --speed_multiplier 3.0

# Generuj również gotowe pliki MP4
python video_processor.py --input_folder "clips/" --speed_multiplier 2.5 --generate_video

# Używaj Whisper dla lepszej detekcji mowy
python video_processor.py --input_folder "input/" --speed_multiplier 3.0 --use_whisper
```

### Parametry CLI

| Parametr | Opis | Domyślnie | Przykład |
|----------|------|-----------|----------|
| `--input_folder` | Folder z plikami MP4 *(wymagany)* | - | `videos/` |
| `--output_folder` | Folder wyjściowy | `output` | `processed/` |
| `--speed_multiplier` | Mnożnik dla ciszy *(wymagany)* | - | `3.0` |
| `--min_silence_duration` | Min. długość ciszy (s) | `1.5` | `2.0` |
| `--generate_video` | Generuj pliki MP4 | `False` | `--generate_video` |
| `--generate_timeline` | Generuj FCPXML | `True` | `--generate_timeline` |
| `--combine_clips` | Połącz wszystkie klipy | `False` | `--combine_clips` |
| `--use_whisper` | Użyj Whisper AI | `False` | `--use_whisper` |
| `--debug` | Tryb debugowania | `False` | `--debug` |

### Batch processing
```bash
# Przetwórz wszystkie foldery z nagraniami
python batch_processor.py --input_root "D:\Recordings\awesomegameplayclips" --speed_multiplier 3.0

# Z automatycznym potwierdzeniem
python batch_processor.py --input_root "recordings/" --speed_multiplier 2.5 --auto_confirm
```

## 📁 Struktura wyjściowa

```
output/
├── timeline.fcpxml              # Timeline do DaVinci Resolve
├── timeline_data.json           # Dane techniczne segmentów
├── video1_processed.mp4         # Przetworzone wideo (opcjonalnie)
├── video2_processed.mp4         # ...
├── combined_video.mp4           # Połączone wideo (opcjonalnie)
└── video_processor.log          # Logi przetwarzania
```

## 🎬 Import do DaVinci Resolve

### Metoda 1: Timeline FCPXML (zalecana) ⭐
1. **File → Import → Timeline**
2. Wybierz `timeline.fcpxml`
3. Timeline zostanie zaimportowany z wszystkimi segmentami i efektami prędkości
4. Oryginalne pliki MP4 muszą być dostępne

### Metoda 2: Gotowe pliki MP4
1. Zaimportuj przetworzone pliki `.mp4` do Media Pool
2. Przeciągnij na timeline
3. Overlay "x3" jest wbudowany w wideo

### Metoda 3: Połączone wideo
1. Zaimportuj `combined_video.mp4`
2. Gotowe do dalszej edycji

## ⚙️ Konfiguracja dla różnych treści

| Typ nagrania | Speed Multiplier | Min Silence | Detekcja |
|--------------|------------------|-------------|----------|
| 🎮 Gaming z komentarzem | 2.5-3.0 | 1.5s | WebRTC |
| 📚 Tutorial/Prezentacja | 2.0-2.5 | 2.0s | Whisper |
| 🎙️ Podcast/Rozmowa | 1.5-2.0 | 1.0s | Whisper |
| 🎮 Gameplay bez głosu | 4.0-5.0 | 3.0s | WebRTC |

## 🔧 Rozwiązywanie problemów

### ImageMagick nie działa
```bash
# Instalacja
choco install imagemagick

# Sprawdź instalację
magick -version
```

**Bez ImageMagick:** Program używa kolorowych wskaźników zamiast tekstu

### Brak bibliotek
```bash
# Reinstaluj wymagania
pip install -r requirements.txt

# Lub podstawowe
pip install moviepy librosa numpy
```

### Brak segmentów ciszy
- Zmniejsz `--min_silence_duration` (np. 1.0)
- Włącz `--debug` dla diagnostyki
- Spróbuj `--use_whisper`

### Problemy z pamięcią
- Przetwarzaj mniejsze pliki
- Zamknij inne aplikacje
- Rozważ zwiększenie RAM

## 📈 Przykładowy workflow

### 1. Przygotowanie
```bash
# Skopiuj nagrania do folderu
mkdir input
copy "C:\Users\Username\Videos\*.mp4" input\
```

### 2. Przetwarzanie
```bash
python video_processor.py \
    --input_folder input/ \
    --output_folder processed/ \
    --speed_multiplier 3.0 \
    --generate_video \
    --use_whisper
```

### 3. Import do DaVinci
1. **File → Import → Timeline** → `processed/timeline.fcpxml`
2. Edytuj kolorystykę i efekty
3. **Deliver** → Export końcowy

## 🎨 Customizacja overlay

W `video_processor.py` możesz zmienić wygląd tekstu "x3":

```python
# W funkcji create_speed_overlay()
txt_clip = mp.TextClip(
    text,
    fontsize=50,           # Rozmiar
    color='yellow',        # Kolor
    font='Arial-Black',    # Czcionka
    stroke_color='red',    # Kolor konturu
    stroke_width=3         # Grubość konturu
)

# Zmień pozycję (lewy górny róg)
txt_clip = txt_clip.set_position((margin, margin))
```

## 📊 Wydajność

**Orientacyjny czas przetwarzania:**
- **5-minutowe wideo**: 2-5 min (WebRTC) / 5-15 min (Whisper)
- **30-minutowe wideo**: 10-20 min (WebRTC) / 30-60 min (Whisper)

*Zależy od: mocy komputera, ilości ciszy, jakości audio*

## 🤝 Wsparcie

### Diagnostyka
```bash
# Włącz tryb debug
python video_processor.py --input_folder videos/ --speed_multiplier 3.0 --debug

# Sprawdź logi
type video_processor.log
```

### Częste problemy
1. **FFmpeg** - sprawdź `ffmpeg -version`
2. **Biblioteki** - sprawdź `pip list`
3. **Segmentacja** - użyj `--debug`
4. **Pamięć** - przetwarzaj mniejsze pliki

## 📄 Licencja

MIT License - możesz swobodnie używać i modyfikować.

## 🙏 Podziękowania

- **MoviePy** - przetwarzanie wideo
- **Librosa** - analiza audio  
- **WebRTC VAD** - detekcja mowy
- **OpenAI Whisper** - transkrypcja AI

---

**Miłego montażu! 🎬✨**