#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Speed Processor for SteelSeries Moments
Automatycznie przetwarza nagrania wideo, przyspieszając fragmenty ciszy
i eksportując do DaVinci Resolve.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json


# Importy zewnętrzne
def check_dependencies():
    """Sprawdza dostępność wymaganych bibliotek."""
    missing_libs = []

    try:
        import moviepy.editor as mp
    except ImportError:
        missing_libs.append("moviepy")

    try:
        import librosa
        import numpy as np
    except ImportError:
        missing_libs.append("librosa i numpy")

    if missing_libs:
        print("❌ Błąd: Brak wymaganych bibliotek:")
        for lib in missing_libs:
            print(f"   • {lib}")
        print("\n🔧 Rozwiązanie:")
        print("   1. Aktywuj środowisko: venv\\Scripts\\activate")
        print("   2. Zainstaluj biblioteki: pip install -r requirements.txt")
        print("   3. Lub użyj: run_processor.bat zamiast python video_processor.py")
        sys.exit(1)


# Sprawdź zależności na początku
check_dependencies()

try:
    import moviepy.editor as mp
    from moviepy.video.tools.drawing import color_gradient
except ImportError:
    print("Błąd: Brak biblioteki moviepy. Zainstaluj: pip install moviepy")
    sys.exit(1)

try:
    import librosa
    import numpy as np
except ImportError:
    print("Błąd: Brak bibliotek audio. Zainstaluj: pip install librosa numpy")
    sys.exit(1)

# Opcjonalne importy
try:
    import webrtcvad

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("Uwaga: webrtcvad niedostępne. Użyj --use_whisper lub zainstaluj: pip install webrtcvad")

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Uwaga: whisper niedostępne. Zainstaluj: pip install openai-whisper")


class VideoProcessor:
    """Główna klasa do przetwarzania wideo."""

    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Konfiguracja logowania."""
        level = logging.DEBUG if self.config.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('video_processor.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)

    def detect_speech_segments(self, audio_path: str) -> List[Tuple[float, float, bool]]:
        """
        Wykrywa segmenty mowy i ciszy w pliku audio.
        Zwraca listę tupli: (start_time, end_time, is_speech)
        """
        self.logger.info(f"Wykrywanie segmentów mowy w: {audio_path}")

        if self.config.use_whisper and WHISPER_AVAILABLE:
            return self._detect_speech_whisper(audio_path)
        elif WEBRTC_AVAILABLE:
            return self._detect_speech_webrtc(audio_path)
        else:
            return self._detect_speech_energy(audio_path)

    def _detect_speech_whisper(self, audio_path: str) -> List[Tuple[float, float, bool]]:
        """Detekcja mowy używając Whisper."""
        self.logger.info("Używam Whisper do detekcji mowy...")

        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, word_timestamps=True)

            segments = []
            audio_duration = librosa.get_duration(filename=audio_path)

            # Konwertuj wyniki Whisper na segmenty
            current_time = 0.0

            for segment in result['segments']:
                start = segment['start']
                end = segment['end']

                # Dodaj ciszę przed segmentem
                if start > current_time + self.config.min_silence_duration:
                    segments.append((current_time, start, False))  # Cisza

                # Dodaj segment mowy
                segments.append((start, end, True))  # Mowa
                current_time = end

            # Dodaj pozostałą ciszę na końcu
            if current_time < audio_duration:
                segments.append((current_time, audio_duration, False))

            return segments

        except Exception as e:
            self.logger.error(f"Błąd Whisper: {e}")
            return self._detect_speech_energy(audio_path)

    def _detect_speech_webrtc(self, audio_path: str) -> List[Tuple[float, float, bool]]:
        """Detekcja mowy używając WebRTC VAD."""
        self.logger.info("Używam WebRTC VAD do detekcji mowy...")

        try:
            # Wczytaj audio
            y, sr = librosa.load(audio_path, sr=16000)  # WebRTC wymaga 16kHz

            # Inicjalizuj VAD
            vad = webrtcvad.Vad(2)  # Agresywność 0-3

            # Podziel na ramki 30ms
            frame_duration = 0.03  # 30ms
            frame_length = int(sr * frame_duration)

            segments = []
            current_segment_start = 0.0
            current_is_speech = False

            for i in range(0, len(y) - frame_length, frame_length):
                frame = y[i:i + frame_length]

                # Konwertuj do int16
                frame_int16 = (frame * 32767).astype(np.int16).tobytes()

                # Sprawdź czy to mowa
                is_speech = vad.is_speech(frame_int16, sr)
                frame_time = i / sr

                # Jeśli zmienił się typ segmentu
                if is_speech != current_is_speech:
                    if frame_time - current_segment_start >= self.config.min_silence_duration:
                        segments.append((current_segment_start, frame_time, current_is_speech))
                    current_segment_start = frame_time
                    current_is_speech = is_speech

            # Dodaj ostatni segment
            audio_duration = len(y) / sr
            if audio_duration - current_segment_start >= self.config.min_silence_duration:
                segments.append((current_segment_start, audio_duration, current_is_speech))

            return segments

        except Exception as e:
            self.logger.error(f"Błąd WebRTC VAD: {e}")
            return self._detect_speech_energy(audio_path)

    def _detect_speech_energy(self, audio_path: str) -> List[Tuple[float, float, bool]]:
        """Prosta detekcja na podstawie energii audio."""
        self.logger.info("Używam detekcji energii audio...")

        try:
            y, sr = librosa.load(audio_path)

            # Oblicz RMS energy w oknach
            frame_length = int(sr * 0.1)  # 100ms okna
            hop_length = int(sr * 0.05)  # 50ms przesunięcie

            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Próg na podstawie percentyla
            threshold = np.percentile(rms, 30)  # 30% najcichszych fragmentów to cisza

            # Konwertuj na segmenty czasowe
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

            segments = []
            current_start = 0.0
            current_is_speech = rms[0] > threshold

            for i, (time, energy) in enumerate(zip(times, rms)):
                is_speech = energy > threshold

                if is_speech != current_is_speech:
                    if time - current_start >= self.config.min_silence_duration:
                        segments.append((current_start, time, current_is_speech))
                    current_start = time
                    current_is_speech = is_speech

            # Dodaj ostatni segment
            audio_duration = len(y) / sr
            if audio_duration - current_start >= self.config.min_silence_duration:
                segments.append((current_start, audio_duration, current_is_speech))

            return segments

        except Exception as e:
            self.logger.error(f"Błąd detekcji energii: {e}")
            return []

    def create_speed_overlay(self, duration: float, speed: float, size: Tuple[int, int]) -> mp.VideoClip:
        """Tworzy overlay z tekstem prędkości."""
        if speed == 1.0:
            return None

        text = f"x{speed:.1f}" if speed != int(speed) else f"x{int(speed)}"

        # Utwórz tekstowy klip
        txt_clip = mp.TextClip(
            text,
            fontsize=min(size[0], size[1]) // 20,  # Dynamiczny rozmiar czcionki
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2
        ).set_duration(duration)

        # Pozycjonuj w prawym dolnym rogu
        margin = min(size[0], size[1]) // 40
        txt_clip = txt_clip.set_position((size[0] - txt_clip.w - margin, size[1] - txt_clip.h - margin))

        return txt_clip

    def process_video_file(self, input_path: str, output_folder: str) -> dict:
        """Przetwarza pojedynczy plik wideo."""
        self.logger.info(f"Przetwarzanie: {input_path}")

        try:
            # Wczytaj wideo
            video = mp.VideoFileClip(input_path)

            # Wyodrębnij audio do tymczasowego pliku
            temp_audio = os.path.join(output_folder, "temp_audio.wav")
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)

            # Wykryj segmenty mowy
            segments = self.detect_speech_segments(temp_audio)

            # Usuń tymczasowy plik audio
            os.remove(temp_audio)

            # Przetwórz segmenty
            processed_clips = []
            timeline_data = []

            for start, end, is_speech in segments:
                segment_clip = video.subclip(start, end)

                if is_speech:
                    # Fragment mowy - normalne tempo
                    processed_clips.append(segment_clip)
                    timeline_data.append({
                        'start': start,
                        'end': end,
                        'duration': end - start,
                        'speed': 1.0,
                        'type': 'speech'
                    })
                else:
                    # Fragment ciszy - przyspiesz
                    speed = self.config.speed_multiplier
                    sped_clip = segment_clip.fx(mp.vfx.speedx, speed)

                    # Dodaj overlay z prędkością
                    overlay = self.create_speed_overlay(
                        sped_clip.duration,
                        speed,
                        (video.w, video.h)
                    )

                    if overlay:
                        sped_clip = mp.CompositeVideoClip([sped_clip, overlay])

                    processed_clips.append(sped_clip)
                    timeline_data.append({
                        'start': start,
                        'end': end,
                        'duration': sped_clip.duration,
                        'original_duration': end - start,
                        'speed': speed,
                        'type': 'silence'
                    })

            # Połącz wszystkie klipy
            if processed_clips:
                final_video = mp.concatenate_videoclips(processed_clips)

                # Przygotuj nazwy plików wyjściowych
                input_name = Path(input_path).stem

                result = {
                    'input_file': input_path,
                    'input_name': input_name,
                    'timeline_data': timeline_data,
                    'output_duration': final_video.duration,
                    'original_duration': video.duration,
                    'segments_count': len(segments)
                }

                # Zapisz wideo jeśli wymagane
                if self.config.generate_video:
                    output_video_path = os.path.join(output_folder, f"{input_name}_processed.mp4")
                    final_video.write_videofile(
                        output_video_path,
                        codec='libx264',
                        audio_codec='aac',
                        verbose=False,
                        logger=None
                    )
                    result['output_video'] = output_video_path
                    self.logger.info(f"Zapisano wideo: {output_video_path}")

                # Zamknij klipy
                final_video.close()
                video.close()

                return result
            else:
                self.logger.warning(f"Brak segmentów do przetworzenia w: {input_path}")
                video.close()
                return None

        except Exception as e:
            self.logger.error(f"Błąd przetwarzania {input_path}: {e}")
            return None

    def generate_fcpxml(self, results: List[dict], output_path: str):
        """Generuje plik FCPXML dla DaVinci Resolve."""
        self.logger.info(f"Generowanie FCPXML: {output_path}")

        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom

            # Główny element FCPXML
            fcpxml = Element('fcpxml', version='1.8')

            # Zasób projektowy
            resources = SubElement(fcpxml, 'resources')

            # Projekt
            project = SubElement(fcpxml, 'project', name='Processed_Videos')
            sequence = SubElement(project, 'sequence',
                                  format='r1',
                                  duration='3600s',
                                  tcStart='0s')
            spine = SubElement(sequence, 'spine')

            current_offset = 0
            asset_id = 1

            for result in results:
                if not result:
                    continue

                # Dodaj zasób wideo
                asset = SubElement(resources, 'asset',
                                   id=f'r{asset_id}',
                                   name=result['input_name'],
                                   src=result['input_file'],
                                   duration=f"{result['original_duration']}s")

                # Dodaj klipy do timeline
                for segment in result['timeline_data']:
                    clip = SubElement(spine, 'video',
                                      ref=f'r{asset_id}',
                                      offset=f'{current_offset}s',
                                      duration=f"{segment['duration']}s",
                                      start=f"{segment['start']}s")

                    # Dodaj informacje o prędkości jako efekt
                    if segment['speed'] != 1.0:
                        effect = SubElement(clip, 'timeMap')
                        timept = SubElement(effect, 'timept',
                                            time='0s',
                                            value='0s')
                        timept2 = SubElement(effect, 'timept',
                                             time=f"{segment['duration']}s",
                                             value=f"{segment['original_duration']}s")

                    current_offset += segment['duration']

                asset_id += 1

            # Zapisz XML
            rough_string = tostring(fcpxml, 'utf-8')
            reparsed = minidom.parseString(rough_string)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(reparsed.toprettyxml(indent='  '))

            self.logger.info(f"FCPXML zapisany: {output_path}")

        except Exception as e:
            self.logger.error(f"Błąd generowania FCPXML: {e}")

    def combine_videos(self, results: List[dict], output_path: str):
        """Łączy wszystkie przetworzone wideo w jeden plik."""
        self.logger.info("Łączenie wszystkich wideo...")

        try:
            video_paths = [r.get('output_video') for r in results if r and r.get('output_video')]

            if not video_paths:
                self.logger.warning("Brak plików wideo do połączenia")
                return

            clips = [mp.VideoFileClip(path) for path in video_paths]
            final_video = mp.concatenate_videoclips(clips)

            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )

            # Zamknij klipy
            final_video.close()
            for clip in clips:
                clip.close()

            self.logger.info(f"Połączone wideo zapisane: {output_path}")

        except Exception as e:
            self.logger.error(f"Błąd łączenia wideo: {e}")

    def process_folder(self):
        """Przetwarza wszystkie pliki MP4 z folderu."""
        input_folder = Path(self.config.input_folder)
        output_folder = Path(self.config.output_folder)

        # Utwórz folder wyjściowy
        output_folder.mkdir(parents=True, exist_ok=True)

        # Znajdź wszystkie pliki MP4
        mp4_files = list(input_folder.glob("*.mp4"))

        if not mp4_files:
            self.logger.error(f"Nie znaleziono plików MP4 w: {input_folder}")
            return

        self.logger.info(f"Znaleziono {len(mp4_files)} plików MP4")

        # Przetwórz każdy plik
        results = []
        for mp4_file in mp4_files:
            self.logger.info(f"Przetwarzanie {mp4_file.name}...")
            result = self.process_video_file(str(mp4_file), str(output_folder))
            if result:
                results.append(result)

        # Zapisz dane timeline jako JSON
        timeline_json = output_folder / "timeline_data.json"
        with open(timeline_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Generuj timeline jeśli wymagane
        if self.config.generate_timeline:
            fcpxml_path = output_folder / "timeline.fcpxml"
            self.generate_fcpxml(results, str(fcpxml_path))

        # Połącz wideo jeśli wymagane
        if self.config.combine_clips and self.config.generate_video:
            combined_path = output_folder / "combined_video.mp4"
            self.combine_videos(results, str(combined_path))

        self.logger.info(f"Przetwarzanie zakończone. Wyniki w: {output_folder}")
        print(f"\n✅ Przetworzono {len(results)} plików")
        print(f"📁 Wyniki zapisane w: {output_folder}")
        if self.config.generate_timeline:
            print(f"🎬 Timeline FCPXML: {output_folder}/timeline.fcpxml")


def main():
    """Główna funkcja programu."""
    parser = argparse.ArgumentParser(
        description="Video Speed Processor - automatyczne przetwarzanie nagrań SteelSeries Moments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python video_processor.py --input_folder videos/ --speed_multiplier 3.0
  python video_processor.py --input_folder clips/ --speed_multiplier 2.5 --generate_video --combine_clips
  python video_processor.py --input_folder input/ --use_whisper --min_silence_duration 2.0
        """
    )

    parser.add_argument('--input_folder', required=True,
                        help='Folder z plikami MP4 do przetworzenia')
    parser.add_argument('--output_folder', default='output',
                        help='Folder wyjściowy (domyślnie: output)')
    parser.add_argument('--speed_multiplier', type=float, required=True,
                        help='Mnożnik prędkości dla fragmentów ciszy (np. 3.0)')
    parser.add_argument('--min_silence_duration', type=float, default=1.5,
                        help='Minimalna długość ciszy w sekundach (domyślnie: 1.5)')
    parser.add_argument('--generate_video', action='store_true',
                        help='Generuj gotowe pliki MP4')
    parser.add_argument('--generate_timeline', action='store_true', default=True,
                        help='Generuj timeline FCPXML (domyślnie: True)')
    parser.add_argument('--combine_clips', action='store_true',
                        help='Połącz wszystkie klipy w jeden film')
    parser.add_argument('--use_whisper', action='store_true',
                        help='Użyj Whisper do detekcji mowy')
    parser.add_argument('--debug', action='store_true',
                        help='Włącz tryb debugowania')

    args = parser.parse_args()

    # Sprawdź czy folder wejściowy istnieje
    if not os.path.exists(args.input_folder):
        print(f"❌ Błąd: Folder {args.input_folder} nie istnieje!")
        sys.exit(1)

    # Sprawdź dostępność bibliotek
    if args.use_whisper and not WHISPER_AVAILABLE:
        print("❌ Błąd: Whisper niedostępny. Zainstaluj: pip install openai-whisper")
        sys.exit(1)

    if not args.use_whisper and not WEBRTC_AVAILABLE:
        print("⚠️  Uwaga: WebRTC VAD niedostępny. Używam detekcji energii.")
        print("   Dla lepszych wyników zainstaluj: pip install webrtcvad")
        print("   lub użyj --use_whisper")

    # Utwórz procesor i uruchom
    try:
        processor = VideoProcessor(args)
        processor.process_folder()
    except KeyboardInterrupt:
        print("\n⏹️  Przerwano przez użytkownika")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()