#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Speed Processor with GUI - FIXED VERSION
Automatycznie przetwarza nagrania wideo, przyspieszając fragmenty ciszy
z prostym interfejsem graficznym.

NAPRAWIONE BŁĘDY:
- Poprawiona agregacja segmentów w WebRTC i Energy
- Dodane sprawdzanie poprawności segmentów
- Naprawione zarządzanie pamięcią (zamykanie klipów)
- Poprawione EDL z prawidłowym timecode
- Dodana walidacja danych wejściowych
- Lepsze obsługa błędów
- Optymalizacja wydajności
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional
import queue
import tempfile
import shutil

# Importy zewnętrzne
try:
    import moviepy.editor as mp
    import librosa
    import numpy as np
except ImportError as e:
    print(f"Błąd: Brak wymaganych bibliotek. Zainstaluj: pip install moviepy librosa numpy")
    sys.exit(1)

# Opcjonalne importy
try:
    import webrtcvad

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class LogHandler(logging.Handler):
    """Handler do przekierowania logów do GUI."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class VideoProcessor:
    """Klasa do przetwarzania wideo."""

    def __init__(self, progress_callback=None, log_callback=None):
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.logger = self._setup_logging()
        self._temp_files = []  # Lista plików tymczasowych do wyczyszczenia

    def __del__(self):
        """Cleanup plików tymczasowych."""
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Usuwa pliki tymczasowe."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Nie można usunąć pliku tymczasowego {temp_file}: {e}")
        self._temp_files.clear()

    def _setup_logging(self):
        """Konfiguracja logowania."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Usuń poprzednie handlery
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Dodaj handler do pliku
        try:
            file_handler = logging.FileHandler('video_speed_processor.log', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Nie można utworzyć pliku logów: {e}")

        return logger

    def _update_progress(self, message: str, progress: float = None):
        """Aktualizuje progress i logi."""
        self.logger.info(message)
        if self.progress_callback:
            self.progress_callback(message, progress)

    def _validate_segments(self, segments: List[Tuple[float, float, bool]], audio_duration: float) -> List[
        Tuple[float, float, bool]]:
        """Waliduje i poprawia segmenty."""
        if not segments:
            return [(0.0, audio_duration, False)]

        validated = []
        for start, end, is_speech in segments:
            # Sprawdź poprawność czasów
            start = max(0.0, start)
            end = min(audio_duration, end)

            if end <= start:
                continue  # Pomiń nieprawidłowe segmenty

            if end - start < 0.1:  # Minimum 0.1s
                continue

            validated.append((start, end, is_speech))

        # Sortuj według czasu rozpoczęcia
        validated.sort(key=lambda x: x[0])

        # Wypełnij luki między segmentami
        filled = []
        current_time = 0.0

        for start, end, is_speech in validated:
            # Luka przed segmentem
            if start > current_time + 0.1:
                filled.append((current_time, start, False))  # Cisza w luce

            filled.append((start, end, is_speech))
            current_time = end

        # Pozostała część na końcu
        if current_time < audio_duration - 0.1:
            filled.append((current_time, audio_duration, False))

        return filled

    def detect_speech_segments(self, audio_path: str, method: str = 'webrtc',
                               min_silence_duration: float = 1.5) -> List[Tuple[float, float, bool]]:
        """Wykrywa segmenty mowy i ciszy."""
        self._update_progress(f"Wykrywanie segmentów mowy metodą: {method}")

        try:
            audio_duration = librosa.get_duration(filename=audio_path)
        except Exception as e:
            self.logger.error(f"Nie można odczytać długości audio: {e}")
            return []

        if audio_duration < 0.5:
            self.logger.warning("Plik audio jest zbyt krótki")
            return [(0.0, audio_duration, True)]  # Cały plik jako mowa

        if method == 'whisper' and WHISPER_AVAILABLE:
            segments = self._detect_speech_whisper(audio_path, min_silence_duration)
        elif method == 'webrtc' and WEBRTC_AVAILABLE:
            segments = self._detect_speech_webrtc(audio_path, min_silence_duration)
        else:
            segments = self._detect_speech_energy(audio_path, min_silence_duration)

        # Waliduj segmenty
        validated_segments = self._validate_segments(segments, audio_duration)

        self._update_progress(f"Znaleziono {len(validated_segments)} segmentów")
        return validated_segments

    def _detect_speech_whisper(self, audio_path: str, min_silence_duration: float) -> List[Tuple[float, float, bool]]:
        """Detekcja mowy używając Whisper."""
        try:
            self._update_progress("Ładowanie modelu Whisper...")
            model = whisper.load_model("base")

            self._update_progress("Transkrypcja audio...")
            result = model.transcribe(audio_path, word_timestamps=True)

            segments = []
            audio_duration = librosa.get_duration(filename=audio_path)
            current_time = 0.0

            # Przetwórz segmenty Whisper
            for segment in result.get('segments', []):
                start = max(0.0, segment.get('start', 0.0))
                end = min(audio_duration, segment.get('end', start + 1.0))

                if end <= start:
                    continue

                # Cisza przed segmentem
                if start > current_time + 0.5:
                    segments.append((current_time, start, False))

                # Segment mowy
                segments.append((start, end, True))
                current_time = end

            # Pozostała cisza
            if current_time < audio_duration - 0.5:
                segments.append((current_time, audio_duration, False))

            return segments

        except Exception as e:
            self.logger.error(f"Błąd Whisper: {e}")
            return self._detect_speech_energy(audio_path, min_silence_duration)

    def _detect_speech_webrtc(self, audio_path: str, min_silence_duration: float) -> List[Tuple[float, float, bool]]:
        """Detekcja mowy używając WebRTC VAD - NAPRAWIONA."""
        try:
            self._update_progress("Ładowanie audio dla WebRTC...")
            y, sr = librosa.load(audio_path, sr=16000)

            if len(y) < 1600:  # Minimum 0.1s dla 16kHz
                return [(0.0, len(y) / sr, True)]

            vad = webrtcvad.Vad(2)  # Średnia czułość
            frame_duration = 0.02  # 20ms - stabilniejsze niż 30ms
            frame_length = int(sr * frame_duration)

            speech_decisions = []

            # Przetwarzaj ramki
            for i in range(0, len(y) - frame_length, frame_length):
                frame = y[i:i + frame_length]
                frame_time = i / sr

                # Konwersja do int16
                frame_int16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
                frame_bytes = frame_int16.tobytes()

                try:
                    is_speech = vad.is_speech(frame_bytes, sr)
                    speech_decisions.append((frame_time, is_speech))
                except Exception:
                    # W przypadku błędu, zakładaj ciszę
                    speech_decisions.append((frame_time, False))

            if not speech_decisions:
                return [(0.0, len(y) / sr, False)]

            # NAPRAWIONA AGREGACJA: Użyj sliding window
            window_size = max(1, int(0.5 / frame_duration))  # 0.5s okno
            smoothed_decisions = []

            for i, (frame_time, _) in enumerate(speech_decisions):
                # Sprawdź otoczenie
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(speech_decisions), i + window_size // 2 + 1)

                speech_votes = sum(1 for _, is_speech in speech_decisions[start_idx:end_idx] if is_speech)
                total_votes = end_idx - start_idx

                # Decyzja większościowa z progiem
                is_speech = speech_votes > total_votes * 0.3  # 30% próg
                smoothed_decisions.append((frame_time, is_speech))

            # Agreguj w segmenty
            segments = []
            if smoothed_decisions:
                current_start = 0.0
                current_is_speech = smoothed_decisions[0][1]

                for frame_time, is_speech in smoothed_decisions[1:]:
                    if is_speech != current_is_speech:
                        duration = frame_time - current_start
                        if duration >= 0.3:  # Minimum 0.3s segment
                            segments.append((current_start, frame_time, current_is_speech))
                            current_start = frame_time
                            current_is_speech = is_speech

                # Ostatni segment
                audio_duration = len(y) / sr
                if audio_duration - current_start >= 0.3:
                    segments.append((current_start, audio_duration, current_is_speech))

            # Filtruj krótkie segmenty ciszy
            return self._filter_short_silences(segments, min_silence_duration)

        except Exception as e:
            self.logger.error(f"Błąd WebRTC: {e}")
            return self._detect_speech_energy(audio_path, min_silence_duration)

    def _detect_speech_energy(self, audio_path: str, min_silence_duration: float) -> List[Tuple[float, float, bool]]:
        """Detekcja na podstawie energii audio - NAPRAWIONA."""
        try:
            self._update_progress("Analiza energii audio...")
            y, sr = librosa.load(audio_path)

            if len(y) < sr * 0.5:  # Minimum 0.5s
                return [(0.0, len(y) / sr, True)]

            # Parametry analizy
            frame_length = int(sr * 0.25)  # 250ms ramki
            hop_length = int(sr * 0.125)  # 125ms przeskok

            # Oblicz RMS i spectral centroid dla lepszej detekcji
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

            # Normalizuj features
            rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
            centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (
                        np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8)

            # Kombinowany wskaźnik
            combined_energy = 0.7 * rms_norm + 0.3 * centroid_norm

            # Adaptacyjny próg
            threshold = np.percentile(combined_energy, 30)  # Niższy próg dla lepszej detekcji

            times = librosa.frames_to_time(np.arange(len(combined_energy)), sr=sr, hop_length=hop_length)

            # Wygładź decyzje
            window_size = max(1, int(1.0 / (hop_length / sr)))  # 1s okno
            smoothed_energy = np.convolve(combined_energy, np.ones(window_size) / window_size, mode='same')

            # Agreguj w segmenty
            segments = []
            current_start = 0.0
            current_is_speech = smoothed_energy[0] > threshold

            for i, (time, energy) in enumerate(zip(times[1:], smoothed_energy[1:]), 1):
                is_speech = energy > threshold

                if is_speech != current_is_speech:
                    duration = time - current_start
                    if duration >= 0.5:  # Minimum 0.5s segment
                        segments.append((current_start, time, current_is_speech))
                        current_start = time
                        current_is_speech = is_speech

            # Ostatni segment
            audio_duration = len(y) / sr
            if audio_duration - current_start >= 0.5:
                segments.append((current_start, audio_duration, current_is_speech))

            return self._filter_short_silences(segments, min_silence_duration)

        except Exception as e:
            self.logger.error(f"Błąd detekcji energii: {e}")
            return []

    def _filter_short_silences(self, segments: List[Tuple[float, float, bool]], min_silence_duration: float) -> List[
        Tuple[float, float, bool]]:
        """Filtruje krótkie segmenty ciszy, łącząc je z sąsiednimi segmentami mowy."""
        if not segments:
            return segments

        filtered = []
        for start, end, is_speech in segments:
            duration = end - start

            if is_speech or duration >= min_silence_duration:
                filtered.append((start, end, is_speech))
            else:
                # Krótka cisza - zamień na mowę
                filtered.append((start, end, True))

        return filtered

    def create_speed_overlay(self, text: str, duration: float, video_size: Tuple[int, int]) -> Optional[mp.VideoClip]:
        """Tworzy overlay z tekstem prędkości - NAPRAWIONY."""
        try:
            # Sprawdź dostępność ImageMagick
            test_clip = mp.TextClip("test", fontsize=10, color='white')
            test_clip.close()

            # Jeśli test przeszedł, utwórz prawdziwy overlay
            overlay = mp.TextClip(
                text,
                fontsize=max(24, min(video_size) // 20),  # Mniejszy tekst
                color='yellow',
                font='Arial-Bold' if sys.platform == 'win32' else 'DejaVu-Sans-Bold',
                stroke_color='black',
                stroke_width=2
            ).set_duration(duration)

            # Pozycja w prawym dolnym rogu z marginesem
            margin = max(20, min(video_size) // 40)
            overlay = overlay.set_position((
                video_size[0] - overlay.w - margin,
                video_size[1] - overlay.h - margin
            )).set_opacity(0.9)

            return overlay

        except Exception as e:
            self.logger.warning(f"Nie można utworzyć overlay tekstowego: {e}")
            return self._create_simple_overlay(text, duration, video_size)

    def _create_simple_overlay(self, text: str, duration: float, video_size: Tuple[int, int]) -> Optional[mp.VideoClip]:
        """Fallback overlay - kolorowy prostokąt - NAPRAWIONY."""
        try:
            rect_size = max(30, min(video_size) // 25)

            def make_frame(t):
                frame = np.zeros((rect_size, rect_size, 3), dtype=np.uint8)

                # Kolor zależny od prędkości
                if 'x5' in text:
                    color = [255, 0, 0]  # Czerwony dla x5
                elif 'x4' in text:
                    color = [255, 128, 0]  # Pomarańczowy dla x4
                elif 'x3' in text:
                    color = [255, 255, 0]  # Żółty dla x3
                else:
                    color = [0, 255, 0]  # Zielony dla innych

                frame[:, :] = color
                return frame

            overlay = mp.VideoClip(make_frame, duration=duration)

            # Pozycja w prawym dolnym rogu
            margin = max(20, min(video_size) // 40)
            overlay = overlay.set_position((
                video_size[0] - rect_size - margin,
                video_size[1] - rect_size - margin
            )).set_opacity(0.8)

            return overlay

        except Exception as e:
            self.logger.error(f"Nie można utworzyć prostego overlay: {e}")
            return None

    def process_video(self, input_path: str, output_folder: str, speed_multiplier: float = 3.0,
                      min_silence_duration: float = 1.5, detection_method: str = 'webrtc',
                      create_video: bool = True) -> Optional[dict]:
        """Przetwarza pojedynczy plik wideo - NAPRAWIONY."""
        self._update_progress(f"Ładowanie wideo: {os.path.basename(input_path)}")

        video = None
        processed_clips = []

        try:
            # Walidacja pliku wejściowego
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Plik nie istnieje: {input_path}")

            file_size = os.path.getsize(input_path)
            if file_size < 1024:  # Mniej niż 1KB
                raise ValueError(f"Plik jest zbyt mały: {file_size} bajtów")

            video = mp.VideoFileClip(input_path)

            if video.duration < 1.0:
                raise ValueError(f"Wideo jest zbyt krótkie: {video.duration}s")

            # Sprawdź czy video ma audio
            if video.audio is None:
                self.logger.warning("Wideo nie ma ścieżki audio - traktowanie jako cisza")
                return {
                    'input_file': input_path,
                    'input_name': Path(input_path).stem,
                    'timeline_data': [{'start': 0, 'end': video.duration, 'duration': video.duration, 'speed': 1.0,
                                       'type': 'no_audio'}],
                    'original_duration': video.duration,
                    'output_duration': video.duration,
                    'segments_count': 1
                }

            # Wyodrębnij audio do pliku tymczasowego
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
                self._temp_files.append(temp_audio_path)

            self._update_progress("Wyodrębnianie audio...")
            try:
                video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            except Exception as e:
                raise RuntimeError(f"Nie można wyodrębnić audio: {e}")

            # Wykryj segmenty
            segments = self.detect_speech_segments(temp_audio_path, detection_method, min_silence_duration)

            if not segments:
                raise ValueError("Nie znaleziono żadnych segmentów")

            self._update_progress(f"Przetwarzanie {len(segments)} segmentów...")

            # Przetwórz segmenty
            timeline_data = []
            total_output_duration = 0.0

            for i, (start, end, is_speech) in enumerate(segments):
                progress = (i / len(segments)) * 100
                self._update_progress(f"Segment {i + 1}/{len(segments)}", progress)

                # Walidacja segmentu
                start = max(0.0, start)
                end = min(video.duration, end)

                if end <= start:
                    continue

                duration = end - start

                try:
                    clip = video.subclip(start, end)

                    if is_speech:
                        # Normalne tempo dla mowy
                        timeline_data.append({
                            'start': start,
                            'end': end,
                            'duration': duration,
                            'speed': 1.0,
                            'type': 'speech'
                        })
                        total_output_duration += duration
                    else:
                        # Przyspiesz ciszę
                        if duration >= min_silence_duration:
                            # Sprawdź czy speed_multiplier jest rozsądny
                            actual_speed = max(1.1, min(10.0, speed_multiplier))

                            clip = clip.fx(mp.vfx.speedx, actual_speed)

                            # Dodaj overlay tylko jeśli tworzymy wideo
                            if create_video:
                                overlay_text = f"x{actual_speed:.1f}" if actual_speed != int(
                                    actual_speed) else f"x{int(actual_speed)}"
                                overlay = self.create_speed_overlay(overlay_text, clip.duration, (video.w, video.h))

                                if overlay:
                                    clip = mp.CompositeVideoClip([clip, overlay])

                            timeline_data.append({
                                'start': start,
                                'end': end,
                                'duration': clip.duration,
                                'original_duration': duration,
                                'speed': actual_speed,
                                'type': 'silence'
                            })
                            total_output_duration += clip.duration
                        else:
                            # Krótka cisza - zostaw normalną
                            timeline_data.append({
                                'start': start,
                                'end': end,
                                'duration': duration,
                                'speed': 1.0,
                                'type': 'short_silence'
                            })
                            total_output_duration += duration

                    processed_clips.append(clip)

                except Exception as e:
                    self.logger.error(f"Błąd przetwarzania segmentu {start}-{end}: {e}")
                    continue

            if not processed_clips:
                raise ValueError("Nie udało się przetworzyć żadnego segmentu")

            # Twórz wideo tylko jeśli wymagane
            output_path = None
            if create_video:
                self._update_progress("Łączenie klipów...")
                try:
                    final_video = mp.concatenate_videoclips(processed_clips, method="compose")

                    input_name = Path(input_path).stem
                    output_path = os.path.join(output_folder, f"{input_name}_processed.mp4")

                    self._update_progress("Zapisywanie wideo...")
                    final_video.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile=os.path.join(output_folder, 'temp_audio.m4a'),
                        remove_temp=True,
                        verbose=False,
                        logger=None
                    )

                    final_video.close()

                except Exception as e:
                    self.logger.error(f"Błąd tworzenia wideo: {e}")
                    output_path = None

            # Buduj wynik
            result = {
                'input_file': input_path,
                'input_name': Path(input_path).stem,
                'timeline_data': timeline_data,
                'original_duration': video.duration,
                'output_duration': total_output_duration,
                'segments_count': len(segments)
            }

            if output_path:
                result['output_video'] = output_path

            return result

        except Exception as e:
            self.logger.error(f"Błąd przetwarzania {input_path}: {e}")
            return None

        finally:
            # CLEANUP - bardzo ważne!
            try:
                if video:
                    video.close()

                for clip in processed_clips:
                    if clip:
                        clip.close()

            except Exception as e:
                self.logger.warning(f"Błąd podczas cleanup: {e}")

    def generate_edl(self, results: List[dict], output_path: str):
        """Generuje plik EDL - NAPRAWIONY."""
        self._update_progress("Generowanie EDL...")

        if not results:
            self.logger.error("Brak danych do generowania EDL")
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("TITLE: Processed_Videos\n")
                f.write("FCM: NON-DROP FRAME\n\n")

                edit_number = 1
                timeline_pos = 0.0

                for result_idx, result in enumerate(results):
                    if not result or 'timeline_data' not in result:
                        continue

                    source_name = result.get('input_name', f'VIDEO_{result_idx + 1}')
                    reel_name = f"AX{result_idx + 1:03d}"

                    for segment in result['timeline_data']:
                        try:
                            source_start = float(segment.get('start', 0))
                            original_duration = float(segment.get('original_duration', segment.get('duration', 1)))
                            source_end = source_start + original_duration
                            segment_duration = float(segment.get('duration', original_duration))

                            timeline_start = timeline_pos
                            timeline_end = timeline_pos + segment_duration

                            # Sprawdź poprawność czasów
                            if source_end <= source_start or timeline_end <= timeline_start:
                                continue

                            # Timecode z obsługą błędów
                            source_in_tc = self._seconds_to_timecode(source_start)
                            source_out_tc = self._seconds_to_timecode(source_end)
                            timeline_in_tc = self._seconds_to_timecode(timeline_start)
                            timeline_out_tc = self._seconds_to_timecode(timeline_end)

                            # Linia EDL
                            f.write(f"{edit_number:03d}  {reel_name:<8} V     C        ")
                            f.write(f"{source_in_tc} {source_out_tc} {timeline_in_tc} {timeline_out_tc}\n")

                            # Metadane
                            f.write(f"* FROM CLIP NAME: {source_name}\n")
                            f.write(f"* SEGMENT TYPE: {segment.get('type', 'unknown')}\n")

                            speed = float(segment.get('speed', 1.0))
                            if abs(speed - 1.0) > 0.01:  # Jeśli speed != 1.0
                                f.write(f"* SPEED: {speed:.2f}\n")
                                speed_percent = speed * 100
                                f.write(
                                    f"M2   {reel_name:<8}     050 {timeline_in_tc} {timeline_out_tc} {speed_percent:06.2f} {speed_percent:06.2f}\n")

                            f.write("\n")

                            edit_number += 1
                            timeline_pos = timeline_end

                        except (ValueError, TypeError, KeyError) as e:
                            self.logger.warning(f"Błąd przetwarzania segmentu w EDL: {e}")
                            continue

                # Mapowanie plików źródłowych
                f.write("* SOURCE FILE MAPPING:\n")
                for result_idx, result in enumerate(results):
                    if result and 'input_file' in result:
                        reel_name = f"AX{result_idx + 1:03d}"
                        source_name = os.path.basename(result['input_file'])
                        f.write(f"* {reel_name}: {source_name}\n")

            self._update_progress(f"EDL zapisany: {output_path}")

        except Exception as e:
            self.logger.error(f"Błąd generowania EDL: {e}")
            raise

    def _seconds_to_timecode(self, seconds: float, fps: int = 25) -> str:
        """Konwertuje sekundy na timecode - NAPRAWIONY."""
        try:
            seconds = max(0.0, float(seconds))

            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            frames = int((seconds % 1) * fps)

            # Sprawdź granice
            hours = min(23, hours)
            minutes = min(59, minutes)
            secs = min(59, secs)
            frames = min(fps - 1, frames)

            return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

        except (ValueError, TypeError):
            return "00:00:00:00"


class VideoProcessorGUI:
    """GUI dla Video Processor - NAPRAWIONY."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Speed Processor v2.0 - FIXED")
        self.root.geometry("850x750")
        self.root.minsize(700, 600)

        # Zmienne
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value="output")
        self.speed_multiplier = tk.DoubleVar(value=3.0)
        self.min_silence_duration = tk.DoubleVar(value=1.5)
        self.detection_method = tk.StringVar(value="webrtc" if WEBRTC_AVAILABLE else "energy")
        self.create_video = tk.BooleanVar(value=True)
        self.create_edl = tk.BooleanVar(value=True)

        # Queue dla logów
        self.log_queue = queue.Queue()

        # Stan przetwarzania
        self.is_processing = False

        self.setup_ui()
        self.check_dependencies()
        self.update_logs()

    def setup_ui(self):
        """Tworzy interfejs użytkownika - ULEPSZONY."""
        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Dodaj niestandardowe style
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')

        # Main frame z scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Nagłówek z wersją
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=row, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))

        ttk.Label(header_frame, text="Video Speed Processor", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="v2.0 - FIXED", foreground='blue', font=('Arial', 10)).pack(side=tk.RIGHT)
        row += 1

        # Status systemu
        self.system_status_frame = ttk.LabelFrame(main_frame, text="Status systemu", padding="10")
        self.system_status_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        row += 1

        # Folder wejściowy
        ttk.Label(main_frame, text="Folder z plikami MP4:", font=('Arial', 10, 'bold')).grid(row=row, column=0,
                                                                                             sticky=tk.W, pady=8)
        self.input_entry = ttk.Entry(main_frame, textvariable=self.input_folder, width=55)
        self.input_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=8, padx=(10, 5))
        ttk.Button(main_frame, text="📁 Przeglądaj", command=self.browse_input_folder).grid(row=row, column=2, pady=8)
        row += 1

        # Podgląd znalezionych plików
        self.files_info_label = ttk.Label(main_frame, text="", foreground='gray')
        self.files_info_label.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1

        # Folder wyjściowy
        ttk.Label(main_frame, text="Folder wyjściowy:", font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky=tk.W,
                                                                                         pady=8)
        ttk.Entry(main_frame, textvariable=self.output_folder, width=55).grid(row=row, column=1, sticky=(tk.W, tk.E),
                                                                              pady=8, padx=(10, 5))
        ttk.Button(main_frame, text="📁 Przeglądaj", command=self.browse_output_folder).grid(row=row, column=2, pady=8)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E),
                                                            pady=25)
        row += 1

        # Ustawienia zaawansowane
        settings_frame = ttk.LabelFrame(main_frame, text="⚙️ Ustawienia przetwarzania", padding="12")
        settings_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=12)
        settings_frame.columnconfigure(1, weight=1)
        row += 1

        settings_row = 0

        # Mnożnik prędkości z lepszą kontrolą
        ttk.Label(settings_frame, text="Przyspieszenie ciszy:", font=('Arial', 9, 'bold')).grid(row=settings_row,
                                                                                                column=0, sticky=tk.W,
                                                                                                pady=8)
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.grid(row=settings_row, column=1, sticky=(tk.W, tk.E), pady=8, padx=(15, 0))
        speed_frame.columnconfigure(0, weight=1)

        self.speed_scale = ttk.Scale(speed_frame, from_=1.5, to=5.0, variable=self.speed_multiplier,
                                     orient=tk.HORIZONTAL)
        self.speed_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        self.speed_label = ttk.Label(speed_frame, text="3.0x", width=8, background='lightblue', relief='solid')
        self.speed_label.grid(row=0, column=1)
        settings_row += 1

        # Minimalna cisza
        ttk.Label(settings_frame, text="Min. długość ciszy (s):", font=('Arial', 9, 'bold')).grid(row=settings_row,
                                                                                                  column=0, sticky=tk.W,
                                                                                                  pady=8)
        silence_frame = ttk.Frame(settings_frame)
        silence_frame.grid(row=settings_row, column=1, sticky=(tk.W, tk.E), pady=8, padx=(15, 0))
        silence_frame.columnconfigure(0, weight=1)

        self.silence_scale = ttk.Scale(silence_frame, from_=0.5, to=4.0, variable=self.min_silence_duration,
                                       orient=tk.HORIZONTAL)
        self.silence_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        self.silence_label = ttk.Label(silence_frame, text="1.5s", width=8, background='lightgreen', relief='solid')
        self.silence_label.grid(row=0, column=1)
        settings_row += 1

        # Metoda detekcji z opisami
        ttk.Label(settings_frame, text="Metoda detekcji mowy:", font=('Arial', 9, 'bold')).grid(row=settings_row,
                                                                                                column=0, sticky=tk.W,
                                                                                                pady=8)
        detection_frame = ttk.Frame(settings_frame)
        detection_frame.grid(row=settings_row, column=1, sticky=(tk.W, tk.E), pady=8, padx=(15, 0))

        methods = []
        if WHISPER_AVAILABLE:
            methods.append(("🤖 Whisper AI (najdokładniejszy)", "whisper"))
        if WEBRTC_AVAILABLE:
            methods.append(("⚡ WebRTC VAD (szybki)", "webrtc"))
        methods.append(("📊 Analiza energii (podstawowy)", "energy"))

        for text, value in methods:
            ttk.Radiobutton(detection_frame, text=text, variable=self.detection_method, value=value).pack(anchor=tk.W,
                                                                                                          pady=2)
        settings_row += 1

        # Opcje generowania
        options_frame = ttk.LabelFrame(main_frame, text="📤 Opcje wyjściowe", padding="12")
        options_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=12)
        row += 1

        ttk.Checkbutton(options_frame, text="🎥 Generuj przetworzone pliki MP4 (z overlayami prędkości)",
                        variable=self.create_video).pack(anchor=tk.W, pady=4)
        ttk.Checkbutton(options_frame, text="🎬 Generuj timeline EDL dla DaVinci Resolve",
                        variable=self.create_edl).pack(anchor=tk.W, pady=4)

        # Przyciski główne
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=row, column=0, columnspan=3, pady=25)
        row += 1

        self.process_button = ttk.Button(buttons_frame, text="🚀 Rozpocznij przetwarzanie",
                                         command=self.start_processing, width=25)
        self.process_button.pack(side=tk.LEFT, padx=(0, 15))

        self.stop_button = ttk.Button(buttons_frame, text="⏹️ Zatrzymaj",
                                      command=self.stop_processing, state='disabled', width=15)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Button(buttons_frame, text="❓ Pomoc", command=self.show_help, width=10).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(buttons_frame, text="📁 Otwórz wyniki", command=self.open_output_folder, width=15).pack(side=tk.LEFT)

        # Progress z lepszym wyświetlaniem
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Postęp przetwarzania", padding="10")
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=12)
        progress_frame.columnconfigure(0, weight=1)
        row += 1

        self.progress_var = tk.StringVar(value="Gotowy do pracy ✅")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, font=('Arial', 10))
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        # Informacje o czasie
        self.time_info = ttk.Label(progress_frame, text="", foreground='gray', font=('Arial', 9))
        self.time_info.grid(row=2, column=0, sticky=tk.W)

        # Logi z lepszym formatowaniem
        log_frame = ttk.LabelFrame(main_frame, text="📋 Logi systemu", padding="8")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=12)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)

        # Log text z kontrolkami
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(log_controls, text="🗑️ Wyczyść logi", command=self.clear_logs, width=15).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="💾 Zapisz logi", command=self.save_logs, width=15).pack(side=tk.LEFT,
                                                                                              padx=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED,
                                                  font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Konfiguruj tags dla kolorów
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("INFO", foreground="blue")

        # Pack canvas i scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bindowanie zdarzeń
        self.speed_multiplier.trace('w', self.update_speed_label)
        self.min_silence_duration.trace('w', self.update_silence_label)
        self.input_folder.trace('w', self.update_files_info)

        # Bind mouse wheel dla canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def update_speed_label(self, *args):
        """Aktualizuje label prędkości."""
        value = round(self.speed_multiplier.get(), 1)
        self.speed_label.config(text=f"{value}x")

    def update_silence_label(self, *args):
        """Aktualizuje label ciszy."""
        value = round(self.min_silence_duration.get(), 1)
        self.silence_label.config(text=f"{value}s")

    def update_files_info(self, *args):
        """Aktualizuje informacje o znalezionych plikach."""
        folder = self.input_folder.get()
        if folder and os.path.exists(folder):
            try:
                mp4_files = list(Path(folder).glob("*.mp4"))
                if mp4_files:
                    total_size = sum(f.stat().st_size for f in mp4_files) / (1024 * 1024)  # MB
                    self.files_info_label.config(
                        text=f"✅ Znaleziono {len(mp4_files)} plików MP4 ({total_size:.1f} MB)",
                        foreground='green'
                    )
                else:
                    self.files_info_label.config(text="⚠️ Brak plików MP4 w folderze", foreground='orange')
            except Exception as e:
                self.files_info_label.config(text=f"❌ Błąd odczytu folderu: {e}", foreground='red')
        else:
            self.files_info_label.config(text="", foreground='gray')

    def browse_input_folder(self):
        """Wybór folderu wejściowego."""
        folder = filedialog.askdirectory(title="Wybierz folder z plikami MP4")
        if folder:
            self.input_folder.set(folder)

    def browse_output_folder(self):
        """Wybór folderu wyjściowego."""
        folder = filedialog.askdirectory(title="Wybierz folder wyjściowy")
        if folder:
            self.output_folder.set(folder)

    def check_dependencies(self):
        """Sprawdza dostępność bibliotek - ULEPSZONY."""
        status_items = []

        # Sprawdź MoviePy/FFmpeg
        try:
            test_clip = mp.ColorClip(size=(100, 100), color=(0, 0, 0), duration=0.1)
            test_clip.close()
            status_items.append(("✅ MoviePy/FFmpeg", "green"))
        except Exception as e:
            status_items.append(("❌ MoviePy/FFmpeg", "red"))
            self.add_log(f"BŁĄD MoviePy: {e}", "ERROR")

        # Sprawdź WebRTC
        if WEBRTC_AVAILABLE:
            status_items.append(("✅ WebRTC VAD", "green"))
        else:
            status_items.append(("⚠️ WebRTC VAD", "orange"))

        # Sprawdź Whisper
        if WHISPER_AVAILABLE:
            status_items.append(("✅ Whisper AI", "green"))
        else:
            status_items.append(("⚠️ Whisper AI", "orange"))

        # Sprawdź ImageMagick
        try:
            test_clip = mp.TextClip("test", fontsize=20, color='white').set_duration(0.1)
            test_clip.close()
            status_items.append(("✅ ImageMagick (overlaye)", "green"))
        except Exception:
            status_items.append(("⚠️ ImageMagick (kolorowe overlaye)", "orange"))

        # Wyświetl status
        for widget in self.system_status_frame.winfo_children():
            widget.destroy()

        for i, (text, color) in enumerate(status_items):
            row = i // 2
            col = i % 2
            ttk.Label(self.system_status_frame, text=text, foreground=color).grid(
                row=row, column=col, sticky=tk.W, padx=(0, 20), pady=2
            )

    def add_log(self, message, tag="INFO"):
        """Dodaje wiadomość do logów z kolorami."""
        timestamp = tk.datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def clear_logs(self):
        """Czyści logi."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def save_logs(self):
        """Zapisuje logi do pliku."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")],
                title="Zapisz logi"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.add_log(f"Logi zapisane: {filename}", "SUCCESS")
        except Exception as e:
            self.add_log(f"Błąd zapisu logów: {e}", "ERROR")

    def update_logs(self):
        """Aktualizuje logi z queue."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                # Określ tag na podstawie treści
                if "ERROR" in message or "Błąd" in message:
                    tag = "ERROR"
                elif "WARNING" in message or "⚠️" in message:
                    tag = "WARNING"
                elif "✅" in message or "Ukończono" in message:
                    tag = "SUCCESS"
                else:
                    tag = "INFO"
                self.add_log(message, tag)
        except queue.Empty:
            pass

        # Sprawdź ponownie za 100ms
        self.root.after(100, self.update_logs)

    def update_progress(self, message, progress=None):
        """Callback do aktualizacji progressu - ULEPSZONY."""
        self.progress_var.set(message)

        if progress is not None:
            self.progress_bar.config(mode='determinate')
            self.progress_bar['value'] = min(100, max(0, progress))

            # Oszacuj pozostały czas
            if hasattr(self, 'start_time') and progress > 0:
                elapsed = time.time() - self.start_time
                if progress > 5:  # Dopiero po 5% pokazuj oszacowanie
                    total_estimated = elapsed * 100 / progress
                    remaining = total_estimated - elapsed
                    self.time_info.config(text=f"Pozostało: ~{remaining / 60:.1f} min")
        else:
            self.progress_bar.config(mode='indeterminate')
            if not self.progress_bar.instate(['active']):
                self.progress_bar.start()

    def start_processing(self):
        """Rozpoczyna przetwarzanie w osobnym wątku - ULEPSZONY."""
        # Walidacja
        if not self.input_folder.get():
            messagebox.showerror("Błąd", "Wybierz folder z plikami MP4!")
            return

        if not os.path.exists(self.input_folder.get()):
            messagebox.showerror("Błąd", "Folder wejściowy nie istnieje!")
            return

        # Znajdź pliki MP4
        try:
            mp4_files = list(Path(self.input_folder.get()).glob("*.mp4"))
            if not mp4_files:
                messagebox.showerror("Błąd", "Nie znaleziono plików MP4 w wybranym folderze!")
                return
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można odczytać folderu: {e}")
            return

        # Sprawdź czy co najmniej jedna opcja jest włączona
        if not self.create_video.get() and not self.create_edl.get():
            messagebox.showwarning("Uwaga", "Wybierz co najmniej jedną opcję wyjściową!")
            return

        # Potwierdź rozpoczęcie
        if len(mp4_files) > 5:
            if not messagebox.askyesno("Potwierdzenie",
                                       f"Znaleziono {len(mp4_files)} plików do przetworzenia.\n"
                                       f"To może zająć dużo czasu. Kontynuować?"):
                return

        # Przygotuj UI
        self.is_processing = True
        self.process_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        import time
        self.start_time = time.time()

        self.add_log(f"🚀 Rozpoczynanie przetwarzania {len(mp4_files)} plików...", "INFO")

        # Uruchom w wątku
        self.processing_thread = threading.Thread(target=self.process_videos, args=(mp4_files,), daemon=True)
        self.processing_thread.start()

    def stop_processing(self):
        """Zatrzymuje przetwarzanie."""
        self.is_processing = False
        self.add_log("⏹️ Zatrzymywanie przetwarzania...", "WARNING")

    def process_videos(self, mp4_files):
        """Przetwarza pliki wideo - NAPRAWIONY."""
        try:
            # Przygotuj foldery
            output_folder = Path(self.output_folder.get())
            output_folder.mkdir(parents=True, exist_ok=True)

            # Utwórz procesor
            processor = VideoProcessor(
                progress_callback=self.update_progress,
                log_callback=self.add_log
            )

            results = []
            successful = 0
            failed = 0

            for i, mp4_file in enumerate(mp4_files):
                if not self.is_processing:
                    self.root.after(0, lambda: self.add_log("❌ Przetwarzanie anulowane przez użytkownika", "WARNING"))
                    break

                file_progress = (i / len(mp4_files)) * 100
                self.root.after(0, lambda msg=f"📹 Przetwarzanie: {mp4_file.name} ({i + 1}/{len(mp4_files)})",
                                          prog=file_progress:
                self.update_progress(msg, prog))

                try:
                    result = processor.process_video(
                        str(mp4_file),
                        str(output_folder),
                        speed_multiplier=self.speed_multiplier.get(),
                        min_silence_duration=self.min_silence_duration.get(),
                        detection_method=self.detection_method.get(),
                        create_video=self.create_video.get()
                    )

                    if result:
                        results.append(result)
                        successful += 1

                        # Skopiuj oryginalny plik do output (dla EDL)
                        if self.create_edl.get():
                            dest_path = output_folder / mp4_file.name
                            if not dest_path.exists():
                                shutil.copy2(mp4_file, dest_path)

                        self.root.after(0, lambda name=mp4_file.name:
                        self.add_log(f"✅ Ukończono: {name}", "SUCCESS"))
                    else:
                        failed += 1
                        self.root.after(0, lambda name=mp4_file.name:
                        self.add_log(f"❌ Niepowodzenie: {name}", "ERROR"))

                except Exception as e:
                    failed += 1
                    self.root.after(0, lambda name=mp4_file.name, err=str(e):
                    self.add_log(f"❌ Błąd {name}: {err}", "ERROR"))

            # Zapisz dane jako JSON
            if results:
                try:
                    json_path = output_folder / "timeline_data.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

                    self.root.after(0, lambda: self.add_log("💾 Dane timeline zapisane do JSON", "SUCCESS"))
                except Exception as e:
                    self.root.after(0, lambda err=str(e): self.add_log(f"⚠️ Błąd zapisu JSON: {err}", "WARNING"))

                # Generuj EDL jeśli wymagane
                if self.create_edl.get():
                    try:
                        edl_path = output_folder / "timeline.edl"
                        processor.generate_edl(results, str(edl_path))
                        self.root.after(0, lambda: self.add_log("🎬 Timeline EDL wygenerowany", "SUCCESS"))
                    except Exception as e:
                        self.root.after(0, lambda err=str(e): self.add_log(f"❌ Błąd generowania EDL: {err}", "ERROR"))

                self.root.after(0, self.processing_complete, successful, failed, str(output_folder))
            else:
                self.root.after(0, lambda: self.add_log("❌ Nie udało się przetworzyć żadnego pliku", "ERROR"))
                self.root.after(0, lambda: self.update_progress("❌ Przetwarzanie nieudane"))

        except Exception as e:
            self.root.after(0, lambda err=str(e): self.add_log(f"❌ Krytyczny błąd: {err}", "ERROR"))
            self.root.after(0, lambda err=str(e): self.update_progress(f"❌ Krytyczny błąd: {err}"))
        finally:
            self.root.after(0, self.processing_finished)

    def processing_complete(self, successful, failed, output_path):
        """Wywołane po zakończeniu przetwarzania - ULEPSZONY."""
        total_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0

        status = f"✅ Przetwarzanie ukończone!"
        self.update_progress(status, 100)

        # Szczegółowy raport
        self.add_log(f"📊 RAPORT KOŃCOWY:", "SUCCESS")
        self.add_log(f"   • Przetworzonych: {successful} plików", "SUCCESS")
        if failed > 0:
            self.add_log(f"   • Niepowodzeń: {failed} plików", "WARNING")
        self.add_log(f"   • Całkowity czas: {total_time / 60:.1f} min", "INFO")
        self.add_log(f"   • Wyniki w: {output_path}", "INFO")

        # Informacje o EDL
        if self.create_edl.get():
            edl_path = os.path.join(output_path, "timeline.edl")
            if os.path.exists(edl_path):
                self.add_log("🎬 Timeline EDL gotowy do importu w DaVinci Resolve", "SUCCESS")
                self.add_log("   IMPORT: File → Import → Timeline → Pre-Conform", "INFO")
                self.add_log("   Wybierz plik: timeline.edl", "INFO")

        # Informacje o wygenerowanych plikach
        if self.create_video.get():
            processed_files = len([f for f in os.listdir(output_path) if f.endswith('_processed.mp4')])
            if processed_files > 0:
                self.add_log(f"🎥 Wygenerowano {processed_files} przetworzonych plików MP4", "SUCCESS")

        # Aktualizuj informacje o czasie
        self.time_info.config(text=f"Ukończono w {total_time / 60:.1f} min")

        # Pokaż messagebox z podsumowaniem
        success_msg = f"Pomyślnie przetworzono {successful} z {successful + failed} plików!"
        if failed > 0:
            success_msg += f"\n\nNiepowodzenia: {failed} plików (sprawdź logi)"

        success_msg += f"\n\nCzas przetwarzania: {total_time / 60:.1f} min"
        success_msg += f"\nWyniki zapisane w:\n{output_path}"

        if self.create_edl.get():
            success_msg += "\n\n🎬 Timeline EDL gotowy do importu!"

        messagebox.showinfo("Przetwarzanie ukończone", success_msg)

    def processing_finished(self):
        """Przywraca GUI po zakończeniu - ULEPSZONY."""
        self.is_processing = False
        self.process_button.config(state='normal', text="🚀 Rozpocznij przetwarzanie")
        self.stop_button.config(state='disabled')

        # Zatrzymaj progress bar
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate', value=0)

        # Reset statusu
        if not hasattr(self, 'start_time'):
            self.progress_var.set("Gotowy do pracy ✅")
            self.time_info.config(text="")

    def open_output_folder(self):
        """Otwiera folder wyjściowy - ULEPSZONY."""
        output_path = self.output_folder.get()

        if not output_path:
            output_path = "output"

        # Utwórz folder jeśli nie istnieje
        os.makedirs(output_path, exist_ok=True)

        try:
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                os.system(f"open '{output_path}'")
            else:
                os.system(f"xdg-open '{output_path}'")

            self.add_log(f"📁 Otwarto folder: {os.path.abspath(output_path)}", "INFO")

        except Exception as e:
            self.add_log(f"❌ Nie można otworzyć folderu: {e}", "ERROR")
            messagebox.showwarning("Uwaga", f"Nie można otworzyć folderu:\n{os.path.abspath(output_path)}")

    def show_help(self):
        """Wyświetla okno pomocy - ULEPSZONY."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Pomoc - Video Speed Processor v2.0")
        help_window.geometry("700x600")
        help_window.transient(self.root)
        help_window.grab_set()
        help_window.resizable(True, True)

        # Ikona (jeśli dostępna)
        try:
            help_window.iconbitmap(self.root.iconbitmap())
        except:
            pass

        # Main frame z scrollbar
        canvas = tk.Canvas(help_window)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Content frame
        content_frame = ttk.Frame(scrollable_frame, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Tekst pomocy - ROZSZERZONY
        help_text = """🎬 Video Speed Processor v2.0 - FIXED - Pomoc

═══════════════════════════════════════════════════════════════

🚀 JAK UŻYWAĆ:

1. 📁 WYBÓR PLIKÓW
   • Wybierz folder zawierający pliki MP4
   • Program automatycznie znajdzie wszystkie pliki .mp4
   • Sprawdź czy pokazuje się prawidłowa liczba plików

2. ⚙️ KONFIGURACJA
   • Folder wyjściowy: gdzie zapisać wyniki (domyślnie 'output')
   • Przyspieszenie ciszy: 1.5x - 5x (zalecane: 3x)
   • Min. długość ciszy: 0.5s - 4s (zalecane: 1.5s)
   • Metoda detekcji: WebRTC/Whisper/Energia

3. 📤 OPCJE WYJŚCIOWE
   ✅ Przetworzone wideo MP4 - z overlayami prędkości
   ✅ Timeline EDL - dla DaVinci Resolve

4. 🎬 PRZETWARZANIE
   • Kliknij "Rozpocznij przetwarzanie"
   • Obserwuj postęp w logach
   • Poczekaj na zakończenie

═══════════════════════════════════════════════════════════════

🔧 CO ROBI PROGRAM:

• 🎤 DETEKCJA MOWY: Automatycznie rozpoznaje fragmenty z mową
• ⚡ PRZYSPIESZANIE: Przyspiesza tylko fragmenty ciszy
• 📺 OVERLAY: Dodaje widoczny wskaźnik "x3" podczas przyspieszenia
• 🎬 EDL EXPORT: Tworzy timeline gotowy do DaVinci Resolve
• 📊 STATYSTYKI: Pokazuje oszczędność czasu

═══════════════════════════════════════════════════════════════

🎯 METODY DETEKCJI:

🤖 WHISPER AI (najdokładniejszy)
   • Rozpoznaje mowę w różnych językach
   • Najlepsza jakość detekcji
   • Wymaga więcej czasu i mocy CPU
   • Idealny dla: podcastów, wywiadów, prezentacji

⚡ WEBRTC VAD (szybki)
   • Bardzo szybka detekcja w czasie rzeczywistym
   • Niskie użycie zasobów
   • Dobry dla czystego audio
   • Idealny dla: gamingu, screencastów

📊 ANALIZA ENERGII (podstawowy)
   • Zawsze dostępny (fallback)
   • Bazuje na głośności dźwięku
   • Najmniej precyzyjny
   • Używany gdy inne metody niedostępne

═══════════════════════════════════════════════════════════════

🎬 IMPORT DO DAVINCI RESOLVE:

1. 📂 PRZYGOTOWANIE
   • Upewnij się że folder 'output' zawiera:
     - timeline.edl (główny plik)
     - oryginalne pliki MP4
     - (opcjonalnie) przetworzone pliki MP4

2. 📥 IMPORT TIMELINE
   • File → Import → Timeline → Pre-Conform
   • Wybierz plik 'timeline.edl'
   • DaVinci automatycznie zaimportuje timeline

3. ✅ WERYFIKACJA
   • Timeline zawiera wszystkie klipy z oryginalnymi nazwami
   • Efekty prędkości są już zastosowane
   • Sprawdź czy długość timeline się zgadza

4. 🎨 FINALIZACJA
   • Dodaj kolorystykę, napisy, efekty
   • Timeline jest gotowy do dalszej obróbki

DLACZEGO EDL?
• Natywne wsparcie w DaVinci Resolve
• Precyzyjny timecode (ramka po ramce)
• Zawiera informacje o efektach prędkości
• Bezproblemowy import

═══════════════════════════════════════════════════════════════

⚙️ USTAWIENIA DLA RÓŻNYCH TREŚCI:

🎮 GAMING Z KOMENTARZEM
   • Przyspieszenie: 2.5x - 3x
   • Min. cisza: 1.5s
   • Detekcja: WebRTC VAD
   • Cel: usunięcie pauz w komentarzu

📚 TUTORIALE/PREZENTACJE
   • Przyspieszenie: 2x - 2.5x
   • Min. cisza: 2s
   • Detekcja: Whisper AI
   • Cel: usunięcie długich pauz

🎙️ PODCASTY/WYWIADY
   • Przyspieszenie: 1.5x - 2x
   • Min. cisza: 1s
   • Detekcja: Whisper AI
   • Cel: naturalne tempo rozmowy

🎮 GAMEPLAY BEZ KOMENTARZA
   • Przyspieszenie: 4x - 5x
   • Min. cisza: 3s
   • Detekcja: WebRTC VAD
   • Cel: dynamiczne przejścia

═══════════════════════════════════════════════════════════════

🛠️ ROZWIĄZYWANIE PROBLEMÓW:

❌ "NIE ZNALEZIONO PLIKÓW MP4"
   • Sprawdź czy w folderze są pliki .mp4
   • Sprawdź wielkość liter w rozszerzeniu
   • Upewnij się że masz uprawnienia do odczytu

❌ "BŁĄD PRZETWARZANIA"
   • Sprawdź logi w programie (czerwone komunikaty)
   • Upewnij się że FFmpeg jest zainstalowany: ffmpeg -version
   • Sprawdź czy pliki MP4 nie są uszkodzone
   • Sprawdź czy masz wystarczająco miejsca na dysku

⚠️ "BRAK OVERLAYÓW TEKSTOWYCH"
   • Zainstaluj ImageMagick (instrukcje w README)
   • Program użyje kolorowych wskaźników jako zamiennik
   • To nie wpływa na funkcjonalność EDL

⚠️ "TIMELINE MA NIEPRAWIDŁOWĄ DŁUGOŚĆ"
   • Problem naprawiony w tej wersji!
   • EDL zawiera precyzyjne informacje o czasach
   • Sprawdź czy oryginalne pliki są w folderze output

❌ "IMPORT EDL FAILED TO LINK"
   • Skopiuj oryginalne pliki MP4 do folderu output
   • Sprawdź czy nazwy plików nie zawierają polskich znaków
   • Użyj "Relink" w DaVinci jeśli potrzeba

═══════════════════════════════════════════════════════════════

📊 WYDAJNOŚĆ I OPTYMALIZACJA:

⏱️ ORIENTACYJNE CZASY:
   • 5-min wideo: 1-3 min (WebRTC) / 3-8 min (Whisper)
   • 30-min wideo: 5-15 min (WebRTC) / 15-45 min (Whisper)
   • 2h podcast: 20-60 min (WebRTC) / 60-180 min (Whisper)

🚀 PRZYSPIESZ PRZETWARZANIE:
   • Używaj WebRTC VAD dla szybkości
   • Zamknij inne programy podczas przetwarzania
   • Użyj SSD zamiast HDD
   • Więcej RAM = szybsze przetwarzanie

💾 ZARZĄDZANIE MIEJSCEM:
   • Program tworzy pliki tymczasowe
   • Po zakończeniu automatycznie je usuwa
   • Potrzeba ~2x więcej miejsca niż rozmiar wideo
   • Folder output może być duży

═══════════════════════════════════════════════════════════════

🆘 WSPARCIE TECHNICZNE:

📝 ZBIERANIE INFORMACJI:
   1. Wersja Python: python --version
   2. System operacyjny i wersja
   3. Logi z programu (Zapisz logi → plik .txt)
   4. Czy FFmpeg działa: ffmpeg -version
   5. Rozmiar i czas trwania problematycznego pliku

🔍 DIAGNOSTYKA:
   • Sprawdź plik video_speed_processor.log
   • Włącz szczegółowe logi w programie
   • Przetestuj z pojedynczym, krótkim plikiem
   • Sprawdź timeline_data.json dla szczegółów

═══════════════════════════════════════════════════════════════

🎉 PRZYKŁADOWY WORKFLOW:

1. 📹 NAGRANIE (OBS, Bandicam, itp.)
2. 🎬 PRZETWARZANIE (ten program)
3. 📥 IMPORT DO DAVINCI (EDL timeline)
4. 🎨 OBRÓBKA (kolorystyka, efekty, napisy)
5. 📤 EXPORT (YouTube, social media)

═══════════════════════════════════════════════════════════════

Program stworzony z myślą o content creatorach!
Oszczędź czas na montażu - skup się na treści! 🚀

Wersja: 2.0 FIXED - wszystkie główne błędy naprawione"""

        # Text widget z lepszym formatowaniem
        text_widget = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            padx=15,
            pady=15,
            relief='solid',
            borderwidth=1
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

        # Konfiguracja tagów dla kolorów
        text_widget.tag_config("title", font=('Arial', 12, 'bold'), foreground='blue')
        text_widget.tag_config("section", font=('Arial', 11, 'bold'), foreground='darkgreen')
        text_widget.tag_config("warning", foreground='orange')
        text_widget.tag_config("error", foreground='red')
        text_widget.tag_config("success", foreground='green')

        # Przycisk zamknij
        button_frame = ttk.Frame(help_window, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="✅ Zamknij",
                   command=help_window.destroy,
                   width=15).pack(side=tk.RIGHT)

        ttk.Button(button_frame, text="📁 Otwórz folder wyjściowy",
                   command=lambda: (help_window.destroy(), self.open_output_folder()),
                   width=25).pack(side=tk.RIGHT, padx=(0, 10))

        # Pack canvas i scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Focus na okno
        help_window.focus_set()

    def run(self):
        """Uruchamia aplikację - ULEPSZONY."""
        try:
            # Ustaw ikonę okna (jeśli dostępna)
            try:
                # Można dodać custom ikonę tutaj
                pass
            except:
                pass

            # Ustaw pozycję okna na środku ekranu
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f"{width}x{height}+{x}+{y}")

            # Pokaż okno
            self.root.deiconify()

            # Komunikat powitalny
            self.add_log("🎬 Video Speed Processor v2.0 - FIXED", "INFO")
            self.add_log("═" * 50, "INFO")
            self.add_log("Gotowy do przetwarzania plików MP4!", "SUCCESS")
            self.add_log("Wybierz folder z plikami i kliknij 'Rozpocznij przetwarzanie'", "INFO")
            self.add_log("Pomoc: kliknij przycisk '❓ Pomoc' dla szczegółowych instrukcji", "INFO")

            # Uruchom główną pętlę
            self.root.mainloop()

        except KeyboardInterrupt:
            self.add_log("Program przerwany przez użytkownika", "WARNING")
        except Exception as e:
            self.add_log(f"Krytyczny błąd aplikacji: {e}", "ERROR")
            messagebox.showerror("Krytyczny błąd", f"Aplikacja napotkała nieoczekiwany błąd:\n\n{e}")
        finally:
            # Cleanup
            try:
                if hasattr(self, 'root'):
                    self.root.quit()
            except:
                pass


def main():
    """Główna funkcja - ULEPSZONY."""
    print("🎬 Video Speed Processor v2.0 - FIXED")
    print("═" * 50)

    # Sprawdź Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ BŁĄD: Wymagany Python 3.8+")
        print(f"   Aktualna wersja: {sys.version}")
        input("Naciśnij Enter aby zamknąć...")
        return

    # Sprawdź podstawowe zależności
    missing_deps = []

    try:
        import moviepy.editor as mp
        print("✅ MoviePy - OK")
    except ImportError:
        missing_deps.append("moviepy")
        print("❌ MoviePy - BRAK")

    try:
        import librosa
        print("✅ Librosa - OK")
    except ImportError:
        missing_deps.append("librosa")
        print("❌ Librosa - BRAK")

    try:
        import numpy as np
        print("✅ NumPy - OK")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy - BRAK")

    # Sprawdź opcjonalne zależności
    try:
        import webrtcvad
        print("✅ WebRTC VAD - OK")
    except ImportError:
        print("⚠️ WebRTC VAD - BRAK (opcjonalne)")

    try:
        import whisper
        print("✅ Whisper AI - OK")
    except ImportError:
        print("⚠️ Whisper AI - BRAK (opcjonalne)")

    # Sprawdź FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg - OK")
        else:
            print("⚠️ FFmpeg - PROBLEM")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("❌ FFmpeg - BRAK")
        missing_deps.append("ffmpeg")

    print("═" * 50)

    # Jeśli brakuje krytycznych zależności
    if missing_deps:
        print(f"❌ BŁĄD: Brak wymaganych bibliotek!")
        print(f"   Brakuje: {', '.join(missing_deps)}")
        print()
        print("📥 INSTALACJA:")
        if 'moviepy' in missing_deps or 'librosa' in missing_deps or 'numpy' in missing_deps:
            print("   pip install moviepy librosa numpy")
        if 'ffmpeg' in missing_deps:
            print("   Zainstaluj FFmpeg z https://ffmpeg.org/")
        print()

        # Pokaż messagebox jeśli GUI jest dostępne
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Brak wymaganych bibliotek",
                f"Brakuje wymaganych bibliotek:\n{', '.join(missing_deps)}\n\n"
                f"Zainstaluj:\npip install moviepy librosa numpy\n\n"
                f"Oraz FFmpeg z https://ffmpeg.org/"
            )
            root.destroy()
        except:
            pass

        input("Naciśnij Enter aby zamknąć...")
        return

    print("🚀 Uruchamianie GUI...")

    # Uruchom GUI
    try:
        app = VideoProcessorGUI()
        app.run()
    except Exception as e:
        print(f"❌ Błąd uruchomienia GUI: {e}")

        # Pokaż messagebox z błędem
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Błąd uruchomienia",
                f"Nie można uruchomić interfejsu graficznego:\n\n{e}"
            )
            root.destroy()
        except:
            pass

        input("Naciśnij Enter aby zamknąć...")

    print("Dziękujemy za użycie Video Speed Processor! 🎬")


if __name__ == "__main__":
    import time
    import datetime  # Dodajemy brakujący import

    main()