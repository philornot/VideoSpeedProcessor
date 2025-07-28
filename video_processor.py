#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Speed Processor
Automatycznie przetwarza nagrania wideo, przyspieszając fragmenty ciszy
i eksportując do DaVinci Resolve.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Importy zewnętrzne
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
                """Detekcja mowy używając WebRTC VAD - ULEPSZONA AGREGACJA."""
                self.logger.info("Używam WebRTC VAD do detekcji mowy...")

                try:
                    # Wczytaj audio
                    y, sr = librosa.load(audio_path, sr=16000)  # WebRTC wymaga 16kHz

                    # Inicjalizuj VAD
                    vad = webrtcvad.Vad(1)  # Mniej agresywny (0-3)

                    # Podziel na ramki 30ms
                    frame_duration = 0.03  # 30ms
                    frame_length = int(sr * frame_duration)

                    # Analizuj każdą ramkę
                    speech_frames = []
                    for i in range(0, len(y) - frame_length, frame_length):
                        frame = y[i:i + frame_length]
                        frame_time = i / sr

                        # Konwertuj do int16
                        frame_int16 = (frame * 32767).astype(np.int16).tobytes()

                        # Sprawdź czy to mowa
                        try:
                            is_speech = vad.is_speech(frame_int16, sr)
                            speech_frames.append((frame_time, is_speech))
                        except:
                            speech_frames.append((frame_time, False))

                    if not speech_frames:
                        audio_duration = len(y) / sr
                        return [(0.0, audio_duration, False)]

                    # KLUCZOWA ZMIANA: Agreguj ramki w większe segmenty
                    segments = []
                    current_start = 0.0
                    current_is_speech = speech_frames[0][1]

                    # Parametry agregacji
                    min_segment_duration = 0.5  # Minimum 0.5s na segment
                    speech_merge_gap = 0.3      # Łącz mowę z przerwami < 0.3s
                    silence_merge_gap = 0.5     # Łącz ciszę z przerwami < 0.5s

                    for frame_time, is_speech in speech_frames:
                        # Jeśli zmienił się typ segmentu
                        if is_speech != current_is_speech:
                            duration = frame_time - current_start

                            # Dodaj segment tylko jeśli ma minimalną długość
                            if duration >= min_segment_duration:
                                segments.append((current_start, frame_time, current_is_speech))
                                current_start = frame_time
                                current_is_speech = is_speech
                            # Jeśli za krótki, kontynuuj poprzedni segment

                    # Dodaj ostatni segment
                    audio_duration = len(y) / sr
                    final_duration = audio_duration - current_start
                    if final_duration >= min_segment_duration:
                        segments.append((current_start, audio_duration, current_is_speech))

                    # POST-PROCESSING: Połącz podobne segmenty z krótkimi przerwami
                    merged_segments = []
                    i = 0
                    while i < len(segments):
                        start, end, is_speech = segments[i]

                        # Sprawdź czy można połączyć z następnym
                        while i + 1 < len(segments):
                            next_start, next_end, next_is_speech = segments[i + 1]
                            gap = next_start - end

                            # Połącz jeśli:
                            # 1. Ten sam typ (mowa/cisza)
                            # 2. Krótka przerwa między segmentami
                            merge_gap = speech_merge_gap if is_speech else silence_merge_gap

                            if next_is_speech == is_speech and gap <= merge_gap:
                                # Połącz segmenty
                                end = next_end
                                i += 1
                            else:
                                break

                        merged_segments.append((start, end, is_speech))
                        i += 1

                    # FINAL FILTER: Usuń bardzo krótkie segmenty ciszy
                    final_segments = []
                    for start, end, is_speech in merged_segments:
                        duration = end - start

                        if is_speech or duration >= self.config.min_silence_duration:
                            final_segments.append((start, end, is_speech))
                        else:
                            # Bardzo krótka cisza - połącz z sąsiednim segmentem
                            if final_segments:
                                # Rozszerz poprzedni segment
                                prev_start, prev_end, prev_is_speech = final_segments[-1]
                                final_segments[-1] = (prev_start, end, prev_is_speech)
                            else:
                                # Pierwszy segment - zostaw jako mowę
                                final_segments.append((start, end, True))

                    self.logger.info(f"WebRTC: {len(speech_frames)} ramek -> {len(segments)} segmentów -> {len(final_segments)} końcowych")

                    return final_segments

                except Exception as e:
                    self.logger.error(f"Błąd WebRTC VAD: {e}")
                    return self._detect_speech_energy(audio_path)


            def _detect_speech_energy(self, audio_path: str) -> List[Tuple[float, float, bool]]:
                """Prosta detekcja na podstawie energii audio - ULEPSZONA."""
                self.logger.info("Używam detekcji energii audio...")

                try:
                    y, sr = librosa.load(audio_path)

                    # Oblicz RMS energy w większych oknach
                    frame_length = int(sr * 0.5)  # 500ms okna (większe!)
                    hop_length = int(sr * 0.25)   # 250ms przesunięcie

                    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

                    # Próg na podstawie percentyla
                    threshold = np.percentile(rms, 25)  # 25% najcichszych fragmentów to cisza

                    # Konwertuj na segmenty czasowe
                    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

                    segments = []
                    current_start = 0.0
                    current_is_speech = rms[0] > threshold

                    for i, (time, energy) in enumerate(zip(times, rms)):
                        is_speech = energy > threshold

                        if is_speech != current_is_speech:
                            duration = time - current_start
                            if duration >= 0.5:  # Minimum 0.5s
                                segments.append((current_start, time, current_is_speech))
                                current_start = time
                                current_is_speech = is_speech

                    # Dodaj ostatni segment
                    audio_duration = len(y) / sr
                    if audio_duration - current_start >= 0.5:
                        segments.append((current_start, audio_duration, current_is_speech))

                    # Filtruj bardzo krótkie segmenty ciszy
                    filtered_segments = []
                    for start, end, is_speech in segments:
                        duration = end - start
                        if is_speech or duration >= self.config.min_silence_duration:
                            filtered_segments.append((start, end, is_speech))

                    self.logger.info(f"Energy: {len(segments)} segmentów -> {len(filtered_segments)} po filtracji")
                    return filtered_segments

                except Exception as e:
                    self.logger.error(f"Błąd detekcji energii: {e}")
                    return []


            def _seconds_to_timecode(self, seconds: float, fps: int = 25) -> str:
                """Konwertuje sekundy na timecode HH:MM:SS:FF - POPRAWIONA."""
                # Zabezpieczenie przed ujemnymi wartościami
                seconds = max(0, seconds)

                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                frames = int((seconds % 1) * fps)

                # Zabezpieczenie przed przekroczeniem fps
                if frames >= fps:
                    frames = fps - 1
                    secs += 1

                if secs >= 60:
                    secs = 0
                    minutes += 1

                if minutes >= 60:
                    minutes = 0
                    hours += 1

                return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
            def generate_edl(self, results: List[dict], output_path: str):
                """Generuje plik EDL dla DaVinci Resolve - FIX ŚCIEŻEK."""
                self.logger.info(f"Generowanie EDL: {output_path}")

                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        # Nagłówek EDL
                        f.write("TITLE: Processed_Videos\n")
                        f.write("FCM: NON-DROP FRAME\n\n")

                        edit_number = 1
                        timeline_pos = 0.0

                        for result_idx, result in enumerate(results):
                            if not result:
                                continue

                            source_name = result['input_name']
                            reel_name = f"AX{result_idx + 1:03d}"

                            self.logger.info(f"Przetwarzanie EDL dla {source_name}: {len(result['timeline_data'])} segmentów")

                            for segment_idx, segment in enumerate(result['timeline_data']):
                                # Oblicz dokładny czas źródłowy
                                source_start = segment['start']
                                original_duration = segment.get('original_duration', segment['duration'])
                                source_end = source_start + original_duration

                                # Czas w timeline
                                timeline_start = timeline_pos
                                timeline_end = timeline_pos + segment['duration']

                                # Konwertuj sekundy na timecode (25fps)
                                source_in_tc = self._seconds_to_timecode(source_start, 25)
                                source_out_tc = self._seconds_to_timecode(source_end, 25)
                                timeline_in_tc = self._seconds_to_timecode(timeline_start, 25)
                                timeline_out_tc = self._seconds_to_timecode(timeline_end, 25)

                                # Linia EDL
                                edit_type = "C"
                                track = "V"

                                f.write(f"{edit_number:03d}  {reel_name:<8} {track}     {edit_type}        ")
                                f.write(f"{source_in_tc} {source_out_tc} {timeline_in_tc} {timeline_out_tc}\n")

                                # Metadane klipu
                                f.write(f"* FROM CLIP NAME: {source_name}\n")
                                f.write(f"* SEGMENT TYPE: {segment['type']}\n")

                                # Efekty prędkości
                                if segment['speed'] != 1.0:
                                    f.write(f"* SPEED: {segment['speed']:.2f}\n")
                                    f.write(f"* OVERLAY: x{segment['speed']:.1f}\n")  # Informacja o overlay

                                    # M2 effect line dla DaVinci
                                    speed_percent = segment['speed'] * 100
                                    f.write(f"M2   {reel_name:<8}     050 {timeline_in_tc} {timeline_out_tc} {speed_percent:06.2f} {speed_percent:06.2f}\n")
                                else:
                                    f.write(f"* SPEED: 1.00 (normal)\n")

                                f.write("\n")

                                edit_number += 1
                                timeline_pos = timeline_end

                        # KLUCZOWA ZMIANA: Poprawne ścieżki plików
                        f.write("* SOURCE FILE MAPPING:\n")
                        for result_idx, result in enumerate(results):
                            if result:
                                reel_name = f"AX{result_idx + 1:03d}"
                                # Użyj nazwy pliku, nie pełnej ścieżki (DaVinci szuka w Media Pool)
                                source_name = os.path.basename(result['input_file'])
                                f.write(f"* {reel_name}: {source_name}\n")

                    self.logger.info(f"✅ EDL zapisany: {output_path}")
                    self.logger.info(f"📊 Timeline zawiera {edit_number - 1} edytów")
                    self.logger.info(f"⏱️  Całkowity czas timeline: {timeline_pos:.2f}s")

                    # Podsumowanie
                    total_speech = sum(len([s for s in r['timeline_data'] if s['type'] == 'speech']) for r in results if r)
                    total_silence = sum(len([s for s in r['timeline_data'] if s['type'] in ['silence', 'short_silence']]) for r in results if r)
                    self.logger.info(f"🎤 Segmenty mowy: {total_speech}")
                    self.logger.info(f"🔇 Segmenty ciszy: {total_silence}")

                except Exception as e:
                    self.logger.error(f"❌ Błąd generowania EDL: {e}")
                    import traceback
                    self.logger.error(f"Szczegóły błędu: {traceback.format_exc()}")


    def generate_avid_log(self, results: List[dict], output_path: str):
        """Generuje plik Avid Log Exchange (ALE) jako alternatywę."""
        self.logger.info(f"Generowanie ALE: {output_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Nagłówek ALE
                f.write("Heading\n")
                f.write("FIELD_DELIM\tTABS\n")
                f.write("VIDEO_FORMAT\t1080\n")
                f.write("AUDIO_FORMAT\t48kHz\n")
                f.write("FPS\t25\n\n")

                f.write("Column\n")
                f.write("Name\tTape\tStart\tEnd\tDuration\tSpeed\tType\tSource File\n\n")

                f.write("Data\n")

                for result_idx, result in enumerate(results):
                    if not result:
                        continue

                    source_name = result['input_name']
                    tape_name = f"AX{result_idx + 1:03d}"

                    for clip_idx, segment in enumerate(result['timeline_data']):
                        clip_name = f"{source_name}_{clip_idx + 1:02d}"

                        start_tc = self._seconds_to_timecode(segment['start'], 25)
                        end_tc = self._seconds_to_timecode(
                            segment['start'] + segment.get('original_duration', segment['duration']), 25)
                        duration_tc = self._seconds_to_timecode(segment['duration'], 25)

                        f.write(f"{clip_name}\t{tape_name}\t{start_tc}\t{end_tc}\t{duration_tc}\t")
                        f.write(f"{segment['speed']:.1f}x\t{segment['type']}\t{result['input_file']}\n")

            self.logger.info(f"ALE zapisany: {output_path}")

        except Exception as e:
            self.logger.error(f"Błąd generowania ALE: {e}")

    def generate_csv_timeline(self, results: List[dict], output_path: str):
        """Generuje prosty CSV z informacjami o timeline."""
        self.logger.info(f"Generowanie CSV timeline: {output_path}")

        try:
            import csv

            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Edit_Number', 'Source_File', 'Timeline_Start', 'Timeline_End',
                              'Source_Start', 'Source_End', 'Duration', 'Speed', 'Type', 'Notes']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                edit_number = 1
                timeline_pos = 0.0

                for result in results:
                    if not result:
                        continue

                    for segment in result['timeline_data']:
                        writer.writerow({
                            'Edit_Number': edit_number,
                            'Source_File': result['input_name'],
                            'Timeline_Start': f"{timeline_pos:.3f}s",
                            'Timeline_End': f"{timeline_pos + segment['duration']:.3f}s",
                            'Source_Start': f"{segment['start']:.3f}s",
                            'Source_End': f"{segment['start'] + segment.get('original_duration', segment['duration']):.3f}s",
                            'Duration': f"{segment['duration']:.3f}s",
                            'Speed': f"{segment['speed']:.1f}x",
                            'Type': segment['type'],
                            'Notes': f"Original duration: {segment.get('original_duration', segment['duration']):.3f}s"
                        })

                        edit_number += 1
                        timeline_pos += segment['duration']

            self.logger.info(f"CSV zapisany: {output_path}")

        except Exception as e:
            self.logger.error(f"Błąd generowania CSV: {e}")

    def create_simple_overlay(self, duration: float, speed: float, size: Tuple[int, int]) -> mp.VideoClip:
        """Tworzy prosty overlay bez tekstu (fallback dla braku ImageMagick)."""
        try:
            # Utwórz kolorowy prostokąt jako wskaźnik przyspieszenia
            import numpy as np

            # Rozmiar prostokąta zależny od prędkości
            rect_size = int(min(size[0], size[1]) * 0.05)  # 5% rozmiaru ekranu

            # Kolor zależny od prędkości (czerwony = szybko, żółty = średnio)
            if speed >= 4.0:
                color = [255, 0, 0]  # Czerwony
            elif speed >= 2.5:
                color = [255, 165, 0]  # Pomarańczowy
            else:
                color = [255, 255, 0]  # Żółty

            # Stwórz prostokąt
            def make_frame(t):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

                # Pozycja w prawym dolnym rogu
                margin = rect_size // 2
                y_start = size[1] - rect_size - margin
                y_end = size[1] - margin
                x_start = size[0] - rect_size - margin
                x_end = size[0] - margin

                # Rysuj prostokąt
                frame[y_start:y_end, x_start:x_end] = color

                # Dodaj miganie dla większej widoczności
                if int(t * 2) % 2 == 0:  # Migaj co 0.5s
                    frame[y_start:y_end, x_start:x_end] = [min(c + 50, 255) for c in color]

                return frame

            overlay = mp.VideoClip(make_frame, duration=duration)
            overlay = overlay.set_opacity(0.8)  # Półprzezroczysty

            return overlay

        except Exception as e:
            self.logger.warning(f"Nie można utworzyć prostego overlay: {e}")
            return None

    def create_speed_overlay(self, duration: float, speed: float, size: Tuple[int, int]) -> mp.VideoClip:
        """Tworzy overlay z tekstem prędkości - WERSJA BEZ KOMPOZYCJI."""
        if speed == 1.0:
            return None

        text = f"x{speed:.1f}" if speed != int(speed) else f"x{int(speed)}"

        try:
            # ZMIANA: Tworzymy overlay bez kompozycji - będzie dodany w DaVinci
            self.logger.info(f"Tworzę overlay '{text}' dla segmentu {duration:.2f}s")

            # Sprawdź czy ImageMagick działa
            try:
                txt_clip = mp.TextClip(
                    text,
                    fontsize=max(48, min(size[0], size[1]) // 15),  # Większy tekst
                    color='white',
                    font='Arial-Bold',
                    stroke_color='black',
                    stroke_width=3
                ).set_duration(duration)

                # Pozycjonuj w prawym dolnym rogu
                margin = min(size[0], size[1]) // 30
                txt_clip = txt_clip.set_position((size[0] - txt_clip.w - margin, size[1] - txt_clip.h - margin))

                self.logger.debug(f"TextClip utworzony: {txt_clip.w}x{txt_clip.h} na {duration:.2f}s")
                return txt_clip

            except Exception as e:
                self.logger.warning(f"TextClip niedostępny (ImageMagick): {e}")

                # Fallback: kolorowy prostokąt
                return self.create_simple_overlay(duration, speed, size)

        except Exception as e:
            self.logger.error(f"Błąd tworzenia overlay: {e}")
            return None


    def process_video_file(self, input_path: str, output_folder: str) -> dict:
        """Przetwarza pojedynczy plik wideo - BEZ OVERLAYÓW W KLIPACH."""
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

            if not segments:
                self.logger.warning(f"Brak segmentów w: {input_path}")
                video.close()
                return None

            # ZMIANA: Nie tworzymy fizycznego wideo z overlayami
            # Tylko zbieramy informacje dla EDL
            timeline_data = []

            self.logger.info(f"Przetwarzanie {len(segments)} segmentów")

            for i, (start, end, is_speech) in enumerate(segments):
                duration = end - start

                self.logger.debug(f"Segment {i+1}: {start:.2f}-{end:.2f}s, "
                                f"duration={duration:.2f}s, speech={is_speech}")

                if is_speech:
                    # Fragment mowy - normalne tempo
                    timeline_data.append({
                        'start': start,
                        'end': end,
                        'duration': duration,
                        'speed': 1.0,
                        'type': 'speech',
                        'has_overlay': False
                    })
                    self.logger.debug(f"  -> Mowa: {duration:.2f}s (1.0x)")
                else:
                    # Fragment ciszy - przyspiesz TYLKO jeśli spełnia minimum
                    if duration >= self.config.min_silence_duration:
                        speed = self.config.speed_multiplier
                        new_duration = duration / speed

                        timeline_data.append({
                            'start': start,
                            'end': end,
                            'duration': new_duration,
                            'original_duration': duration,
                            'speed': speed,
                            'type': 'silence',
                            'has_overlay': True  # Informacja dla DaVinci
                        })
                        self.logger.debug(f"  -> Cisza: {duration:.2f}s -> {new_duration:.2f}s ({speed}x)")
                    else:
                        # Krótka cisza - pozostaw normalną
                        timeline_data.append({
                            'start': start,
                            'end': end,
                            'duration': duration,
                            'speed': 1.0,
                            'type': 'short_silence',
                            'has_overlay': False
                        })
                        self.logger.debug(f"  -> Krótka cisza: {duration:.2f}s (1.0x)")

            # Oblicz całkowity czas wyjściowy
            output_duration = sum(segment['duration'] for segment in timeline_data)

            # Przygotuj nazwy plików wyjściowych
            input_name = Path(input_path).stem

            result = {
                'input_file': input_path,
                'input_name': input_name,
                'timeline_data': timeline_data,
                'output_duration': output_duration,
                'original_duration': video.duration,
                'segments_count': len(segments),
                'speech_segments': len([s for s in timeline_data if s['type'] == 'speech']),
                'silence_segments': len([s for s in timeline_data if s['type'] in ['silence', 'short_silence']])
            }

            self.logger.info(f"Rezultat: {result['speech_segments']} segmentów mowy, "
                            f"{result['silence_segments']} segmentów ciszy")
            self.logger.info(f"Czas: {video.duration:.2f}s -> {output_duration:.2f}s")

            # Opcjonalnie: zapisz przetworzone wideo (tylko jeśli requested)
            if self.config.generate_video:
                self.logger.info("Generowanie wideo z overlayami...")
                processed_clips = []

                for segment in timeline_data:
                    segment_clip = video.subclip(segment['start'], segment['start'] + segment.get('original_duration', segment['duration']))

                    if segment['speed'] != 1.0:
                        segment_clip = segment_clip.fx(mp.vfx.speedx, segment['speed'])

                        # Dodaj overlay tylko do wideo (nie do EDL)
                        overlay = self.create_speed_overlay(
                            segment_clip.duration,
                            segment['speed'],
                            (video.w, video.h)
                        )

                        if overlay:
                            segment_clip = mp.CompositeVideoClip([segment_clip, overlay])

                    processed_clips.append(segment_clip)

                if processed_clips:
                    final_video = mp.concatenate_videoclips(processed_clips)
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
                    final_video.close()

            # Zamknij klip
            video.close()
            return result

        except Exception as e:
            self.logger.error(f"Błąd przetwarzania {input_path}: {e}")
            import traceback
            self.logger.error(f"Szczegóły: {traceback.format_exc()}")
            return None


    def copy_source_files_to_output(self, results: List[dict], output_folder: str):
        """Kopiuje oryginalne pliki wideo do folderu wyjściowego - POPRAWIONE ŚCIEŻKI."""
        import shutil

        self.logger.info("Kopiowanie oryginalnych plików do folderu wyjściowego...")

        for result in results:
            if not result:
                continue

            source_path = result['input_file']
            source_name = os.path.basename(source_path)
            dest_path = os.path.join(output_folder, source_name)

            try:
                if not os.path.exists(dest_path):
                    shutil.copy2(source_path, dest_path)
                    self.logger.info(f"Skopiowano: {source_name}")

                # KLUCZOWA ZMIANA: Zaktualizuj ścieżkę na względną nazwę pliku
                result['input_file'] = source_name  # Tylko nazwa pliku, nie pełna ścieżka

            except Exception as e:
                self.logger.warning(f"Nie można skopiować {source_name}: {e}")


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

            # Dodaj assety dla każdego pliku wideo
            asset_id = 1
            for result in results:
                if not result:
                    continue

                # Dodaj zasób wideo z pełną ścieżką
                input_file_path = os.path.abspath(result['input_file'])
                asset = SubElement(resources, 'asset',
                                   id=f'r{asset_id}',
                                   name=result['input_name'],
                                   src=f"file://{input_file_path.replace(os.sep, '/')}",
                                   duration=f"{result['original_duration']}s")
                asset_id += 1

            # Dodaj bibliotekę (wymagane przez DaVinci)
            library = SubElement(fcpxml, 'library')

            # Dodaj event (wymagane przez DaVinci)
            event = SubElement(library, 'event', name='Processed Videos Event')

            # Projekt
            project = SubElement(event, 'project', name='Processed_Videos')

            # Oblicz całkowity czas trwania timeline (suma wszystkich segmentów)
            total_duration = 0
            for result in results:
                if result and result['timeline_data']:
                    for segment in result['timeline_data']:
                        total_duration += segment['duration']

            self.logger.info(f"Obliczony całkowity czas timeline: {total_duration:.2f}s")

            sequence = SubElement(project, 'sequence',
                                  format='r1',
                                  duration=f'{max(total_duration + 10, 60)}s',  # Min 60s, +10s bufor
                                  tcStart='0s',
                                  tcFormat='NDF')

            spine = SubElement(sequence, 'spine')

            # Dodaj klipy do timeline
            current_offset = 0
            asset_id = 1

            for result in results:
                if not result:
                    continue

                # Dodaj klipy do timeline dla każdego segmentu
                for segment in result['timeline_data']:
                    # Oblicz dokładne czasy
                    clip_duration = segment['duration']
                    source_start = segment['start']  # Początek w oryginalnym pliku
                    source_duration = segment.get('original_duration', clip_duration)

                    # Podstawowe atrybuty klipu
                    clip_attrs = {
                        'ref': f'r{asset_id}',
                        'offset': f'{current_offset}s',
                        'duration': f'{clip_duration}s'
                    }

                    # Dla segmentów ciszy (przyspieszonych) dodaj start/end
                    if segment['speed'] != 1.0:
                        clip_attrs['start'] = f'{source_start}s'
                        clip_attrs['end'] = f'{source_start + source_duration}s'
                    else:
                        # Dla normalnych segmentów też dodaj start/end
                        clip_attrs['start'] = f'{source_start}s'
                        clip_attrs['end'] = f'{source_start + source_duration}s'

                    clip = SubElement(spine, 'video', **clip_attrs)

                    # Dodaj efekt prędkości dla przyspieszonych segmentów
                    if segment['speed'] != 1.0:
                        # Użyj timeMap dla kontroli prędkości
                        timemap = SubElement(clip, 'timeMap')

                        # Punkt początkowy
                        timept1 = SubElement(timemap, 'timept')
                        timept1.set('time', '0s')
                        timept1.set('value', '0s')
                        timept1.set('interp', 'smooth2')

                        # Punkt końcowy - mapowanie czasu
                        timept2 = SubElement(timemap, 'timept')
                        timept2.set('time', f'{clip_duration}s')  # Czas w timeline
                        timept2.set('value', f'{source_duration}s')  # Czas w źródle
                        timept2.set('interp', 'smooth2')

                        # Dodaj audio remap jeśli potrzebne
                        audio_map = SubElement(clip, 'audioRoleMap')
                        audio_map.set('enabled', 'false')  # Wyłącz audio dla przyspieszonych

                    # Debug info jako komentarz
                    comment = SubElement(clip, 'note')
                    comment.text = f"Speed: {segment['speed']}x, Type: {segment['type']}, Source: {source_start}-{source_start + source_duration}"

                    current_offset += clip_duration

                asset_id += 1

            # Dodaj podstawowe ustawienia projektu
            project_settings = SubElement(sequence, 'projectSettings')
            project_settings.set('width', '1920')
            project_settings.set('height', '1080')
            project_settings.set('frameDuration', '1/25s')
            project_settings.set('audioLayout', 'stereo')
            project_settings.set('audioRate', '48k')

            # Zapisz XML z lepszym formatowaniem
            rough_string = tostring(fcpxml, 'utf-8')
            reparsed = minidom.parseString(rough_string)

            # Usuń puste linie i popraw formatowanie
            pretty_xml = reparsed.toprettyxml(indent='  ')
            pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

            self.logger.info(f"FCPXML zapisany: {output_path}")
            self.logger.info(f"Timeline zawiera {len([r for r in results if r])} plików wideo")
            self.logger.info(f"Całkowity czas: {total_duration:.2f}s")

        except Exception as e:
            self.logger.error(f"Błąd generowania FCPXML: {e}")
            import traceback
            self.logger.error(f"Szczegóły błędu: {traceback.format_exc()}")

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

        # Skopiuj oryginalne pliki do folderu wyjściowego (dla timeline)
        if self.config.generate_timeline:
            self.copy_source_files_to_output(results, str(output_folder))

        # Generuj timeline jeśli wymagane
        if self.config.generate_timeline:
            # Generuj EDL (główny format)
            edl_path = output_folder / "timeline.edl"
            self.generate_edl(results, str(edl_path))

            # Generuj ALE jako alternatywę
            ale_path = output_folder / "timeline.ale"
            self.generate_avid_log(results, str(ale_path))

            # Generuj CSV dla ułatwienia
            csv_path = output_folder / "timeline.csv"
            self.generate_csv_timeline(results, str(csv_path))

            # Opcjonalnie zachowaj FCPXML
            if hasattr(self.config, 'generate_fcpxml') and self.config.generate_fcpxml:
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
            print(f"🎬 Timeline EDL: {output_folder}/timeline.edl")
            print(f"📊 Timeline ALE: {output_folder}/timeline.ale")
            print(f"📋 Timeline CSV: {output_folder}/timeline.csv")


def check_imagemagick():
    """Sprawdza czy ImageMagick jest dostępny."""
    try:
        # Wymuś użycie naszej ścieżki
        os.environ['IMAGEMAGICK_BINARY'] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

        import moviepy.editor as mp
        import moviepy.config as mp_config
        mp_config.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

        # Test czy TextClip działa
        test_clip = mp.TextClip("test", fontsize=20, color='white').set_duration(0.1)
        test_clip.close()
        return True
    except Exception as e:
        print(f"ImageMagick test failed: {e}")
        return False


def print_imagemagick_instructions():
    """Wyświetla instrukcje instalacji ImageMagick."""
    print("\n" + "=" * 60)
    print("⚠️  UWAGA: ImageMagick nie jest zainstalowany")
    print("=" * 60)
    print("Overlay tekstowy (x3) nie będzie wyświetlany.")
    print("Zamiast tego użyję kolorowego wskaźnika.\n")
    print("🔧 Aby naprawić (opcjonalne):")
    print("1. Pobierz ImageMagick: https://imagemagick.org/script/download.php#windows")
    print("2. Podczas instalacji zaznacz 'Install development headers'")
    print("3. Lub przez chocolatey: choco install imagemagick")
    print("4. Uruchom skrypt ponownie")
    print("=" * 60 + "\n")


def main():
    """Główna funkcja programu."""
    parser = argparse.ArgumentParser(
        description="Video Speed Processor - automatyczne przetwarzanie nagrań .mp4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python video_processor.py --input_folder videos/ --speed_multiplier 3.0
  python video_processor.py --input_folder clips/ --speed_multiplier 2.5 --generate_video --combine_clips
  python video_processor.py --input_folder input/ --use_whisper --min_silence_duration 2.0
  python video_processor.py --input_folder input/ --speed_multiplier 3.0 --generate_fcpxml
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
                        help='Generuj timeline EDL/ALE/CSV (domyślnie: True)')
    parser.add_argument('--generate_fcpxml', action='store_true',
                        help='Generuj również FCPXML (opcjonalnie)')
    parser.add_argument('--combine_clips', action='store_true',
                        help='Połącz wszystkie klipy w jeden film')
    parser.add_argument('--use_whisper', action='store_true',
                        help='Użyj Whisper do detekcji mowy')
    parser.add_argument('--debug', action='store_true',
                        help='Włącz tryb debugowania')

    args = parser.parse_args()

    # Sprawdź ImageMagick
    if not check_imagemagick():
        print_imagemagick_instructions()

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
