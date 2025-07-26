#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Video Processor - Skrypt wsadowy dla wielu folderów
Automatycznie przetwarza wszystkie foldery z nagraniami SteelSeries Moments
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json


class BatchProcessor:
    """Klasa do wsadowego przetwarzania wielu folderów z nagraniami."""

    def __init__(self, config):
        self.config = config
        self.processor_script = "video_processor.py"
        self.results = []

    def find_video_folders(self, root_path: str) -> list:
        """Znajduje wszystkie foldery zawierające pliki MP4."""
        video_folders = []
        root = Path(root_path)

        print(f"🔍 Szukam folderów z plikami MP4 w: {root_path}")

        for folder in root.rglob("*"):
            if folder.is_dir():
                mp4_files = list(folder.glob("*.mp4"))
                if mp4_files:
                    video_folders.append({
                        'path': str(folder),
                        'name': folder.name,
                        'mp4_count': len(mp4_files),
                        'files': [f.name for f in mp4_files]
                    })
                    print(f"  📁 {folder.name}: {len(mp4_files)} plików MP4")

        return video_folders

    def process_folder(self, folder_info: dict) -> dict:
        """Przetwarza pojedynczy folder."""
        folder_path = folder_info['path']
        folder_name = folder_info['name']

        print(f"\n🎬 Przetwarzanie folderu: {folder_name}")
        print(f"   Lokalizacja: {folder_path}")
        print(f"   Plików MP4: {folder_info['mp4_count']}")

        # Przygotuj folder wyjściowy
        output_folder = Path(self.config.output_root) / f"processed_{folder_name}"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Przygotuj komendę
        cmd = [
            sys.executable, self.processor_script,
            "--input_folder", folder_path,
            "--output_folder", str(output_folder),
            "--speed_multiplier", str(self.config.speed_multiplier),
            "--min_silence_duration", str(self.config.min_silence_duration)
        ]

        # Dodaj opcjonalne parametry
        if self.config.generate_video:
            cmd.append("--generate_video")
        if self.config.generate_timeline:
            cmd.append("--generate_timeline")
        if self.config.combine_clips:
            cmd.append("--combine_clips")
        if self.config.use_whisper:
            cmd.append("--use_whisper")
        if self.config.debug:
            cmd.append("--debug")

        # Uruchom przetwarzanie
        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = result.returncode == 0

            folder_result = {
                'folder_name': folder_name,
                'folder_path': folder_path,
                'output_path': str(output_folder),
                'mp4_count': folder_info['mp4_count'],
                'success': success,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr if result.stderr else None
            }

            if success:
                print(f"   ✅ Sukces! Czas: {duration:.1f}s")
                print(f"   📁 Wyniki w: {output_folder}")
            else:
                print(f"   ❌ Błąd! Kod: {result.returncode}")
                print(f"   📝 Błąd: {result.stderr}")

            return folder_result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"   ❌ Wyjątek: {e}")

            return {
                'folder_name': folder_name,
                'folder_path': folder_path,
                'output_path': str(output_folder),
                'mp4_count': folder_info['mp4_count'],
                'success': False,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'error': str(e)
            }

    def generate_report(self):
        """Generuje raport z przetwarzania."""
        report_path = Path(self.config.output_root) / "batch_report.json"

        # Przygotuj statystyki
        total_folders = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total_folders - successful
        total_mp4s = sum(r['mp4_count'] for r in self.results)
        total_time = sum(r['duration_seconds'] for r in self.results)

        report = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'input_root': self.config.input_root,
                'output_root': self.config.output_root,
                'speed_multiplier': self.config.speed_multiplier,
                'use_whisper': self.config.use_whisper
            },
            'statistics': {
                'total_folders': total_folders,
                'successful_folders': successful,
                'failed_folders': failed,
                'total_mp4_files': total_mp4s,
                'total_processing_time_seconds': total_time,
                'average_time_per_folder': total_time / total_folders if total_folders > 0 else 0
            },
            'results': self.results
        }

        # Zapisz raport
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Wyświetl podsumowanie
        print(f"\n📊 PODSUMOWANIE BATCH PROCESSING")
        print(f"{'=' * 50}")
        print(f"📁 Przetworzone foldery: {successful}/{total_folders}")
        print(f"🎬 Łączna liczba plików MP4: {total_mp4s}")
        print(f"⏱️  Całkowity czas: {total_time / 60:.1f} minut")
        print(f"📈 Średni czas na folder: {total_time / total_folders / 60:.1f} minut")
        print(f"📄 Raport zapisany: {report_path}")

        if failed > 0:
            print(f"\n❌ Foldery z błędami:")
            for result in self.results:
                if not result['success']:
                    print(f"   • {result['folder_name']}: {result.get('error', 'Nieznany błąd')}")

    def run(self):
        """Uruchamia batch processing."""
        print("🚀 BATCH VIDEO PROCESSOR")
        print("=" * 50)

        # Sprawdź czy skrypt główny istnieje
        if not os.path.exists(self.processor_script):
            print(f"❌ Błąd: Nie znaleziono {self.processor_script}")
            print("   Upewnij się, że skrypt główny jest w tym samym folderze.")
            return

        # Znajdź foldery z wideo
        video_folders = self.find_video_folders(self.config.input_root)

        if not video_folders:
            print(f"❌ Nie znaleziono folderów z plikami MP4 w: {self.config.input_root}")
            return

        print(f"\n📋 Znaleziono {len(video_folders)} folderów do przetworzenia")

        # Poproś o potwierdzenie
        if not self.config.auto_confirm:
            response = input("\n🤔 Kontynuować? (y/N): ").strip().lower()
            if response not in ['y', 'yes', 'tak', 't']:
                print("⏹️  Anulowano przez użytkownika")
                return

        # Przetwórz wszystkie foldery
        print(f"\n🎬 Rozpoczynam przetwarzanie...")
        for i, folder_info in enumerate(video_folders, 1):
            print(f"\n[{i}/{len(video_folders)}] ", end="")
            result = self.process_folder(folder_info)
            self.results.append(result)

        # Wygeneruj raport
        self.generate_report()

        print(f"\n🎉 Batch processing zakończony!")


def main():
    """Główna funkcja batch processora."""
    parser = argparse.ArgumentParser(
        description="Batch Video Processor - wsadowe przetwarzanie wielu folderów",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python batch_processor.py --input_root "D:\\SteelSeries\\Moments" --speed_multiplier 3.0
  python batch_processor.py --input_root "C:\\Gaming\\Records" --speed_multiplier 2.5 --use_whisper --auto_confirm
        """
    )

    parser.add_argument('--input_root', required=True,
                        help='Główny folder do przeszukania (rekurencyjnie)')
    parser.add_argument('--output_root', default='batch_output',
                        help='Główny folder wyjściowy (domyślnie: batch_output)')
    parser.add_argument('--speed_multiplier', type=float, required=True,
                        help='Mnożnik prędkości dla fragmentów ciszy')
    parser.add_argument('--min_silence_duration', type=float, default=1.5,
                        help='Minimalna długość ciszy w sekundach')
    parser.add_argument('--generate_video', action='store_true',
                        help='Generuj gotowe pliki MP4')
    parser.add_argument('--generate_timeline', action='store_true', default=True,
                        help='Generuj timeline FCPXML')
    parser.add_argument('--combine_clips', action='store_true',
                        help='Połącz klipy w każdym folderze')
    parser.add_argument('--use_whisper', action='store_true',
                        help='Użyj Whisper do detekcji mowy')
    parser.add_argument('--debug', action='store_true',
                        help='Włącz tryb debugowania')
    parser.add_argument('--auto_confirm', action='store_true',
                        help='Nie pytaj o potwierdzenie')

    args = parser.parse_args()

    # Sprawdź czy folder główny istnieje
    if not os.path.exists(args.input_root):
        print(f"❌ Błąd: Folder {args.input_root} nie istnieje!")
        sys.exit(1)

    # Utwórz folder wyjściowy
    os.makedirs(args.output_root, exist_ok=True)

    # Uruchom batch processor
    try:
        processor = BatchProcessor(args)
        processor.run()
    except KeyboardInterrupt:
        print("\n⏹️  Przerwano przez użytkownika")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()