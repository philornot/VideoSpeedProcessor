#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URUCHOM MNIE JEŚLI MASZ PROBLEM Z ImageMagick
Naprawa ImageMagick - modyfikuje bezpośrednio video_processor.py
"""

import os
import re
import subprocess
from pathlib import Path


def find_imagemagick():
    """Agresywnie znajduje ImageMagick."""
    print("🔍 Szukam ImageMagick wszędzie...")

    candidates = []

    # 1. Sprawdź PATH
    try:
        result = subprocess.run(['where', 'magick'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            for path in result.stdout.strip().split('\n'):
                if path.strip() and os.path.exists(path.strip()):
                    candidates.append(path.strip())
                    print(f"   ✅ PATH: {path.strip()}")
    except:
        pass

    # 2. Szukaj w Program Files
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"D:\Program Files",
        r"D:\Program Files (x86)"
    ]

    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            try:
                for item in os.listdir(base_dir):
                    if 'imagemagick' in item.lower():
                        magick_exe = os.path.join(base_dir, item, 'magick.exe')
                        if os.path.exists(magick_exe):
                            candidates.append(magick_exe)
                            print(f"   ✅ Znaleziono: {magick_exe}")
            except:
                pass

    # 3. Sprawdź typowe lokalizacje
    typical_paths = [
        r"C:\ImageMagick\magick.exe",
        r"C:\Tools\ImageMagick\magick.exe",
        r"C:\magick.exe"
    ]

    for path in typical_paths:
        if os.path.exists(path):
            candidates.append(path)
            print(f"   ✅ Typowa lokalizacja: {path}")

    if candidates:
        # Wybierz pierwszy działający
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, '-version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"🎯 Wybrany: {candidate}")
                    return candidate
            except:
                continue

    print("❌ Nie znaleziono działającego ImageMagick")
    return None


def modify_video_processor():
    """Modyfikuje video_processor.py aby na pewno używał ImageMagick."""

    magick_path = find_imagemagick()
    if not magick_path:
        print("❌ Nie można naprawić - brak ImageMagick")
        return False

    print(f"🔧 Modyfikuję video_processor.py...")

    # Przeczytaj obecny plik
    with open('video_processor.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Znajdź linię z importami zewnętrznymi
    import_section = '''# Importy zewnętrzne
try:
    import moviepy.editor as mp
    from moviepy.video.tools.drawing import color_gradient
except ImportError:
    print("Błąd: Brak biblioteki moviepy. Zainstaluj: pip install moviepy")
    sys.exit(1)'''

    # Nowa sekcja z konfiguracją ImageMagick
    new_import_section = f'''# Konfiguracja ImageMagick - HARDCODED
import os
os.environ['IMAGEMAGICK_BINARY'] = r"{magick_path}"

# Importy zewnętrzne
try:
    import moviepy.editor as mp
    from moviepy.video.tools.drawing import color_gradient

    # Ustaw konfigurację MoviePy bezpośrednio
    import moviepy.config as mp_config
    mp_config.IMAGEMAGICK_BINARY = r"{magick_path}"

except ImportError:
    print("Błąd: Brak biblioteki moviepy. Zainstaluj: pip install moviepy")
    sys.exit(1)'''

    # Zastąp sekcję importów
    if import_section in content:
        content = content.replace(import_section, new_import_section)
        print("✅ Zmodyfikowano sekcję importów")
    else:
        print("⚠️  Nie znaleziono sekcji importów - dodaję na początku")
        # Dodaj na początku po shebang
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip():
                insert_pos = i
                break

        lines.insert(insert_pos, f'# Konfiguracja ImageMagick - HARDCODED')
        lines.insert(insert_pos + 1, f'import os')
        lines.insert(insert_pos + 2, f'os.environ["IMAGEMAGICK_BINARY"] = r"{magick_path}"')
        lines.insert(insert_pos + 3, '')
        content = '\n'.join(lines)

    # Zmodyfikuj funkcję check_imagemagick()
    old_check = '''def check_imagemagick():
    """Sprawdza czy ImageMagick jest dostępny."""
    try:
        import moviepy.editor as mp
        # Test czy TextClip działa
        test_clip = mp.TextClip("test", fontsize=20, color='white').set_duration(0.1)
        test_clip.close()
        return True
    except Exception:
        return False'''

    new_check = f'''def check_imagemagick():
    """Sprawdza czy ImageMagick jest dostępny."""
    try:
        # Wymuś użycie naszej ścieżki
        os.environ['IMAGEMAGICK_BINARY'] = r"{magick_path}"

        import moviepy.editor as mp
        import moviepy.config as mp_config
        mp_config.IMAGEMAGICK_BINARY = r"{magick_path}"

        # Test czy TextClip działa
        test_clip = mp.TextClip("test", fontsize=20, color='white').set_duration(0.1)
        test_clip.close()
        return True
    except Exception as e:
        print(f"ImageMagick test failed: {{e}}")
        return False'''

    if old_check in content:
        content = content.replace(old_check, new_check)
        print("✅ Zmodyfikowano funkcję check_imagemagick()")

    # Zapisz zmodyfikowany plik
    with open('video_processor_fixed.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Utworzono video_processor_fixed.py")
    return True


def create_simple_test():
    """Tworzy prosty test ImageMagick."""
    magick_path = find_imagemagick()
    if not magick_path:
        return

    test_code = f'''#!/usr/bin/env python3
import os
os.environ['IMAGEMAGICK_BINARY'] = r"{magick_path}"

try:
    import moviepy.editor as mp
    import moviepy.config as mp_config
    mp_config.IMAGEMAGICK_BINARY = r"{magick_path}"

    print("🧪 Testuję ImageMagick...")
    print(f"📍 Ścieżka: {magick_path}")

    # Test TextClip
    txt = mp.TextClip("x3.0", fontsize=50, color='white', font='Arial')
    txt = txt.set_duration(1)

    print("✅ TextClip działa!")
    txt.close()

except Exception as e:
    print(f"❌ Błąd: {{e}}")
'''

    with open('test_imagemagick.py', 'w', encoding='utf-8') as f:
        f.write(test_code)

    print("📝 Utworzono test_imagemagick.py")


def main():
    """Główna funkcja."""
    print("🛠️  NAPRAWA IMAGEMAGICK")
    print("=" * 50)

    # Znajdź ImageMagick
    magick_path = find_imagemagick()
    if not magick_path:
        print("\n❌ KRYTYCZNY BŁĄD: Nie znaleziono ImageMagick!")
        print("\n🔧 ROZWIĄZANIA:")
        print("1. Reinstaluj ImageMagick: https://imagemagick.org/script/download.php#windows")
        print("2. Podczas instalacji zaznacz 'Install development headers'")
        print("3. Upewnij się, że jest w PATH")
        print("4. Uruchom PowerShell jako Administrator i sprawdź: magick -version")
        return

    # Modyfikuj video_processor
    if modify_video_processor():
        print("\n🎉 NAPRAWA ZAKOŃCZONA!")
        print("✅ Utworzono video_processor_fixed.py z hardcoded ImageMagick")

        # Utwórz test
        create_simple_test()

        print("\n🚀 NASTĘPNE KROKI:")
        print("1. Przetestuj: python test_imagemagick.py")
        print("2. Uruchom: python video_processor_fixed.py --input_folder \".\" --speed_multiplier 3.0")
        print("\n🎯 Jeśli nadal nie działa, ImageMagick może być niepoprawnie zainstalowany")
    else:
        print("❌ Naprawa nie powiodła się")


if __name__ == "__main__":
    main()