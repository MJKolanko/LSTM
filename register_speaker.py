#!/usr/bin/env python3
"""
Prosty rejestrator mówcy z monitoringiem na żywo
Nagrywa audio i natychmiast odtwarza na domyślnym głośniku
WINDOWS COMPATIBLE VERSION
"""

import os
import torch
import pickle
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import threading
import queue
import time
import sys
import platform
import datetime

# Check platform
IS_WINDOWS = platform.system() == "Windows"

# Platform-specific imports
if IS_WINDOWS:
    import msvcrt
else:
    # Unix-specific imports
    import select
    import termios
    import tty
    import fcntl


# Model (taki sam jak w testach)
class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(60, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)

        self.attention = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.Softmax(dim=2)
        )

        self.fc1 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        w = self.attention(x)
        x = torch.sum(x * w, dim=2)

        x = self.dropout(self.relu(self.bn4(self.fc1(x))))
        x = self.tanh(self.fc2(x))

        x = F.normalize(x, p=2, dim=1)
        return x


def extract_features(audio, sr=16000):
    """Ekstrakcja cech MFCC - bez restrykcji"""
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20,
        n_mels=80,
        n_fft=512,
        hop_length=160
    )

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    features_tensor = torch.FloatTensor(features).unsqueeze(0)

    # Prosta normalizacja
    mean = features_tensor.mean(dim=2, keepdim=True)
    std = features_tensor.std(dim=2, keepdim=True) + 1e-8
    features_tensor = (features_tensor - mean) / std

    return features_tensor


class LiveMonitorRecorder:
    """Nagrywarka z odsłuchem na żywo"""

    def __init__(self, sample_rate=16000, monitor_gain=1.0):
        self.sample_rate = sample_rate
        self.monitor_gain = monitor_gain
        self.recording = False
        self.audio_data = []
        self.is_running = True

        # Bufor dla odsłuchu
        self.monitor_buffer = queue.Queue()

        # Strumień wyjściowy dla monitoringu
        self.output_stream = None

        print(f"🎧 Monitoring: WŁĄCZONY (wzmocnienie: {monitor_gain}x)")

    def input_callback(self, indata, frames, time_info, status):
        """Callback wejściowy - zbiera i odtwarza dźwięk"""
        if status:
            print(f"Input status: {status}")

        if self.recording:
            # Pobierz dane audio
            chunk = indata.copy().flatten()

            # Zapisz do historii
            self.audio_data.append(chunk.copy())

            # Dodaj do bufora monitoringu (z wzmocnieniem)
            if self.monitor_gain != 1.0:
                chunk = chunk * self.monitor_gain
                chunk = np.clip(chunk, -1.0, 1.0)

            self.monitor_buffer.put(chunk)

    def output_callback(self, outdata, frames, time_info, status):
        """Callback wyjściowy - odtwarza dźwięk na głośniku"""
        if status:
            print(f"Output status: {status}")

        try:
            # Pobierz dane z bufora
            chunk = self.monitor_buffer.get_nowait()

            # Upewnij się, że chunk ma właściwy rozmiar
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)), mode='constant')
            elif len(chunk) > frames:
                chunk = chunk[:frames]

            # Wyślij do głośnika
            outdata[:, 0] = chunk

        except queue.Empty:
            # Brak danych - cisza
            outdata.fill(0)

    def start_recording(self, duration_seconds=None):
        """Rozpoczyna nagrywanie z odsłuchem na żywo"""
        print(f"\n🎤 Rozpoczynam nagrywanie z monitoringiem na żywo...")

        if duration_seconds:
            print(f"⏱️  Nagrywanie potrwa {duration_seconds} sekund")
        else:
            print("⏱️  Nagrywanie do momentu naciśnięcia klawisza 's'")

        self.recording = True
        self.audio_data = []

        # Uruchom strumienie
        input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.input_callback
        )

        output_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.output_callback
        )

        input_stream.start()
        output_stream.start()

        print("\n🎙️  MÓW TERAZ...")
        print("   Słyszysz swój głos na głośniku")

        if duration_seconds:
            print("   Naciśnij 's' aby zatrzymać wcześniej")
        else:
            print("   Naciśnij 's' aby zatrzymać")

        start_time = time.time()

        # Windows implementation
        if IS_WINDOWS:
            try:
                while self.recording and self.is_running:
                    # Sprawdź klawisz
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if isinstance(key, bytes):
                            # Try to decode
                            try:
                                key = key.decode('utf-8')
                            except UnicodeDecodeError:
                                key = key.decode('latin-1')

                        if key == 's' or key == 'S':
                            print("\n⏹️  Zatrzymywanie nagrywania...")
                            self.recording = False
                            break

                    # Sprawdź limit czasu
                    if duration_seconds and (time.time() - start_time) >= duration_seconds:
                        print(f"\n⏰ Upłynął limit czasu {duration_seconds} sekund")
                        self.recording = False
                        break

                    # Pokaż postęp
                    elapsed = time.time() - start_time
                    if duration_seconds:
                        progress = min(1.0, elapsed / duration_seconds)
                        sys.stdout.write(f"\r📊 Postęp: {progress * 100:.1f}% ({elapsed:.1f}s / {duration_seconds}s)")
                    else:
                        sys.stdout.write(f"\r⏱️  Nagrywanie: {elapsed:.1f} sekund... (naciśnij 's' aby zatrzymać)")
                    sys.stdout.flush()

                    time.sleep(0.1)

            except Exception as e:
                print(f"\n❌ Błąd: {e}")

        else:  # Unix implementation
            # Konfiguracja klawiatury nieblokującej
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())

                fd = sys.stdin.fileno()
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

                while self.recording and self.is_running:
                    # Sprawdź klawisz
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key == 's' or key == 'S':
                            print("\n⏹️  Zatrzymywanie nagrywania...")
                            self.recording = False
                            break

                    # Sprawdź limit czasu
                    if duration_seconds and (time.time() - start_time) >= duration_seconds:
                        print(f"\n⏰ Upłynął limit czasu {duration_seconds} sekund")
                        self.recording = False
                        break

                    # Pokaż postęp
                    elapsed = time.time() - start_time
                    if duration_seconds:
                        progress = min(1.0, elapsed / duration_seconds)
                        sys.stdout.write(f"\r📊 Postęp: {progress * 100:.1f}% ({elapsed:.1f}s / {duration_seconds}s)")
                    else:
                        sys.stdout.write(f"\r⏱️  Nagrywanie: {elapsed:.1f} sekund... (naciśnij 's' aby zatrzymać)")
                    sys.stdout.flush()

                    time.sleep(0.1)

            finally:
                # Przywróć ustawienia terminala
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Zatrzymaj strumienie
        input_stream.stop()
        input_stream.close()
        output_stream.stop()
        output_stream.close()

        # Połącz wszystkie chunki
        if self.audio_data:
            full_audio = np.concatenate(self.audio_data)
            recording_length = len(full_audio) / self.sample_rate
            print(f"\n✅ Nagrano {recording_length:.2f} sekund audio")
            return full_audio
        else:
            print("\n❌ Nie nagrano żadnych danych")
            return None

    def stop(self):
        """Zatrzymuje nagrywanie"""
        self.is_running = False
        self.recording = False


def register_speaker_simple(name, audio_segments, model, min_segments=2):
    """Prosta rejestracja mówcy - bez weryfikacji"""

    print(f"\n📝 Rejestracja mówcy: {name}")

    if not audio_segments:
        print("❌ Brak danych audio")
        return False

    # Wczytaj lub utwórz bazę danych
    db_path = "./speaker_database.pkl"
    if os.path.exists(db_path):
        try:
            with open(db_path, 'rb') as f:
                database = pickle.load(f)
            print(f"📊 Istniejąca baza: {len(database.get('speakers', {}))} mówców")
        except:
            database = {'speakers': {}, 'speaker_names': {}}
    else:
        database = {'speakers': {}, 'speaker_names': {}}
        print("📊 Nowa baza danych utworzona")

    # Znajdź nowe ID
    existing_ids = list(database['speakers'].keys())
    if existing_ids:
        # Najprostsze ID
        new_id = str(len(existing_ids))
    else:
        new_id = "0"

    print(f"🆔 ID mówcy: {new_id}")

    # Zbierz embeddingi z segmentów
    embeddings = []
    device = next(model.parameters()).device

    print(f"\n🔍 Przetwarzanie {len(audio_segments)} segmentów...")
    for i, audio_segment in enumerate(audio_segments):
        segment_length = len(audio_segment) / 16000

        try:
            # Ekstrakcja cech
            features = extract_features(audio_segment)
            features = features.to(device)

            # Pobierz embedding
            model.eval()
            with torch.no_grad():
                embedding = model(features).squeeze(0).cpu()

            embeddings.append(embedding)
            print(f"  ✅ Segment {i + 1}: {segment_length:.1f}s - OK")

        except Exception as e:
            print(f"  ⚠️  Segment {i + 1}: błąd - {str(e)[:50]}...")
            continue

    if len(embeddings) < min_segments:
        print(f"❌ Za mało segmentów: {len(embeddings)}/{min_segments}")
        print("   Spróbuj nagrać dłuższe audio")
        return False

    # Uśrednij embeddingi
    stacked = torch.stack(embeddings, dim=0)
    final_embedding = torch.mean(stacked, dim=0)
    final_embedding = F.normalize(final_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

    # Zapisz do bazy
    database['speakers'][new_id] = final_embedding
    database['speaker_names'][new_id] = name

    # Zapisz bazę
    try:
        with open(db_path, 'wb') as f:
            pickle.dump(database, f)

        print(f"\n✅ Zarejestrowano mówcę '{name}' z ID: {new_id}")
        print(f"   Embedding: {final_embedding.shape}")
        print(f"   Użyte segmenty: {len(embeddings)}")

        # Tworzenie kopii zapasowej
        backup_path = f"./speaker_database_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(backup_path, 'wb') as f:
            pickle.dump(database, f)

        return True

    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")
        return False


def record_with_monitoring(min_duration=5, target_segment_duration=3):
    """Nagrywa z mikrofonu z odsłuchem na żywo"""

    print("\n" + "=" * 50)
    print("🎤 NAGRYWANIE Z MONITORINGIEM NA ŻYWO")
    print("=" * 50)

    # Wybór trybu nagrywania
    print("\n📋 Wybierz tryb nagrywania:")
    print("   1. Nagrywanie przez określoną liczbę sekund")
    print("   2. Nagrywanie do momentu naciśnięcia klawisza")
    print("   3. Test odsłuchu (bez zapisywania)")

    choice = input("\nWybierz opcję (1-3): ").strip()

    # Wybór wzmocnienia monitoringu
    gain_choice = input("\n🎚️  Wzmocnienie monitoringu (domyślnie 1.0): ").strip()
    monitor_gain = float(gain_choice) if gain_choice else 1.0

    recorder = LiveMonitorRecorder(sample_rate=16000, monitor_gain=monitor_gain)

    try:
        if choice == '1':
            # Tryb z limitem czasu
            while True:
                try:
                    duration = int(input("\n⌛ Podaj czas nagrywania w sekundach (minimum 5): ").strip())
                    if duration >= 5:
                        break
                    else:
                        print("❌ Czas musi być co najmniej 5 sekund")
                except ValueError:
                    print("❌ Wprowadź prawidłową liczbę")

            print(f"\n🔴 Rozpoczynam nagrywanie na {duration} sekund...")
            print("   Słyszysz swój głos na głośniku")
            time.sleep(2)

            # Nagraj audio
            full_audio = recorder.start_recording(duration_seconds=duration)

        elif choice == '2':
            # Tryb bez limitu czasu
            print("\n🔴 Rozpoczynam nagrywanie...")
            print("   Słyszysz swój głos na głośniku")
            print("   Naciśnij 's' gdy skończysz mówić")
            print("   Minimalny czas: 5 sekund")
            time.sleep(2)

            # Nagraj audio
            full_audio = recorder.start_recording(duration_seconds=None)

        elif choice == '3':
            # Tryb testowy - tylko odsłuch
            print("\n🔊 TRYB TESTOWY - TYLKO ODSŁUCH")
            print("   Sprawdzasz działanie mikrofonu i głośnika")
            print("   Mów do mikrofonu - słyszysz się na głośniku")
            print("   Naciśnij 's' aby zakończyć test")
            print("   Nic nie jest zapisywane!")

            input("\nNaciśnij Enter aby rozpocząć test...")

            # Uruchom tylko monitoring
            recorder.recording = True
            recorder.is_running = True

            # Konfiguracja strumieni
            input_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=recorder.input_callback
            )

            output_stream = sd.OutputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=recorder.output_callback
            )

            input_stream.start()
            output_stream.start()

            print("\n🎧 TEST ODSŁUCHU ROZPOCZĘTY")
            print("   Mów do mikrofonu...")
            print("   Naciśnij 's' aby zakończyć")

            # Windows implementation
            if IS_WINDOWS:
                try:
                    while recorder.is_running:
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if isinstance(key, bytes):
                                try:
                                    key = key.decode('utf-8')
                                except UnicodeDecodeError:
                                    key = key.decode('latin-1')

                            if key == 's' or key == 'S':
                                print("\n⏹️  Zakończono test odsłuchu")
                                break

                        time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Błąd: {e}")
            else:  # Unix implementation
                # Prosta pętla z klawiszem
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())

                    while recorder.is_running:
                        if select.select([sys.stdin], [], [], 0)[0]:
                            key = sys.stdin.read(1)
                            if key == 's' or key == 'S':
                                print("\n⏹️  Zakończono test odsłuchu")
                                break

                        time.sleep(0.1)

                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()

            print("\n✅ Test zakończony")
            return None

        else:
            print("❌ Nieprawidłowy wybór")
            return None

        if full_audio is None:
            return None

        # Podziel na segmenty 3-sekundowe
        segment_length = target_segment_duration * 16000
        segments = []

        # Przesuń okno co 1.5 sekundy (50% overlap)
        hop_length = int(1.5 * 16000)

        for start in range(0, len(full_audio) - segment_length + 1, hop_length):
            segment = full_audio[start:start + segment_length]
            segments.append(segment)

        print(f"\n✂️  Podzielono nagranie na {len(segments)} segmentów")

        return segments

    except KeyboardInterrupt:
        print("\n\n⏹️  Nagrywanie przerwane przez użytkownika")
        return None
    except Exception as e:
        print(f"❌ Błąd nagrywania: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        recorder.stop()


def load_audio_file(filepath):
    """Proste wczytywanie pliku audio"""
    try:
        audio, sr = librosa.load(filepath, sr=16000, mono=True)

        # Normalizacja
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio
    except Exception as e:
        print(f"Błąd wczytywania {filepath}: {e}")
        return None


def main():
    print("=" * 60)
    print("🎤 PROSTA REJESTRACJA MÓWCY Z MONITORINGIEM")
    print("=" * 60)
    print(f"📱 Platforma: {platform.system()} ({'Windows' if IS_WINDOWS else 'Unix/Linux'})")

    # Sprawdź CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")

    # Załaduj model
    model_path = "./speaker_models/final_model.pt"
    if not os.path.exists(model_path):
        print("⚠️  Model nie znaleziony - tworzę nowy...")
        model = SpeakerEncoder().to(device)
        print("✅ Nowy model utworzony")
    else:
        print("📦 Wczytywanie modelu...")
        model = SpeakerEncoder().to(device)

        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model załadowany")
        except Exception as e:
            print(f"⚠️  Błąd wczytywania - używam nowego modelu: {e}")

    # Pytanie o dane
    print("\n📝 Podaj dane mówcy:")
    speaker_name = input("   Imię/nazwa mówcy: ").strip()

    if not speaker_name:
        speaker_name = "Unknown_Speaker"
        print(f"   Ustawiono domyślną nazwę: {speaker_name}")

    # Wybór źródła audio
    print("\n📋 Wybierz źródło audio:")
    print("   1. Nagraj z mikrofonu (z odsłuchem na żywo)")
    print("   2. Wczytaj z pliku")
    print("   3. Test odsłuchu (tylko sprawdzenie mikrofonu/głośnika)")

    source_choice = input("\nWybierz opcję (1-3): ").strip()

    audio_segments = None

    if source_choice == '1':
        # Nagrywanie z mikrofonu z monitoringiem
        print("\n🎤 TRYB NAGRYWANIA Z MONITORINGIEM")
        print("   Nagraj kilka zdań w normalnym tempie")
        print("   Słyszysz swój głos na głośniku w czasie rzeczywistym")

        audio_segments = record_with_monitoring()

    elif source_choice == '2':
        # Wczytywanie z plików
        print("\n💾 TRYB WCZYTYWANIA Z PLIKU")

        # Szukaj domyślnych plików
        default_files = ["testwav.wav", "1mowca.wav", "audio.wav", "sample.wav"]
        available_files = [f for f in default_files if os.path.exists(f)]

        if available_files:
            print(f"\n📁 Znalezione pliki:")
            for i, file in enumerate(available_files, 1):
                print(f"   {i}. {file}")

            use_defaults = input("\n   Użyć któregoś z tych plików? (t/n): ").strip().lower()

            if use_defaults == 't':
                # Wybierz plik
                if len(available_files) == 1:
                    audio_files = [available_files[0]]
                else:
                    try:
                        file_num = int(input(f"   Wybierz plik (1-{len(available_files)}): ").strip())
                        audio_files = [available_files[file_num - 1]]
                    except:
                        audio_files = [available_files[0]]
            else:
                # Podaj własną ścieżkę
                file_path = input("   Podaj ścieżkę do pliku audio: ").strip()
                audio_files = [file_path]
        else:
            # Podaj własną ścieżkę
            file_path = input("   Podaj ścieżkę do pliku audio: ").strip()
            audio_files = [file_path]

        # Wczytaj audio z plików
        audio_segments = []
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                audio = load_audio_file(audio_file)
                if audio is not None:
                    audio_segments.append(audio)
                    print(f"✅ Wczytano: {audio_file} ({len(audio) / 16000:.2f}s)")
                else:
                    print(f"❌ Błąd wczytywania: {audio_file}")
            else:
                print(f"❌ Plik nie istnieje: {audio_file}")

        if not audio_segments:
            print("❌ Nie wczytano żadnych plików")
            return

    elif source_choice == '3':
        # Tylko test odsłuchu
        print("\n🔊 TRYB TESTOWY")
        print("   Sprawdzasz tylko działanie mikrofonu i głośnika")
        print("   Nic nie jest zapisywane!")

        record_with_monitoring()  # To wywoła tryb testowy
        print("\n✅ Test zakończony")
        return

    else:
        print("❌ Nieprawidłowy wybór")
        return

    if not audio_segments:
        print("❌ Brak danych audio")
        return

    # Rejestruj mówcę
    print("\n🔄 Przetwarzanie danych audio...")
    success = register_speaker_simple(speaker_name, audio_segments, model)

    if success:
        print("\n" + "=" * 50)
        print("🎉 REJESTRACJA ZAKOŃCZONA!")
        print("=" * 50)
        print("\n📋 Co dalej:")
        print("   1. Uruchom Main.py aby użyć systemu")
        print("   2. Dodaj więcej mówców uruchamiając ten program ponownie")
    else:
        print("\n⚠️  Uwaga: Rejestracja nie powiodła się")
        print("   Spróbuj ponownie z dłuższym nagraniem")

    # Pytanie czy dodać kolejnego mówcę
    another = input("\n➕ Czy chcesz dodać kolejnego mówcę? (t/n): ").strip().lower()
    if another == 't':
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Zamykanie programu...")