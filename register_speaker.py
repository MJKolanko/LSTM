#!/usr/bin/env python3
"""
Rejestracja nowego mówcy w systemie z opcją nagrywania z mikrofonu
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

def load_audio(filepath, target_sr=16000, duration=3.0):
    """Wczytuje i przetwarza audio z pliku"""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        
        # Normalizacja
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Przycięcie/padding
        target_len = int(duration * target_sr)
        if len(audio) > target_len:
            # Losowy fragment dla lepszej generalizacji
            start = np.random.randint(0, len(audio) - target_len)
            audio = audio[start:start + target_len]
        else:
            padding = np.zeros(target_len - len(audio))
            audio = np.concatenate([audio, padding])
        
        return audio
    except Exception as e:
        print(f"Błąd wczytywania {filepath}: {e}")
        return None

def extract_features(audio, sr=16000):
    """Ekstrakcja cech MFCC"""
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
    
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (60, time)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, 60, time)
    
    # Normalizacja
    features_tensor = (features_tensor - features_tensor.mean(dim=2, keepdim=True)) / \
                     (features_tensor.std(dim=2, keepdim=True) + 1e-8)
    
    return features_tensor

class NonBlockingInput:
    """Klasa do nieblokującego odczytu z stdin"""
    def __init__(self):
        self.old_settings = None
        self.setup_nonblocking()
        
    def setup_nonblocking(self):
        """Konfiguruje stdin do nieblokującego odczytu"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        fd = sys.stdin.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    
    def restore(self):
        """Przywraca oryginalne ustawienia terminala"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Pobiera klawisz jeśli został wciśnięty, w przeciwnym razie zwraca None"""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                return char
        except Exception:
            pass
        return None

class MicrophoneRecorder:
    """Nagrywanie z mikrofonu"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.input_queue = queue.Queue()
        self.is_running = True
        
    def input_callback(self, indata, frames, time_info, status):
        """Callback wejściowy"""
        if status:
            print(f"Input status: {status}")
        
        if self.recording:
            chunk = indata.copy().flatten()
            self.input_queue.put(chunk)
    
    def start_recording(self, duration_seconds=None):
        """Rozpoczyna nagrywanie z mikrofonu"""
        print(f"\n🎤 Rozpoczynam nagrywanie z mikrofonu...")
        if duration_seconds:
            print(f"⏱️  Nagrywanie potrwa {duration_seconds} sekund")
        else:
            print("⏱️  Nagrywanie do momentu naciśnięcia klawisza 's'")
        
        self.recording = True
        self.audio_data = []
        
        # Uruchom strumień audio
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.input_callback
        )
        self.stream.start()
        
        start_time = time.time()
        input_handler = NonBlockingInput()
        
        print("\n🎙️  MÓW TERAZ...")
        print("   Naciśnij 's' aby zatrzymać nagrywanie (tylko tryb bez limitu czasu)")
        
        try:
            while self.recording and self.is_running:
                # Pobierz dane z kolejki
                try:
                    while not self.input_queue.empty():
                        chunk = self.input_queue.get_nowait()
                        self.audio_data.append(chunk)
                except queue.Empty:
                    pass
                
                # Sprawdź klawisz (tylko w trybie bez limitu czasu)
                if duration_seconds is None:
                    key = input_handler.get_key()
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
                    sys.stdout.write(f"\r📊 Postęp: {progress*100:.1f}% ({elapsed:.1f}s / {duration_seconds}s)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r⏱️  Nagrywanie: {elapsed:.1f} sekund... (naciśnij 's' aby zatrzymać)")
                    sys.stdout.flush()
                
                time.sleep(0.1)
                
        finally:
            self.stream.stop()
            self.stream.close()
            input_handler.restore()
        
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

def register_speaker(name, audio_segments, model, min_segments=3):
    """Rejestruje nowego mówcę na podstawie segmentów audio"""
    
    print(f"\n📝 Rejestracja mówcy: {name}")
    
    if not audio_segments:
        print("❌ Brak danych audio")
        return False
    
    # Wczytaj lub utwórz bazę danych
    db_path = "./speaker_database.pkl"
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        print(f"📊 Istniejąca baza: {len(database.get('speakers', {}))} mówców")
    else:
        database = {
            'speakers': {},      # ID mówcy -> embedding
            'speaker_names': {}  # ID mówcy -> nazwa
        }
        print("📊 Nowa baza danych utworzona")
    
    # Znajdź nowe ID
    existing_ids = list(database['speakers'].keys())
    if existing_ids:
        # Znajdź największe ID (zakładamy numeryczne ID)
        numeric_ids = [int(id) for id in existing_ids if id.isdigit()]
        if numeric_ids:
            new_id = str(max(numeric_ids) + 1)
        else:
            new_id = "0"
    else:
        new_id = "0"
    
    print(f"🆔 Nowe ID mówcy: {new_id}")
    
    # Zbierz embeddingi z segmentów
    embeddings = []
    device = next(model.parameters()).device
    
    for i, audio_segment in enumerate(audio_segments[:min_segments*3]):  # Maksymalnie 3x więcej niż minimum
        segment_length = len(audio_segment) / 16000
        print(f"  🎧 Przetwarzanie segment {i+1}: {segment_length:.2f}s")
        
        # Ekstrakcja cech
        features = extract_features(audio_segment)
        features = features.to(device)
        
        # Pobierz embedding
        model.eval()
        with torch.no_grad():
            embedding = model(features).squeeze(0).cpu()
        
        embeddings.append(embedding)
        print(f"    Embedding shape: {embedding.shape}")
    
    if not embeddings:
        print("❌ Nie udało się przetworzyć żadnego segmentu audio")
        return False
    
    # Uśrednij embeddingi dla lepszej reprezentacji
    if len(embeddings) > 1:
        # Stack embeddings: (n_samples, 256)
        stacked = torch.stack(embeddings, dim=0)
        # Średnia po próbkach
        final_embedding = torch.mean(stacked, dim=0)
    else:
        final_embedding = embeddings[0]
    
    # Normalizuj ponownie
    final_embedding = F.normalize(final_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    # Zapisz do bazy
    database['speakers'][new_id] = final_embedding
    database['speaker_names'][new_id] = name
    
    # Zapisz bazę
    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"\n✅ Zarejestrowano mówcę '{name}' z ID: {new_id}")
    print(f"   Embedding: {final_embedding.shape}")
    print(f"   Ilość segmentów: {len(embeddings)}")
    
    return True

def record_from_microphone(model, min_duration=10, target_segment_duration=3):
    """Nagrywa z mikrofonu i przygotowuje segmenty do rejestracji"""
    
    print("\n" + "="*50)
    print("🎤 NAGRYWANIE Z MIKROFONU")
    print("="*50)
    
    # Wybór trybu nagrywania
    print("\n📋 Wybierz tryb nagrywania:")
    print("   1. Nagrywanie przez określoną liczbę sekund")
    print("   2. Nagrywanie do momentu naciśnięcia klawisza")
    
    choice = input("\nWybierz opcję (1-2): ").strip()
    
    recorder = MicrophoneRecorder(sample_rate=16000)
    
    try:
        if choice == '1':
            # Tryb z limitem czasu
            while True:
                try:
                    duration = int(input("\n⌛ Podaj czas nagrywania w sekundach (minimum 10): ").strip())
                    if duration >= 10:
                        break
                    else:
                        print("❌ Czas musi być co najmniej 10 sekund")
                except ValueError:
                    print("❌ Wprowadź prawidłową liczbę")
            
            print(f"\n🔴 Rozpoczynam nagrywanie na {duration} sekund...")
            print("   PRZYGOTUJ SIĘ DO MÓWIENIA")
            time.sleep(2)
            
            # Nagraj audio
            full_audio = recorder.start_recording(duration_seconds=duration)
            
        elif choice == '2':
            # Tryb bez limitu czasu
            print("\n🔴 Rozpoczynam nagrywanie...")
            print("   PRZYGOTUJ SIĘ DO MÓWIENIA")
            print("   Naciśnij 's' gdy skończysz mówić")
            time.sleep(2)
            
            # Nagraj audio
            full_audio = recorder.start_recording(duration_seconds=None)
            
        else:
            print("❌ Nieprawidłowy wybór")
            return None
        
        if full_audio is None:
            return None
        
        # Podziel na segmenty 3-sekundowe
        segment_length = target_segment_duration * 16000  # 3 sekundy
        segments = []
        
        # Przesuń okno co 1.5 sekundy (50% overlap)
        hop_length = int(1.5 * 16000)
        
        for start in range(0, len(full_audio) - segment_length + 1, hop_length):
            segment = full_audio[start:start + segment_length]
            segments.append(segment)
        
        print(f"\n✂️  Podzielono nagranie na {len(segments)} segmentów po {target_segment_duration}s")
        
        # Normalizuj każdy segment
        for i in range(len(segments)):
            if np.max(np.abs(segments[i])) > 0:
                segments[i] = segments[i] / np.max(np.abs(segments[i]))
        
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

def main():
    print("=" * 60)
    print("🎤 REJESTRACJA NOWEGO MÓWCY")
    print("=" * 60)
    
    # Sprawdź CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")
    
    # Załaduj model
    model_path = "./speaker_models/final_model.pt"
    if not os.path.exists(model_path):
        print("❌ Model nie znaleziony!")
        print("   Uruchom najpierw trening: python train_recognizer.py")
        return
    
    print("📦 Wczytywanie modelu...")
    model = SpeakerEncoder().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model załadowany")
    except Exception as e:
        print(f"❌ Błąd wczytywania modelu: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Pytanie o dane
    print("\n📝 Podaj dane nowego mówcy:")
    speaker_name = input("   Imię/nazwa mówcy: ").strip()
    
    if not speaker_name:
        print("❌ Nazwa mówcy nie może być pusta")
        return
    
    # Wybór źródła audio
    print("\n📋 Wybierz źródło audio:")
    print("   1. Nagraj z mikrofonu")
    print("   2. Wczytaj z plików")
    
    source_choice = input("\nWybierz opcję (1-2): ").strip()
    
    audio_segments = None
    
    if source_choice == '1':
        # Nagrywanie z mikrofonu
        print("\n🎤 TRYB NAGRYWANIA Z MIKROFONU")
        print("   Nagraj co najmniej 10 sekund czystej mowy")
        print("   Im więcej danych, tym lepsza rejestracja")
        
        audio_segments = record_from_microphone(model)
        
    elif source_choice == '2':
        # Wczytywanie z plików
        print("\n💾 TRYW WCZYTYWANIA Z PLIKÓW")
        
        # Domyślne pliki do rejestracji
        default_files = ["testwav.wav", "1mowca.wav"]
        available_files = [f for f in default_files if os.path.exists(f)]
        
        if not available_files:
            print("❌ Brak domyślnych plików audio do rejestracji")
            print("   Utwórz pliki: testwav.wav i/lub 1mowca.wav")
            return
        
        print(f"\n📁 Dostępne pliki do rejestracji:")
        for i, file in enumerate(available_files, 1):
            print(f"   {i}. {file}")
        
        use_defaults = input("\n   Użyć tych plików? (t/n): ").strip().lower()
        
        if use_defaults == 't':
            audio_files = available_files
        else:
            # Ręczne podanie ścieżek
            print("\n   Podaj ścieżki do plików audio (oddzielone spacją):")
            files_input = input("   > ").strip()
            audio_files = files_input.split()
            
            # Sprawdź czy pliki istnieją
            audio_files = [f for f in audio_files if os.path.exists(f)]
            if not audio_files:
                print("❌ Żaden z podanych plików nie istnieje")
                return
        
        # Wczytaj audio z plików
        audio_segments = []
        for audio_file in audio_files:
            audio = load_audio(audio_file)
            if audio is not None:
                audio_segments.append(audio)
                print(f"✅ Wczytano: {audio_file} ({len(audio)/16000:.2f}s)")
    
    else:
        print("❌ Nieprawidłowy wybór")
        return
    
    if not audio_segments:
        print("❌ Brak danych audio do rejestracji")
        return
    
    # Rejestruj mówcę
    print("\n🔄 Przetwarzanie danych audio...")
    success = register_speaker(speaker_name, audio_segments, model)
    
    if success:
        print("\n" + "="*50)
        print("🎉 REJESTRACJA ZAKOŃCZONA SUKCESEM!")
        print("="*50)
        print("\n📋 Co dalej:")
        print("   1. Przetestuj rozpoznawanie: python test_speaker_recognition.py")
        print("   2. Użyj w systemie czasu rzeczywistego")
        print("   3. Jeśli chcesz dodać więcej próbek, uruchom rejestrację ponownie")
    else:
        print("\n❌ REJESTRACJA NIE POWIODŁA SIĘ")
        print("   Spróbuj ponownie z lepszej jakości audio")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Zamykanie programu...")
