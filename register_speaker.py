#!/usr/bin/env python3
"""
Rejestracja nowego mówcy w systemie z opcją nagrywania z mikrofonu
Z MOŻLIWOŚCIĄ UŻYCIA SEPARATORA i PODGLĄDEM DŹWIĘKU
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
from asteroid.models import ConvTasNet
import warnings
warnings.filterwarnings('ignore')

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

class AudioSeparator:
    """Klasa do separacji audio - tak jak w systemie głównym"""
    def __init__(self, separation_model_path=None, device='cpu'):
        self.device = device
        self.model = None
        self.sample_rate = 16000
        self.debug = False
        self.model_input_size = 16384  # ConvTasNet standard
        
        if separation_model_path and os.path.exists(separation_model_path):
            print("🔧 Ładowanie modelu separacji...")
            try:
                checkpoint = torch.load(separation_model_path, map_location=device)
                print(f"  Wczytywanie checkpointu...")
                
                # Tworzymy model ConvTasNet z 2 źródłami (bo sep_model2.pt)
                self.model = ConvTasNet(n_src=2).to(device)
                
                # Ładujemy stan modelu
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                        print(f"  Załadowano 'state_dict'")
                    elif 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"  Załadowano 'model_state_dict'")
                    else:
                        # Może to być bezpośrednio state_dict
                        try:
                            self.model.load_state_dict(checkpoint)
                            print(f"  Załadowano bezpośredni state_dict")
                        except Exception as e2:
                            print(f"  Błąd ładowania: {e2}")
                            # Spróbujmy zmapować klucze
                            new_state_dict = {}
                            for k, v in checkpoint.items():
                                if k.startswith('module.'):
                                    new_state_dict[k[7:]] = v
                                else:
                                    new_state_dict[k] = v
                            self.model.load_state_dict(new_state_dict)
                            print(f"  Załadowano po mapowaniu kluczy")
                else:
                    # To może być bezpośrednio state_dict
                    self.model.load_state_dict(checkpoint)
                    print(f"  Załadowano bezpośredni state_dict")
                
                self.model.eval()
                print("✅ Model separacji załadowany")
                
                # Test prostego forward pass
                with torch.no_grad():
                    test_input = torch.randn(1, 16384).to(device)
                    test_output = self.model(test_input)
                    print(f"  Test modelu: input {test_input.shape} -> output {test_output.shape}")
                
            except Exception as e:
                print(f"❌ Błąd ładowania modelu separacji: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
        else:
            print("ℹ️ Model separacji nieznaleziony - użycie czystego audio")
            self.model = None
    
    def set_debug(self, debug=True):
        """Włącza/wyłącza tryb debug"""
        self.debug = debug
    
    def separate_audio_short(self, audio_16k):
        """Separuje krótkie audio (do 1 sekundy)"""
        if self.model is None:
            return audio_16k
        
        try:
            with torch.no_grad():
                original_length = len(audio_16k)
                
                # Przygotuj input
                if original_length < self.model_input_size:
                    # Pad do wymaganej długości
                    padding = self.model_input_size - original_length
                    audio_padded = np.pad(audio_16k, (0, padding), mode='constant')
                    if self.debug:
                        print(f"  [SEPARATOR-SHORT] Padded from {original_length} to {len(audio_padded)}")
                elif original_length > self.model_input_size:
                    # Weź początek
                    audio_padded = audio_16k[:self.model_input_size]
                    if self.debug:
                        print(f"  [SEPARATOR-SHORT] Cropped from {original_length} to {len(audio_padded)}")
                else:
                    audio_padded = audio_16k
                
                # Konwertuj do tensora
                audio_tensor = torch.from_numpy(audio_padded).float()
                audio_tensor = audio_tensor.unsqueeze(0).to(self.device)  # (1, samples)
                
                if self.debug:
                    print(f"  [SEPARATOR-SHORT] Input tensor shape: {audio_tensor.shape}")
                
                # Separacja
                separated = self.model(audio_tensor)  # (batch, n_src, samples)
                
                if self.debug:
                    print(f"  [SEPARATOR-SHORT] Output shape: {separated.shape}")
                
                separated = separated.squeeze(0)  # (n_src, samples)
                
                # Wybierz najgłośniejsze źródło
                energies = []
                for i in range(separated.shape[0]):
                    energy = torch.mean(torch.abs(separated[i]))
                    energies.append(energy.item())
                
                source_idx = np.argmax(energies)
                separated_audio = separated[source_idx].cpu().numpy()
                
                # Przyciąć do oryginalnej długości
                separated_audio = separated_audio[:original_length]
                
                # Normalizuj
                if len(separated_audio) > 0 and np.max(np.abs(separated_audio)) > 0:
                    separated_audio = separated_audio / np.max(np.abs(separated_audio))
                
                return separated_audio
                
        except Exception as e:
            if self.debug:
                print(f"❌ Błąd separacji krótkiego audio: {e}")
            return audio_16k
    
    def separate_audio_long(self, audio_16k):
        """Separuje długie audio (> 1 sekundy) - lepsza jakość"""
        if self.model is None:
            return audio_16k
        
        try:
            with torch.no_grad():
                original_length = len(audio_16k)
                
                # Dla dłuższych nagrań, przetwarzaj segmentami
                segment_size = self.model_input_size
                hop_size = segment_size // 2  # 50% overlap
                
                segments = []
                for start in range(0, original_length - segment_size + 1, hop_size):
                    segment = audio_16k[start:start + segment_size]
                    
                    # Konwertuj do tensora
                    audio_tensor = torch.from_numpy(segment).float()
                    audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
                    
                    # Separacja
                    separated = self.model(audio_tensor)
                    separated = separated.squeeze(0)
                    
                    # Wybierz najgłośniejsze źródło
                    energies = []
                    for i in range(separated.shape[0]):
                        energy = torch.mean(torch.abs(separated[i]))
                        energies.append(energy.item())
                    
                    source_idx = np.argmax(energies)
                    separated_segment = separated[source_idx].cpu().numpy()
                    
                    segments.append(separated_segment)
                
                # Złożenie segmentów z overlap-add
                if segments:
                    # Proste złożenie - weź środek każdego segmentu
                    result = np.zeros(original_length)
                    weights = np.zeros(original_length)
                    
                    for i, (start, segment) in enumerate(zip(range(0, original_length - segment_size + 1, hop_size), segments)):
                        end = start + segment_size
                        
                        # Okno Hanninga dla płynnego złożenia
                        window = np.hanning(segment_size)
                        segment_windowed = segment * window
                        
                        # Dodaj do wyniku
                        result[start:end] += segment_windowed
                        weights[start:end] += window
                    
                    # Normalizuj przez sumę wag
                    result = np.where(weights > 0, result / weights, result)
                    
                    # Przyciąć do oryginalnej długości
                    result = result[:original_length]
                    
                    # Normalizuj amplitudę
                    if np.max(np.abs(result)) > 0:
                        result = result / np.max(np.abs(result))
                    
                    return result
                else:
                    return audio_16k
                
        except Exception as e:
            if self.debug:
                print(f"❌ Błąd separacji długiego audio: {e}")
            return audio_16k
    
    def separate_audio(self, audio_16k):
        """Inteligentna separacja - wybiera metodę w zależności od długości"""
        if self.model is None:
            return audio_16k
        
        # Dla krótkich fragmentów (np. preview) używamy prostszej metody
        if len(audio_16k) < self.model_input_size * 2:  # Mniej niż 2 sekundy
            return self.separate_audio_short(audio_16k)
        else:
            return self.separate_audio_long(audio_16k)

class MicrophoneRecorder:
    """Nagrywanie z mikrofonu z opcją przetwarzania przez separator"""
    
    def __init__(self, sample_rate=16000, separator=None, debug=False):
        self.sample_rate = sample_rate
        self.separator = separator
        self.recording = False
        self.audio_data = []
        self.processed_audio_data = []  # Audio po przetworzeniu przez separator
        self.input_queue = queue.Queue()
        self.is_running = True
        self.preview_queue = queue.Queue(maxsize=5)
        self.last_print_time = 0
        self.print_interval = 1.0  # Wydrukuj co 1 sekundę
        self.debug = debug
        
        # Bufor dla gromadzenia chunków do separacji
        self.separation_buffer = []
        self.buffer_size = 16384  # 1 sekunda dla separatora
        
        if separator and separator.model is not None:
            print("🔧 Audio będzie przetwarzane przez separator (jak w systemie głównym)")
            if debug:
                separator.set_debug(True)
    
    def input_callback(self, indata, frames, time_info, status):
        """Callback wejściowy"""
        if status and self.debug:
            print(f"Input status: {status}")
        
        if self.recording:
            chunk = indata.copy().flatten()
            self.input_queue.put(chunk)
    
    def safe_print(self, message):
        """Bezpieczne drukowanie - unika błędów blokujących"""
        try:
            sys.stdout.write(message + '\n')
            sys.stdout.flush()
        except (BlockingIOError, OSError):
            pass
    
    def safe_print_progress(self, message):
        """Bezpieczne drukowanie postępu (bez newline)"""
        try:
            sys.stdout.write('\r' + message)
            sys.stdout.flush()
        except (BlockingIOError, OSError):
            pass
    
    def start_recording(self, duration_seconds=None):
        """Rozpoczyna nagrywanie z mikrofonu"""
        print(f"\n🎤 Rozpoczynam nagrywanie z mikrofonu...")
        if duration_seconds:
            print(f"⏱️  Nagrywanie potrwa {duration_seconds} sekund")
        else:
            print("⏱️  Nagrywanie do momentu naciśnięcia klawisza 's'")
        
        self.recording = True
        self.audio_data = []
        self.processed_audio_data = []
        self.separation_buffer = []
        
        # Uruchom strumień audio z większym blocksize
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self.input_callback,
                blocksize=8192  # Zwiększony blocksize
            )
            self.stream.start()
        except Exception as e:
            print(f"❌ Błąd uruchamiania strumienia: {e}")
            return None
        
        # Uruchom strumień podglądu
        self.preview_stream = None
        self.preview_thread = threading.Thread(target=self._preview_playback, daemon=True)
        self.preview_thread.start()
        
        start_time = time.time()
        input_handler = NonBlockingInput()
        
        print("\n🎙️  MÓW TERAZ...")
        print("   Naciśnij 's' aby zatrzymać nagrywanie (tylko tryb bez limitu czasu)")
        print("   Naciśnij 'p' aby odsłuchać to co trafia do modelu (podgląd)")
        print("   Naciśnij 'd' aby włączyć/wyłączyć debug")
        
        chunk_counter = 0
        separator_active = self.separator and self.separator.model is not None
        
        try:
            while self.recording and self.is_running:
                # Pobierz dane z kolejki
                chunks_processed = 0
                try:
                    while not self.input_queue.empty() and chunks_processed < 5:  # Ogranicz do 5 chunków na iterację
                        raw_chunk = self.input_queue.get_nowait()
                        
                        # Dodaj do bufora separacji
                        self.separation_buffer.extend(raw_chunk)
                        
                        # Jeśli mamy wystarczająco danych w buforze, przetwórz przez separator
                        processed_chunk = raw_chunk.copy()
                        if separator_active and len(self.separation_buffer) >= self.buffer_size:
                            # Weź próbkę z bufora
                            buffer_sample = np.array(self.separation_buffer[:self.buffer_size])
                            
                            # Przetwórz przez separator
                            separated_sample = self.separator.separate_audio_short(buffer_sample)
                            
                            # Użyj tego samego gain co oryginał
                            if np.max(np.abs(raw_chunk)) > 0 and np.max(np.abs(separated_sample[:len(raw_chunk)])) > 0:
                                gain = np.max(np.abs(raw_chunk)) / np.max(np.abs(separated_sample[:len(raw_chunk)]))
                                processed_chunk = separated_sample[:len(raw_chunk)] * gain
                            
                            # Opróżnij część bufora
                            self.separation_buffer = self.separation_buffer[self.buffer_size//2:]  # Zostaw połowę na overlap
                        
                        # Oblicz energię
                        raw_energy = 10 * np.log10(np.mean(raw_chunk**2) + 1e-10) if len(raw_chunk) > 0 else -100
                        proc_energy = 10 * np.log10(np.mean(processed_chunk**2) + 1e-10) if len(processed_chunk) > 0 else -100
                        
                        if self.debug and chunk_counter % 20 == 0:
                            print(f"  [RECORDER] Chunk {chunk_counter}: raw={raw_energy:.1f}dB, processed={proc_energy:.1f}dB, diff={proc_energy-raw_energy:.1f}dB")
                            print(f"  [RECORDER] Buffer size: {len(self.separation_buffer)}")
                        
                        self.audio_data.append(raw_chunk)
                        self.processed_audio_data.append(processed_chunk)
                        
                        # Dodaj do kolejki podglądu
                        if len(self.preview_queue.queue) < 3:  # Ogranicz do 3 chunków
                            self.preview_queue.put(processed_chunk.copy())
                        
                        chunks_processed += 1
                        chunk_counter += 1
                        
                except queue.Empty:
                    pass
                
                # Sprawdź klawisze
                key = input_handler.get_key()
                if key:
                    if key == 's' or key == 'S':
                        print("\n⏹️  Zatrzymywanie nagrywania...")
                        self.recording = False
                        break
                    elif key == 'p' or key == 'P':
                        # Odtwórz ostatni przetworzony chunk
                        if self.processed_audio_data:
                            last_chunk = self.processed_audio_data[-1]
                            if len(last_chunk) > 0 and np.max(np.abs(last_chunk)) > 0:
                                preview = last_chunk / np.max(np.abs(last_chunk))
                                self.preview_queue.put(preview)
                                energy = 10 * np.log10(np.mean(last_chunk**2) + 1e-10)
                                print(f"\n🔊 Odtworzono podgląd przetworzonego audio (energia: {energy:.1f}dB)")
                    elif key == 'd' or key == 'D':
                        self.debug = not self.debug
                        if self.separator:
                            self.separator.set_debug(self.debug)
                        print(f"\n🔧 Debug: {'WŁĄCZONY' if self.debug else 'WYŁĄCZONY'}")
                
                # Sprawdź limit czasu
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    print(f"\n⏰ Upłynął limit czasu {duration_seconds} sekund")
                    self.recording = False
                    break
                
                # Pokaż postęp (ogranicz częstotliwość drukowania)
                current_time = time.time()
                if current_time - self.last_print_time >= self.print_interval:
                    elapsed = current_time - start_time
                    if duration_seconds:
                        progress = min(1.0, elapsed / duration_seconds)
                        self.safe_print_progress(f"📊 Postęp: {progress*100:.1f}% ({elapsed:.1f}s / {duration_seconds}s) | Chunks: {chunk_counter}")
                    else:
                        self.safe_print_progress(f"⏱️  Nagrywanie: {elapsed:.1f}s | Chunks: {chunk_counter} (s-stop, p-preview, d-debug)")
                    self.last_print_time = current_time
                
                time.sleep(0.01)  # Mniejsze opóźnienie
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Nagrywanie przerwane przez użytkownika")
            self.recording = False
        except Exception as e:
            print(f"\n❌ Błąd nagrywania: {e}")
            import traceback
            traceback.print_exc()
            self.recording = False
        finally:
            # Wyczyść linię postępu
            try:
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                sys.stdout.flush()
            except:
                pass
            
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            input_handler.restore()
            
            # Zatrzymaj podgląd
            self.preview_queue.put(None)
            if self.preview_thread:
                self.preview_thread.join(timeout=1.0)
        
        # Połącz wszystkie chunki (przetworzone)
        if self.processed_audio_data:
            full_audio = np.concatenate(self.processed_audio_data)
            recording_length = len(full_audio) / self.sample_rate
            
            # PRZETWORZ CAŁOŚĆ PRZEZ SEPARATOR dla lepszej jakości
            if self.separator and self.separator.model is not None and recording_length > 1.0:
                print("\n🔧 Końcowe przetwarzanie całego nagrania przez separator...")
                full_audio = self.separator.separate_audio_long(full_audio)
                print("✅ Przetworzono całe nagranie")
            
            # Oblicz statystyki
            if self.audio_data:
                raw_full = np.concatenate(self.audio_data)
                raw_energy = 10 * np.log10(np.mean(raw_full**2) + 1e-10)
                proc_energy = 10 * np.log10(np.mean(full_audio**2) + 1e-10)
                
                print(f"\n✅ Nagrano {recording_length:.2f} sekund audio")
                print(f"   Energia surowego: {raw_energy:.1f}dB")
                print(f"   Energia przetworzonego: {proc_energy:.1f}dB")
                print(f"   Różnica: {proc_energy - raw_energy:.1f}dB")
            
            # Zapisz również surowe audio dla porównania
            if self.audio_data:
                raw_audio = np.concatenate(self.audio_data)
                raw_audio_path = "./temp_raw_recording.wav"
                import soundfile as sf
                sf.write(raw_audio_path, raw_audio, self.sample_rate)
                print(f"💾 Surowe nagranie zapisane jako: {raw_audio_path}")
                
                # Zapisz również przetworzone audio
                proc_audio_path = "./temp_processed_recording.wav"
                sf.write(proc_audio_path, full_audio, self.sample_rate)
                print(f"💾 Przetworzone nagranie zapisane jako: {proc_audio_path}")
            
            return full_audio
        else:
            print("\n❌ Nie nagrano żadnych danych")
            return None
    
    def _preview_playback(self):
        """Wątek odtwarzania podglądu"""
        try:
            while self.is_running:
                try:
                    audio = self.preview_queue.get(timeout=0.5)
                    if audio is None:  # Sygnał do zatrzymania
                        break
                    
                    # Odtwórz audio
                    if len(audio) > 0 and np.max(np.abs(audio)) > 0:
                        audio_normalized = audio / np.max(np.abs(audio))
                        
                        # Utwórz strumień jeśli nie istnieje
                        if self.preview_stream is None:
                            self.preview_stream = sd.OutputStream(
                                samplerate=self.sample_rate,
                                channels=1,
                                dtype='float32',
                                blocksize=1024
                            )
                            try:
                                self.preview_stream.start()
                            except Exception as e:
                                if self.debug:
                                    print(f"  [PREVIEW] Stream error: {e}")
                        
                        if self.preview_stream and self.preview_stream.active:
                            try:
                                self.preview_stream.write(audio_normalized.reshape(-1, 1))
                            except Exception as e:
                                if "Stream is stopped" not in str(e) and self.debug:
                                    print(f"  [PREVIEW] Write error: {e}")
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.debug:
                        print(f"  [PREVIEW] Error: {e}")
        finally:
            if self.preview_stream:
                try:
                    self.preview_stream.stop()
                    self.preview_stream.close()
                except:
                    pass
    
    def stop(self):
        """Zatrzymuje nagrywanie"""
        self.is_running = False
        self.recording = False

def register_speaker(name, audio_segments, model, separator=None, device='cpu', min_segments=3):
    """Rejestruje nowego mówcę z opcją użycia separatora"""
    
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
    
    for i, audio_segment in enumerate(audio_segments[:min_segments*3]):
        segment_length = len(audio_segment) / 16000
        print(f"  🎧 Przetwarzanie segment {i+1}: {segment_length:.2f}s")
        
        # Użyj separatora jeśli dostępny (dla spójności z systemem głównym)
        if separator and separator.model is not None:
            print("    🎛️  Przetwarzanie przez separator...")
            processed_segment = separator.separate_audio_long(audio_segment)
            
            # Zapisz próbkę dla weryfikacji
            if i < 3:  # Zapisz tylko 3 pierwsze próbki
                import soundfile as sf
                os.makedirs(f"./speaker_samples/{new_id}", exist_ok=True)
                sample_path = f"./speaker_samples/{new_id}/sample_{i}_separated.wav"
                sf.write(sample_path, processed_segment, 16000)
                print(f"      💾 Zapisano próbkę: {sample_path}")
            
            audio_segment = processed_segment
        
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
    
    # Uśrednij embeddingi
    if len(embeddings) > 1:
        stacked = torch.stack(embeddings, dim=0)
        final_embedding = torch.mean(stacked, dim=0)
    else:
        final_embedding = embeddings[0]
    
    # Normalizuj
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
    
    # Zapisz próbki audio dla weryfikacji
    import soundfile as sf
    samples_dir = f"./speaker_samples/{new_id}_{name}"
    os.makedirs(samples_dir, exist_ok=True)
    
    for i, segment in enumerate(audio_segments[:3]):
        sample_path = os.path.join(samples_dir, f"final_sample_{i}.wav")
        sf.write(sample_path, segment, 16000)
    
    print(f"   Próbki audio zapisane w: {samples_dir}")
    
    return True

def main():
    print("=" * 60)
    print("🎤 REJESTRACJA NOWEGO MÓWCY")
    print("=" * 60)
    
    # Sprawdź CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")
    
    # Załaduj model rozpoznawania
    model_path = "./speaker_models/final_model.pt"
    if not os.path.exists(model_path):
        print("❌ Model nie znaleziony!")
        print("   Uruchom najpierw trening: python train_recognizer.py")
        return
    
    print("📦 Wczytywanie modelu rozpoznawania...")
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
    
    # Załaduj separator (jeśli dostępny)
    separation_model_path = "./sep_model2.pt"
    if not os.path.exists(separation_model_path):
        print(f"⚠️ Model separacji nie znaleziony: {separation_model_path}")
        print("   Sprawdzam alternatywne lokalizacje...")
        alt_paths = ["./sep_model.pt", "./sep_model_new.pt"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                separation_model_path = alt_path
                print(f"   Znaleziono: {alt_path}")
                break
    
    separator = AudioSeparator(separation_model_path, device)
    
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
        
        # Pytanie czy używać separatora
        if separator.model is not None:
            use_separator = input("\n🔧 Używać separatora podczas nagrywania? (t/n): ").strip().lower()
            use_sep = (use_separator == 't')
        else:
            use_sep = False
            print("ℹ️  Separator niedostępny - użycie czystego audio")
        
        # Pytanie o debug
        debug_mode = input("\n🐛 Włączyć tryb debug? (t/n): ").strip().lower()
        debug = (debug_mode == 't')
        
        recorder = MicrophoneRecorder(
            sample_rate=16000,
            separator=separator if use_sep else None,
            debug=debug
        )
        
        print("\n📋 Wybierz tryb nagrywania:")
        print("   1. Nagrywanie przez określoną liczbę sekund")
        print("   2. Nagrywanie do momentu naciśnięcia klawisza")
        
        mode_choice = input("\nWybierz opcję (1-2): ").strip()
        
        try:
            if mode_choice == '1':
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
                print("   Naciśnij 'd' podczas nagrywania aby włączyć/wyłączyć debug")
                time.sleep(2)
                
                # Nagraj audio
                full_audio = recorder.start_recording(duration_seconds=duration)
                
            elif mode_choice == '2':
                print("\n🔴 Rozpoczynam nagrywanie...")
                print("   PRZYGOTUJ SIĘ DO MÓWIENIA")
                print("   Naciśnij 's' gdy skończysz mówić")
                print("   Naciśnij 'p' aby odsłuchać podgląd (to co trafi do modelu)")
                print("   Naciśnij 'd' aby włączyć/wyłączyć debug")
                time.sleep(2)
                
                # Nagraj audio
                full_audio = recorder.start_recording(duration_seconds=None)
                
            else:
                print("❌ Nieprawidłowy wybór")
                return
            
            if full_audio is None:
                return None
            
            # Podziel na segmenty 3-sekundowe
            segment_length = 3 * 16000  # 3 sekundy
            segments = []
            
            # Przesuń okno co 1.5 sekundy (50% overlap)
            hop_length = int(1.5 * 16000)
            
            for start in range(0, len(full_audio) - segment_length + 1, hop_length):
                segment = full_audio[start:start + segment_length]
                segments.append(segment)
            
            print(f"\n✂️  Podzielono nagranie na {len(segments)} segmentów po 3s")
            
            # Normalizuj każdy segment
            for i in range(len(segments)):
                if np.max(np.abs(segments[i])) > 0:
                    segments[i] = segments[i] / np.max(np.abs(segments[i]))
            
            audio_segments = segments
            
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
        
    elif source_choice == '2':
        # Wczytywanie z plików
        print("\n💾 TRYW WCZYTYWANIA Z PLIKÓW")
        
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
            print("\n   Podaj ścieżki do plików audio (oddzielone spacją):")
            files_input = input("   > ").strip()
            audio_files = files_input.split()
            
            audio_files = [f for f in audio_files if os.path.exists(f)]
            if not audio_files:
                print("❌ Żaden z podanych plików nie istnieje")
                return
        
        # Pytanie czy używać separatora
        if separator.model is not None:
            use_separator = input("\n🔧 Przetworzyć pliki przez separator? (t/n): ").strip().lower()
            use_sep = (use_separator == 't')
        else:
            use_sep = False
        
        # Wczytaj audio z plików
        audio_segments = []
        for audio_file in audio_files:
            audio = load_audio(audio_file)
            if audio is not None:
                # Przetwórz przez separator jeśli wybrano
                if use_sep:
                    print(f"🔧 Przetwarzanie {audio_file} przez separator...")
                    audio = separator.separate_audio_long(audio)
                
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
    success = register_speaker(speaker_name, audio_segments, model, separator, device)
    
    if success:
        print("\n" + "="*50)
        print("🎉 REJESTRACJA ZAKOŃCZONA SUKCESEM!")
        print("="*50)
        
        # Odsłuchaj przykładowe próbki
        try:
            print("\n🎧 Czy chcesz odsłuchać zarejestrowane próbki?")
            listen = input("   (t - tak, n - nie): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            listen = 'n'
            print("\n")
        
        if listen == 't' and audio_segments:
            print("🔊 Odtwarzanie 3 pierwszych próbek...")
            try:
                for i, segment in enumerate(audio_segments[:3]):
                    if len(segment) > 0 and np.max(np.abs(segment)) > 0:
                        normalized = segment / np.max(np.abs(segment))
                        print(f"   Próbka {i+1} ({len(segment)/16000:.2f}s)...")
                        sd.play(normalized, 16000)
                        sd.wait()
                        time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n⏹️  Przerwano odtwarzanie")
            except Exception as e:
                print(f"Błąd odtwarzania: {e}")
        
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
