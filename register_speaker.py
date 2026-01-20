#!/usr/bin/env python3
"""
GUI do rejestracji mówcy z monitoringiem na żywo
Używając PySide6 i QUiLoader
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
from pathlib import Path

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QCheckBox, QLineEdit,
    QSpinBox, QProgressBar, QListWidget, QGroupBox, QFormLayout,
    QGridLayout, QMessageBox, QFileDialog, QTabWidget
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject, Signal, QThread, Slot, QTimer, Qt
from PySide6.QtGui import QTextCursor

# Check platform
IS_WINDOWS = platform.system() == "Windows"


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
    """Ekstrakcja cech MFCC"""
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

    # Normalizacja
    mean = features_tensor.mean(dim=2, keepdim=True)
    std = features_tensor.std(dim=2, keepdim=True) + 1e-8
    features_tensor = (features_tensor - mean) / std

    return features_tensor


class RecordingThread(QThread):
    """Wątek do nagrywania audio z monitoringiem"""
    update_progress = Signal(int)
    update_status = Signal(str)
    recording_finished = Signal(list)
    recording_error = Signal(str)

    def __init__(self, device_id=None, duration=None, monitor_gain=1.0):
        super().__init__()
        self.device_id = device_id
        self.duration = duration
        self.monitor_gain = monitor_gain
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000

    def run(self):
        """Główna funkcja wątku nagrywania"""
        try:
            self.update_status.emit("🔴 Rozpoczynanie nagrywania...")
            self.audio_data = []
            self.is_recording = True

            # Bufor dla monitoringu
            monitor_buffer = queue.Queue()

            def input_callback(indata, frames, time_info, status):
                """Callback wejściowy"""
                if self.is_recording:
                    chunk = indata.copy().flatten()
                    self.audio_data.append(chunk.copy())

                    # Dodaj do monitoringu z wzmocnieniem
                    if self.monitor_gain != 1.0:
                        chunk = chunk * self.monitor_gain
                        chunk = np.clip(chunk, -1.0, 1.0)

                    monitor_buffer.put(chunk)

            def output_callback(outdata, frames, time_info, status):
                """Callback wyjściowy"""
                try:
                    chunk = monitor_buffer.get_nowait()
                    if len(chunk) < frames:
                        chunk = np.pad(chunk, (0, frames - len(chunk)), mode='constant')
                    elif len(chunk) > frames:
                        chunk = chunk[:frames]
                    outdata[:, 0] = chunk
                except queue.Empty:
                    outdata.fill(0)

            # Uruchom strumienie
            input_stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=input_callback
            )

            output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=output_callback
            )

            input_stream.start()
            output_stream.start()

            start_time = time.time()
            last_update = start_time

            while self.is_recording:
                elapsed = time.time() - start_time

                # Aktualizuj postęp
                if time.time() - last_update > 0.1:
                    if self.duration:
                        progress = min(100, int((elapsed / self.duration) * 100))
                        self.update_progress.emit(progress)
                        self.update_status.emit(f"Nagrywanie: {elapsed:.1f}s / {self.duration}s")
                    else:
                        self.update_status.emit(f"Nagrywanie: {elapsed:.1f}s (naciśnij STOP)")

                    last_update = time.time()

                # Sprawdź czy upłynął czas
                if self.duration and elapsed >= self.duration:
                    self.is_recording = False
                    break

                time.sleep(0.05)

            # Zatrzymaj strumienie
            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()

            # Połącz dane
            if self.audio_data:
                full_audio = np.concatenate(self.audio_data)
                duration = len(full_audio) / self.sample_rate
                self.update_status.emit(f"✅ Nagrano {duration:.1f} sekund audio")

                # Podziel na segmenty (3 sekundy z 50% overlap)
                segment_length = 3 * self.sample_rate
                hop_length = int(1.5 * self.sample_rate)
                segments = []

                for start in range(0, len(full_audio) - segment_length + 1, hop_length):
                    segment = full_audio[start:start + segment_length]
                    segments.append(segment)

                self.recording_finished.emit(segments)
            else:
                self.recording_error.emit("Nie nagrano żadnych danych")

        except Exception as e:
            self.recording_error.emit(f"Błąd nagrywania: {str(e)}")

    def stop_recording(self):
        """Zatrzymaj nagrywanie"""
        self.is_recording = False


class RegistrationWindow(QMainWindow):
    """Główne okno rejestracji mówcy"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()
        self.setup_ui()
        self.setup_connections()

        # Inicjalizacja zmiennych
        self.speaker_audio_segments = []
        self.model = None
        self.recording_thread = None
        self.recording_device_id = None

        # Załaduj model
        self.load_model()

        # Wypełnij listę urządzeń
        self.refresh_device_list()

    def load_ui(self):
        """Ładuje UI z pliku .ui"""
        try:
            loader = QUiLoader()
            ui_file = "RegisterMenu.ui"

            if not os.path.exists(ui_file):
                raise FileNotFoundError(f"Nie znaleziono pliku UI: {ui_file}")

            file = QFile(ui_file)
            if not file.open(QFile.ReadOnly):
                raise IOError(f"Nie można otworzyć pliku: {ui_file}")

            self.ui = loader.load(file, self)
            file.close()

            if self.ui:
                self.setCentralWidget(self.ui.centralwidget)
            else:
                raise ValueError("Błąd ładowania UI")

        except Exception as e:
            print(f"Błąd ładowania UI: {e}")
            # Tworzenie awaryjnego UI
            self.create_fallback_ui()

    def create_fallback_ui(self):
        """Tworzy awaryjne UI jeśli ładowanie się nie powiedzie"""
        self.ui = QWidget()
        self.setCentralWidget(self.ui)
        layout = QVBoxLayout(self.ui)

        self.ui.textEdit_log = QTextEdit()
        self.ui.textEdit_log.setReadOnly(True)
        layout.addWidget(self.ui.textEdit_log)

        self.ui.label_status = QLabel("Błąd ładowania UI")
        layout.addWidget(self.ui.label_status)

    def setup_ui(self):
        """Inicjalizacja interfejsu"""
        # Ustaw tytuł okna
        self.setWindowTitle("Rejestracja Nowego Mówcy")

        # Ustaw domyślne wartości
        if hasattr(self.ui, 'spinBox_duration'):
            self.ui.spinBox_duration.setValue(30)

        # Zablokuj przyciski które wymagają danych
        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.setEnabled(False)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.setEnabled(False)

    def setup_connections(self):
        """Konfiguruje połączenia między sygnałami a slotami"""
        # Przyciski nagrywania
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.clicked.connect(self.start_recording)

        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.clicked.connect(self.stop_recording)

        if hasattr(self.ui, 'pushButton_preview'):
            self.ui.pushButton_preview.clicked.connect(self.preview_audio)

        # Przyciski plików
        if hasattr(self.ui, 'pushButton_add_files'):
            self.ui.pushButton_add_files.clicked.connect(self.add_audio_files)

        if hasattr(self.ui, 'pushButton_remove_files'):
            self.ui.pushButton_remove_files.clicked.connect(self.remove_audio_files)

        if hasattr(self.ui, 'pushButton_clear_files'):
            self.ui.pushButton_clear_files.clicked.connect(self.clear_audio_files)

        # Przyciski główne
        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.clicked.connect(self.register_speaker)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.clicked.connect(self.play_samples)

        if hasattr(self.ui, 'pushButton_exit'):
            self.ui.pushButton_exit.clicked.connect(self.close)

        if hasattr(self.ui, 'pushButton_refresh_devices'):
            self.ui.pushButton_refresh_devices.clicked.connect(self.refresh_device_list)

        # Zmiana w polu nazwy mówcy
        if hasattr(self.ui, 'lineEdit_speaker_name'):
            self.ui.lineEdit_speaker_name.textChanged.connect(self.update_register_button)

    def load_model(self):
        """Ładuje model rozpoznawania mówców"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Zaktualizuj informacje o systemie
            if hasattr(self.ui, 'label_status'):
                self.ui.label_status.setText("⚙️ Ładowanie modelu...")

            if hasattr(self.ui, 'label_device'):
                self.ui.label_device.setText(f"Urządzenie: {str(device).upper()}")

            # Ścieżka do modelu
            model_path = "./speaker_models/final_model.pt"

            if os.path.exists(model_path):
                self.log_message(f"📦 Wczytywanie modelu z {model_path}")
                self.model = SpeakerEncoder().to(device)

                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)

                    self.model.eval()
                    self.log_message("✅ Model załadowany pomyślnie")

                except Exception as e:
                    self.log_message(f"⚠️ Błąd ładowania modelu: {e}")
                    self.log_message("⚠️ Tworzenie nowego modelu")
                    self.model = SpeakerEncoder().to(device)
            else:
                self.log_message("⚠️ Model nie znaleziony - tworzę nowy")
                self.model = SpeakerEncoder().to(device)

            # Aktualizuj status
            if hasattr(self.ui, 'label_status'):
                self.ui.label_status.setText("✅ System gotowy")

        except Exception as e:
            self.log_message(f"❌ Krytyczny błąd ładowania modelu: {e}")

    def refresh_device_list(self):
        """Odświeża listę dostępnych urządzeń"""
        try:
            devices = sd.query_devices()

            if hasattr(self.ui, 'comboBox_input_devices'):
                combo_box = self.ui.comboBox_input_devices
                combo_box.clear()

                # Dodaj domyślne urządzenie
                default_input = sd.default.device[0]
                default_device = devices[default_input] if default_input < len(devices) else None

                if default_device and default_device['max_input_channels'] > 0:
                    name = default_device['name']
                    if len(name) > 40:
                        name = name[:37] + "..."
                    combo_box.addItem(f"⭐ {name} (domyślne)", default_input)

                # Dodaj inne urządzenia wejściowe
                for i, device in enumerate(devices):
                    if i == default_input:
                        continue

                    if device['max_input_channels'] > 0:
                        name = device['name']
                        if len(name) > 40:
                            name = name[:37] + "..."
                        combo_box.addItem(f"🎤 {name}", i)

                if combo_box.count() == 0:
                    combo_box.addItem("❌ Brak urządzeń wejściowych", None)

                self.log_message(f"📋 Znaleziono {combo_box.count()} urządzeń wejściowych")

        except Exception as e:
            self.log_message(f"❌ Błąd ładowania urządzeń: {e}")

    def log_message(self, message):
        """Dodaje wiadomość do logu"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        if hasattr(self.ui, 'textEdit_log'):
            self.ui.textEdit_log.append(formatted_message)

            # Przewiń do dołu
            cursor = self.ui.textEdit_log.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.ui.textEdit_log.setTextCursor(cursor)

    def start_recording(self):
        """Rozpoczyna nagrywanie"""
        # Sprawdź nazwę mówcy
        if not hasattr(self.ui, 'lineEdit_speaker_name') or not self.ui.lineEdit_speaker_name.text().strip():
            QMessageBox.warning(self, "Brak nazwy", "Proszę wprowadzić imię/nazwę mówcy.")
            return

        # Pobierz ustawienia
        if hasattr(self.ui, 'comboBox_recording_mode'):
            mode = self.ui.comboBox_recording_mode.currentText()

        duration = None
        if hasattr(self.ui, 'spinBox_duration'):
            if "przez czas" in mode:
                duration = self.ui.spinBox_duration.value()

        # Pobierz urządzenie
        if hasattr(self.ui, 'comboBox_input_devices'):
            device_id = self.ui.comboBox_input_devices.currentData()
            if device_id is None:
                QMessageBox.warning(self, "Brak urządzenia", "Nie wybrano urządzenia wejściowego.")
                return

        # Zablokuj przycisk start
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(False)

        # Odblokuj przycisk stop
        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(True)

        # Zresetuj progress bar
        if hasattr(self.ui, 'progressBar_recording'):
            self.ui.progressBar_recording.setValue(0)

        # Aktualizuj status
        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText("Rozpoczynam nagrywanie...")

        self.log_message("🎤 Rozpoczynam nagrywanie...")

        # Uruchom wątek nagrywania
        self.recording_thread = RecordingThread(
            device_id=device_id,
            duration=duration,
            monitor_gain=1.0
        )

        self.recording_thread.update_progress.connect(self.update_recording_progress)
        self.recording_thread.update_status.connect(self.update_recording_status)
        self.recording_thread.recording_finished.connect(self.recording_finished)
        self.recording_thread.recording_error.connect(self.recording_error)

        self.recording_thread.start()

    def stop_recording(self):
        """Zatrzymuje nagrywanie"""
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop_recording()
            self.log_message("⏹️ Zatrzymywanie nagrywania...")

    def update_recording_progress(self, progress):
        """Aktualizuje pasek postępu"""
        if hasattr(self.ui, 'progressBar_recording'):
            self.ui.progressBar_recording.setValue(progress)

    def update_recording_status(self, status):
        """Aktualizuje status nagrywania"""
        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText(status)

    def recording_finished(self, audio_segments):
        """Wywoływane po zakończeniu nagrywania"""
        self.speaker_audio_segments = audio_segments

        # Odblokuj przycisk start
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(True)

        # Zablokuj przycisk stop
        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(False)

        # Odblokuj przycisk podglądu
        if hasattr(self.ui, 'pushButton_preview'):
            self.ui.pushButton_preview.setEnabled(len(audio_segments) > 0)

        # Aktualizuj informacje o audio
        if hasattr(self.ui, 'label_audio_info'):
            total_duration = sum(len(seg) for seg in audio_segments) / 16000
            self.ui.label_audio_info.setText(
                f"Nagrane dane: {len(audio_segments)} segmentów, "
                f"całkowity czas: {total_duration:.1f}s"
            )

        self.log_message(f"✅ Nagrano {len(audio_segments)} segmentów audio")
        self.update_register_button()

    def recording_error(self, error_message):
        """Wywoływane przy błędzie nagrywania"""
        self.log_message(f"❌ {error_message}")

        # Przywróć stan przycisków
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(True)

        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(False)

        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText(f"Błąd: {error_message}")

    def preview_audio(self):
        """Odtwarza podgląd nagranego audio"""
        if not self.speaker_audio_segments:
            QMessageBox.information(self, "Brak danych", "Nie ma nagranych danych do odtworzenia.")
            return

        try:
            # Połącz segmenty
            full_audio = np.concatenate(self.speaker_audio_segments)

            # Normalizuj
            if np.max(np.abs(full_audio)) > 0:
                full_audio = full_audio / np.max(np.abs(full_audio))

            # Odtwórz
            sd.play(full_audio, samplerate=16000)
            sd.wait()

            self.log_message("🔊 Odtworzono podgląd audio")

        except Exception as e:
            self.log_message(f"❌ Błąd odtwarzania: {e}")

    def add_audio_files(self):
        """Dodaje pliki audio z dysku"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Pliki audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()

            if hasattr(self.ui, 'listWidget_audio_files'):
                list_widget = self.ui.listWidget_audio_files

                for file_path in file_paths:
                    list_widget.addItem(file_path)
                    self.log_message(f"📁 Dodano plik: {os.path.basename(file_path)}")

            self.update_register_button()

    def remove_audio_files(self):
        """Usuwa wybrane pliki audio z listy"""
        if hasattr(self.ui, 'listWidget_audio_files'):
            list_widget = self.ui.listWidget_audio_files

            for item in list_widget.selectedItems():
                list_widget.takeItem(list_widget.row(item))
                self.log_message(f"🗑️ Usunięto plik: {item.text()}")

    def clear_audio_files(self):
        """Czyści całą listę plików audio"""
        if hasattr(self.ui, 'listWidget_audio_files'):
            self.ui.listWidget_audio_files.clear()
            self.log_message("🧹 Wyczyszczono listę plików")

    def load_audio_file(self, filepath):
        """Wczytuje plik audio"""
        try:
            audio, sr = librosa.load(filepath, sr=16000, mono=True)

            # Normalizacja
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio

        except Exception as e:
            self.log_message(f"❌ Błąd wczytywania {os.path.basename(filepath)}: {e}")
            return None

    def update_register_button(self):
        """Aktualizuje stan przycisku rejestracji"""
        has_name = hasattr(self.ui, 'lineEdit_speaker_name') and self.ui.lineEdit_speaker_name.text().strip()
        has_audio = len(self.speaker_audio_segments) > 0

        if hasattr(self.ui, 'listWidget_audio_files'):
            has_audio = has_audio or (self.ui.listWidget_audio_files.count() > 0)

        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.setEnabled(has_name and has_audio)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.setEnabled(has_audio)

    def register_speaker(self):
        """Rejestruje nowego mówcę"""
        # Pobierz nazwę mówcy
        if not hasattr(self.ui, 'lineEdit_speaker_name'):
            QMessageBox.warning(self, "Błąd", "Pole nazwy mówcy nie istnieje.")
            return

        speaker_name = self.ui.lineEdit_speaker_name.text().strip()
        if not speaker_name:
            QMessageBox.warning(self, "Brak nazwy", "Proszę wprowadzić imię/nazwę mówcy.")
            return

        # Pobierz segmenty audio
        audio_segments = self.speaker_audio_segments.copy()

        # Sprawdź czy dodano pliki
        if hasattr(self.ui, 'listWidget_audio_files'):
            list_widget = self.ui.listWidget_audio_files

            for i in range(list_widget.count()):
                file_path = list_widget.item(i).text()
                audio = self.load_audio_file(file_path)

                if audio is not None:
                    # Podziel na segmenty 3-sekundowe
                    segment_length = 3 * 16000
                    hop_length = int(1.5 * 16000)

                    for start in range(0, len(audio) - segment_length + 1, hop_length):
                        segment = audio[start:start + segment_length]
                        audio_segments.append(segment)

                    self.log_message(f"✅ Przetworzono plik: {os.path.basename(file_path)}")

        if not audio_segments:
            QMessageBox.warning(self, "Brak danych", "Nie ma danych audio do rejestracji.")
            return

        self.log_message(f"📝 Rejestracja mówcy: {speaker_name}")
        self.log_message(f"📊 Liczba segmentów: {len(audio_segments)}")

        # Wczytaj lub utwórz bazę danych
        db_path = "./speaker_database.pkl"

        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    database = pickle.load(f)
                self.log_message(f"📋 Istniejąca baza: {len(database.get('speakers', {}))} mówców")
            except Exception as e:
                self.log_message(f"⚠️ Błąd wczytywania bazy: {e}")
                database = {'speakers': {}, 'speaker_names': {}}
        else:
            database = {'speakers': {}, 'speaker_names': {}}
            self.log_message("📋 Nowa baza danych utworzona")

        # Znajdź nowe ID
        existing_ids = list(database['speakers'].keys())
        if existing_ids:
            # Najprostsze ID
            new_id = str(len(existing_ids))
        else:
            new_id = "0"

        self.log_message(f"🆔 ID mówcy: {new_id}")

        # Zbierz embeddingi z segmentów
        embeddings = []
        device = next(self.model.parameters()).device

        for i, audio_segment in enumerate(audio_segments):
            segment_length = len(audio_segment) / 16000

            try:
                # Ekstrakcja cech
                features = extract_features(audio_segment)
                features = features.to(device)

                # Pobierz embedding
                with torch.no_grad():
                    embedding = self.model(features).squeeze(0).cpu()

                embeddings.append(embedding)

                if (i + 1) % 5 == 0 or i == len(audio_segments) - 1:
                    self.log_message(f"  📊 Przetworzono {i + 1}/{len(audio_segments)} segmentów")

            except Exception as e:
                self.log_message(f"  ⚠️ Segment {i + 1}: błąd - {str(e)[:50]}...")
                continue

        if len(embeddings) < 2:
            error_msg = f"Za mało segmentów: {len(embeddings)}/2"
            self.log_message(f"❌ {error_msg}")
            QMessageBox.warning(self, "Za mało danych", error_msg + "\nSpróbuj nagrać dłuższe audio.")
            return

        # Uśrednij embeddingi
        stacked = torch.stack(embeddings, dim=0)
        final_embedding = torch.mean(stacked, dim=0)
        final_embedding = F.normalize(final_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Zapisz do bazy
        database['speakers'][new_id] = final_embedding
        database['speaker_names'][new_id] = speaker_name

        # Zapisz bazę
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(database, f)

            # Tworzenie kopii zapasowej
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"./speaker_database_backup_{timestamp}.pkl"
            with open(backup_path, 'wb') as f:
                pickle.dump(database, f)

            success_msg = (
                f"✅ Zarejestrowano mówcę '{speaker_name}' z ID: {new_id}\n"
                f"   Embedding: {final_embedding.shape}\n"
                f"   Użyte segmenty: {len(embeddings)}\n"
                f"   Kopia zapasowa: {backup_path}"
            )

            self.log_message(success_msg)

            # Wyświetl okno sukcesu
            QMessageBox.information(
                self,
                "Rejestracja zakończona",
                success_msg
            )

            # Resetuj dane
            self.speaker_audio_segments = []

            if hasattr(self.ui, 'listWidget_audio_files'):
                self.ui.listWidget_audio_files.clear()

            if hasattr(self.ui, 'label_audio_info'):
                self.ui.label_audio_info.setText("Brak danych audio")

            self.update_register_button()

        except Exception as e:
            error_msg = f"Błąd zapisu bazy danych: {e}"
            self.log_message(f"❌ {error_msg}")
            QMessageBox.critical(self, "Błąd zapisu", error_msg)

    def play_samples(self):
        """Odtwarza wszystkie próbki audio"""
        if not self.speaker_audio_segments:
            QMessageBox.information(self, "Brak danych", "Nie ma danych audio do odtworzenia.")
            return

        try:
            # Połącz wszystkie segmenty
            all_audio = []
            for segment in self.speaker_audio_segments:
                all_audio.append(segment)
                # Dodaj krótką ciszę między segmentami
                all_audio.append(np.zeros(int(0.5 * 16000)))

            if all_audio:
                full_audio = np.concatenate(all_audio)

                # Normalizuj
                if np.max(np.abs(full_audio)) > 0:
                    full_audio = full_audio / np.max(np.abs(full_audio))

                # Odtwórz
                self.log_message("🔊 Odtwarzanie wszystkich próbek...")
                sd.play(full_audio, samplerate=16000)
                sd.wait()
                self.log_message("✅ Zakończono odtwarzanie")

        except Exception as e:
            self.log_message(f"❌ Błąd odtwarzania: {e}")

    def closeEvent(self, event):
        """Obsługuje zamknięcie okna"""
        # Zatrzymaj nagrywanie jeśli działa
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop_recording()
            self.recording_thread.wait(1000)

        event.accept()


def main():
    """Główna funkcja uruchamiająca aplikację"""
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = RegistrationWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()