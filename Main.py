#!/usr/bin/env python3
"""
Real-time Speaker Recognition System with UI
Filters audio to pass only recognized speaker (no denoising)
WINDOWS COMPATIBLE VERSION with PySide6 UI
Integrated with speaker registration using RegisterMenu.ui
WITH AUTOMATIC DATABASE RELOADING
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import threading
import queue
import argparse
import time
import resampy
import scipy.signal as signal
import warnings
from collections import defaultdict, deque
import json
import pickle
from datetime import datetime
import librosa
import os
import sys
import platform
import re

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QCheckBox,
    QDialog, QHBoxLayout, QSpinBox, QDoubleSpinBox, QMenuBar, QMenu,
    QMessageBox, QFileDialog, QListWidget, QGroupBox, QFormLayout,
    QGridLayout, QTabWidget, QLineEdit, QProgressBar
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QRect, QObject, Signal, QThread, Slot, QTimer, Qt
from PySide6.QtGui import QAction, QTextCursor

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

warnings.filterwarnings('ignore')

# Global variables
SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default parameters
DEFAULT_PARAMS = {
    'input_gain': 3.0,
    'output_gain': 1.0,
    'speaker_gain': 2.0,
    'similarity_threshold': 0.55,
    'fail_safe_matches': 3,
    'fail_safe_duration': 5
}


# ============================================
# LOGGER CLASS FOR UI OUTPUT
# ============================================

class UILogger(QObject):
    """Logger that sends messages to UI ConsoleOutput"""
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.main_window = None
        self.console_widget = None

    def set_main_window(self, main_window):
        """Set reference to main window"""
        self.main_window = main_window
        if hasattr(main_window.ui, 'ConsoleOutput'):
            self.console_widget = main_window.ui.ConsoleOutput

    def log(self, message):
        """Log message to UI and console"""
        # Always print to console for debugging
        print(message)

        # Send to UI if available
        if self.console_widget is not None:
            self.log_signal.emit(str(message))
        elif self.main_window is not None and hasattr(self.main_window.ui, 'ConsoleOutput'):
            self.console_widget = self.main_window.ui.ConsoleOutput
            self.log_signal.emit(str(message))


# Create global logger instance
ui_logger = UILogger()


# Helper function to replace all prints
def log(message):
    """Log message to UI and console"""
    ui_logger.log(message)


# ============================================
# SPEAKER ENCODER MODEL
# ============================================

class SpeakerEncoder(nn.Module):
    """Speaker recognition model"""

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


def extract_features_for_recognition(audio, sr=16000):
    """Extract MFCC features for speaker recognition"""
    # Normalize audio first
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20,
        n_mels=80,
        n_fft=512,
        hop_length=160,
        fmin=50,
        fmax=8000
    )

    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (60, time)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, 60, time)

    # Normalization
    features_tensor = (features_tensor - features_tensor.mean(dim=2, keepdim=True)) / \
                      (features_tensor.std(dim=2, keepdim=True) + 1e-8)

    return features_tensor


# ============================================
# SPEAKER RECOGNIZER
# ============================================

class SpeakerRecognizer:
    """Speaker recognition using trained model"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # Load model
        self.model = SpeakerEncoder()
        self.model.to(DEVICE)

        # Load trained weights
        model_path = "./speaker_models/final_model.pt"
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                log("✅ Speaker recognition model loaded")
            except Exception as e:
                log(f"❌ Error loading speaker model: {e}")
                log("⚠️  Using untrained model")
        else:
            log("⚠️  Warning: No trained speaker model found")

        self.model.eval()

        # Speaker database
        self.speaker_database_path = "./speaker_database.pkl"
        self.speaker_embeddings = {}  # speaker_id -> embedding tensor
        self.speaker_names = {}  # speaker_id -> name

        # Similarity threshold
        self.similarity_threshold = 0.3

        # Load existing database
        self.load_database()

    def reload_database(self):
        """Reload speaker database from file - DODANA METODA"""
        log("🔄 Reloading speaker database...")
        try:
            old_count = len(self.speaker_embeddings)

            if os.path.exists(self.speaker_database_path):
                with open(self.speaker_database_path, 'rb') as f:
                    db = pickle.load(f)

                if 'speakers' in db and 'speaker_names' in db:
                    self.speaker_embeddings = db['speakers']
                    self.speaker_names = db['speaker_names']
                    new_count = len(self.speaker_embeddings)

                    log(f"📊 Speaker database reloaded: {new_count} speakers")
                    log(f"📊 Added: {new_count - old_count} new speakers")

                    # Print speaker names
                    for speaker_id, name in self.speaker_names.items():
                        log(f"  ID {speaker_id}: {name}")

                    return True
                else:
                    log("⚠️  Invalid database format")
                    return False
            else:
                log("📊 No database file found")
                return False

        except Exception as e:
            log(f"❌ Error reloading database: {e}")
            return False

    def load_database(self):
        """Load speaker database from file"""
        if os.path.exists(self.speaker_database_path):
            try:
                with open(self.speaker_database_path, 'rb') as f:
                    db = pickle.load(f)

                if 'speakers' in db and 'speaker_names' in db:
                    self.speaker_embeddings = db['speakers']
                    self.speaker_names = db['speaker_names']
                    log(f"📊 Speaker database: {len(self.speaker_embeddings)} speakers loaded")
                    # Print speaker names
                    for speaker_id, name in self.speaker_names.items():
                        log(f"  ID {speaker_id}: {name}")
                else:
                    log("⚠️  Invalid database format")
            except Exception as e:
                log(f"❌ Error loading database: {e}")
        else:
            log("📊 Creating new speaker database")

    def extract_embedding(self, audio):
        """Extract speaker embedding from audio"""
        if len(audio) < 24000:  # Need at least 1.5 second for better recognition
            return None

        # Ustaw model w tryb ewaluacji - KLUCZOWA ZMIANA
        self.model.eval()

        with torch.no_grad():
            # Extract features - use the entire audio for better accuracy
            features = extract_features_for_recognition(audio)
            features = features.to(DEVICE)

            # Get embedding from model
            embedding = self.model(features)

            # Squeeze if needed
            if embedding.dim() > 1:
                embedding = embedding.squeeze(0)

            embedding = embedding.cpu()

            return embedding

    def recognize_speaker(self, audio):
        """Recognize speaker in audio using model and database"""
        embedding = self.extract_embedding(audio)
        if embedding is None or len(self.speaker_embeddings) == 0:
            return None, 0.0, "Unknown"

        best_speaker_id = None
        best_similarity = -1.0

        for speaker_id, ref_embedding in self.speaker_embeddings.items():
            if isinstance(ref_embedding, torch.Tensor):
                # Adjust dimensions
                if embedding.dim() == 1:
                    embedding_2d = embedding.unsqueeze(0)
                else:
                    embedding_2d = embedding

                if ref_embedding.dim() == 1:
                    ref_embedding_2d = ref_embedding.unsqueeze(0)
                else:
                    ref_embedding_2d = ref_embedding

                # Calculate cosine similarity
                try:
                    similarity = F.cosine_similarity(embedding_2d, ref_embedding_2d, dim=1).item()

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_speaker_id = speaker_id
                except Exception as e:
                    log(f"  Error calculating similarity for {speaker_id}: {e}")

        # Check threshold
        if best_similarity >= self.similarity_threshold:
            name = self.speaker_names.get(best_speaker_id, f"Speaker_{best_speaker_id}")
            return best_speaker_id, best_similarity, name

        return None, best_similarity, "Unknown"

    def list_speakers(self):
        """List all registered speakers"""
        speakers = []
        for speaker_id, name in self.speaker_names.items():
            speakers.append(f"ID: {speaker_id} -> '{name}'")
        return speakers

    def get_speaker_name(self, speaker_id):
        """Get speaker name by ID"""
        return self.speaker_names.get(str(speaker_id), f"Speaker_{speaker_id}")

    def get_all_speakers(self):
        """Get all speaker IDs and names"""
        return self.speaker_names.copy()


# ============================================
# SPEAKER REGISTRATION WINDOW (USING RegisterMenu.ui)
# ============================================

class RecordingThread(QThread):
    """Thread for recording audio with monitoring"""
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
        """Main recording thread function"""
        try:
            self.update_status.emit("🔴 Rozpoczynanie nagrywania...")
            self.audio_data = []
            self.is_recording = True

            # Buffer for monitoring
            monitor_buffer = queue.Queue()

            def input_callback(indata, frames, time_info, status):
                """Input callback"""
                if self.is_recording:
                    chunk = indata.copy().flatten()
                    self.audio_data.append(chunk.copy())

                    # Add to monitoring with gain
                    if self.monitor_gain != 1.0:
                        chunk = chunk * self.monitor_gain
                        chunk = np.clip(chunk, -1.0, 1.0)

                    monitor_buffer.put(chunk)

            def output_callback(outdata, frames, time_info, status):
                """Output callback"""
                try:
                    chunk = monitor_buffer.get_nowait()
                    if len(chunk) < frames:
                        chunk = np.pad(chunk, (0, frames - len(chunk)), mode='constant')
                    elif len(chunk) > frames:
                        chunk = chunk[:frames]
                    outdata[:, 0] = chunk
                except queue.Empty:
                    outdata.fill(0)

            # Start streams
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

                # Update progress
                if time.time() - last_update > 0.1:
                    if self.duration:
                        progress = min(100, int((elapsed / self.duration) * 100))
                        self.update_progress.emit(progress)
                        self.update_status.emit(f"Nagrywanie: {elapsed:.1f}s / {self.duration}s")
                    else:
                        self.update_status.emit(f"Nagrywanie: {elapsed:.1f}s (naciśnij STOP)")

                    last_update = time.time()

                # Check if time elapsed
                if self.duration and elapsed >= self.duration:
                    self.is_recording = False
                    break

                time.sleep(0.05)

            # Stop streams
            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()

            # Combine data
            if self.audio_data:
                full_audio = np.concatenate(self.audio_data)
                duration = len(full_audio) / self.sample_rate
                self.update_status.emit(f"✅ Nagrano {duration:.1f} sekund audio")

                # Split into segments (3 seconds with 50% overlap)
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
        """Stop recording"""
        self.is_recording = False


class RegistrationWindow(QMainWindow):
    """Speaker registration window using RegisterMenu.ui"""

    # DODANY SYGNAŁ - baza danych została zaktualizowana
    database_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load_ui()

        if hasattr(self, 'ui'):
            self.setup_ui()
            self.setup_connections()
        else:
            self.create_fallback_ui()
            self.setup_connections()

        # Initialize variables
        self.speaker_audio_segments = []
        self.model = None
        self.recording_thread = None
        self.recording_device_id = None

        # Load model
        self.load_model()

        # Refresh device list
        self.refresh_device_list()

    def load_ui(self):
        """Load UI from RegisterMenu.ui file"""
        try:
            loader = QUiLoader()
            ui_file = "RegisterMenu.ui"

            if not os.path.exists(ui_file):
                log(f"ERROR: Nie znaleziono pliku {ui_file}")
                return

            file = QFile(ui_file)
            if not file.open(QFile.ReadOnly):
                log(f"ERROR: Nie można otworzyć pliku {ui_file}")
                return

            self.ui = loader.load(file, self)
            file.close()

            if self.ui:
                self.setCentralWidget(self.ui.centralwidget)
                log("SUCCESS: Registration UI załadowane z RegisterMenu.ui")
            else:
                log("ERROR: loader.load() zwrócił None")

        except Exception as e:
            log(f"ERROR w load_ui Registration: {e}")

    def create_fallback_ui(self):
        """Create fallback UI if loading fails"""
        log("Tworzenie awaryjnego UI rejestracji...")
        self.ui = QWidget()
        self.setCentralWidget(self.ui)
        layout = QVBoxLayout(self.ui)

        # Speaker name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nazwa mówcy:"))
        self.ui.lineEdit_speaker_name = QLineEdit()
        self.ui.lineEdit_speaker_name.setPlaceholderText("Wprowadź imię lub nazwę")
        name_layout.addWidget(self.ui.lineEdit_speaker_name)
        layout.addLayout(name_layout)

        # Input devices
        device_layout = QVBoxLayout()
        device_layout.addWidget(QLabel("Urządzenie wejścia:"))
        self.ui.comboBox_input_devices = QComboBox()
        device_layout.addWidget(self.ui.comboBox_input_devices)

        self.ui.pushButton_refresh_devices = QPushButton("Odśwież listę")
        device_layout.addWidget(self.ui.pushButton_refresh_devices)

        layout.addLayout(device_layout)

        # Recording mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Tryb nagrywania:"))
        self.ui.comboBox_recording_mode = QComboBox()
        self.ui.comboBox_recording_mode.addItems(["Nagrywanie przez czas", "Nagrywanie do zatrzymania"])
        mode_layout.addWidget(self.ui.comboBox_recording_mode)

        self.ui.spinBox_duration = QSpinBox()
        self.ui.spinBox_duration.setRange(10, 600)
        self.ui.spinBox_duration.setValue(30)
        self.ui.spinBox_duration.setSuffix(" sekund")
        mode_layout.addWidget(self.ui.spinBox_duration)
        layout.addLayout(mode_layout)

        # Recording buttons
        button_layout = QHBoxLayout()
        self.ui.pushButton_start_recording = QPushButton("▶ Rozpocznij nagrywanie")
        self.ui.pushButton_stop_recording = QPushButton("⏹ Zatrzymaj nagrywanie")
        self.ui.pushButton_stop_recording.setEnabled(False)
        self.ui.pushButton_preview = QPushButton("🔊 Podgląd")
        self.ui.pushButton_preview.setEnabled(False)

        button_layout.addWidget(self.ui.pushButton_start_recording)
        button_layout.addWidget(self.ui.pushButton_stop_recording)
        button_layout.addWidget(self.ui.pushButton_preview)
        layout.addLayout(button_layout)

        # Progress bar
        self.ui.progressBar_recording = QProgressBar()
        layout.addWidget(self.ui.progressBar_recording)

        # Status label
        self.ui.label_recording_status = QLabel("Gotowy do nagrywania")
        layout.addWidget(self.ui.label_recording_status)

        # Audio files list
        files_group = QGroupBox("Pliki audio")
        files_layout = QVBoxLayout()

        self.ui.listWidget_audio_files = QListWidget()
        files_layout.addWidget(self.ui.listWidget_audio_files)

        file_button_layout = QHBoxLayout()
        self.ui.pushButton_add_files = QPushButton("Dodaj pliki...")
        self.ui.pushButton_remove_files = QPushButton("Usuń wybrane")
        self.ui.pushButton_clear_files = QPushButton("Wyczyść listę")

        file_button_layout.addWidget(self.ui.pushButton_add_files)
        file_button_layout.addWidget(self.ui.pushButton_remove_files)
        file_button_layout.addWidget(self.ui.pushButton_clear_files)
        files_layout.addLayout(file_button_layout)

        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        # Log
        self.ui.textEdit_log = QTextEdit()
        self.ui.textEdit_log.setReadOnly(True)
        self.ui.textEdit_log.setPlaceholderText("Logi systemowe będą wyświetlane tutaj...")
        layout.addWidget(self.ui.textEdit_log)

        # Audio info label
        self.ui.label_audio_info = QLabel("Brak danych audio")
        layout.addWidget(self.ui.label_audio_info)

        # Action buttons
        action_layout = QHBoxLayout()
        self.ui.pushButton_register = QPushButton("📝 Zarejestruj mówcę")
        self.ui.pushButton_register.setEnabled(False)
        self.ui.pushButton_play_samples = QPushButton("🎧 Odsłuchaj próbki")
        self.ui.pushButton_play_samples.setEnabled(False)
        self.ui.pushButton_exit = QPushButton("Zamknij")

        action_layout.addWidget(self.ui.pushButton_register)
        action_layout.addWidget(self.ui.pushButton_play_samples)
        action_layout.addWidget(self.ui.pushButton_exit)
        layout.addLayout(action_layout)

        # System info labels
        self.ui.label_status = QLabel("⚙️ Inicjalizacja systemu...")
        self.ui.label_device = QLabel("Urządzenie: CPU")
        self.ui.label_separator = QLabel("Separator: Nieznaleziony")

        layout.addWidget(self.ui.label_status)
        layout.addWidget(self.ui.label_device)
        layout.addWidget(self.ui.label_separator)

        log("Awaryjne UI rejestracji utworzone")

    def setup_ui(self):
        """Setup registration UI"""
        self.setWindowTitle("Rejestracja Nowego Mówcy")

        # Set default values
        if hasattr(self.ui, 'spinBox_duration'):
            self.ui.spinBox_duration.setValue(30)

        # Disable buttons that require data
        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.setEnabled(False)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.setEnabled(False)

        # Connect recording mode change
        if hasattr(self.ui, 'comboBox_recording_mode'):
            self.ui.comboBox_recording_mode.currentTextChanged.connect(self.update_recording_mode)

    def setup_connections(self):
        """Setup signal connections"""
        # Recording buttons
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.clicked.connect(self.start_recording)

        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.clicked.connect(self.stop_recording)

        if hasattr(self.ui, 'pushButton_preview'):
            self.ui.pushButton_preview.clicked.connect(self.preview_audio)

        # File buttons
        if hasattr(self.ui, 'pushButton_add_files'):
            self.ui.pushButton_add_files.clicked.connect(self.add_audio_files)

        if hasattr(self.ui, 'pushButton_remove_files'):
            self.ui.pushButton_remove_files.clicked.connect(self.remove_audio_files)

        if hasattr(self.ui, 'pushButton_clear_files'):
            self.ui.pushButton_clear_files.clicked.connect(self.clear_audio_files)

        # Main buttons
        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.clicked.connect(self.register_speaker)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.clicked.connect(self.play_samples)

        if hasattr(self.ui, 'pushButton_exit'):
            self.ui.pushButton_exit.clicked.connect(self.close)

        if hasattr(self.ui, 'pushButton_refresh_devices'):
            self.ui.pushButton_refresh_devices.clicked.connect(self.refresh_device_list)

        # Speaker name field change
        if hasattr(self.ui, 'lineEdit_speaker_name'):
            self.ui.lineEdit_speaker_name.textChanged.connect(self.update_register_button)

    def update_recording_mode(self, mode):
        """Update recording mode UI"""
        if hasattr(self.ui, 'spinBox_duration'):
            if "przez czas" in mode:
                self.ui.spinBox_duration.setEnabled(True)
            else:
                self.ui.spinBox_duration.setEnabled(False)

    def load_model(self):
        """Load speaker recognition model"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Update system info
            if hasattr(self.ui, 'label_status'):
                self.ui.label_status.setText("⚙️ Ładowanie modelu...")

            if hasattr(self.ui, 'label_device'):
                self.ui.label_device.setText(f"Urządzenie: {str(device).upper()}")

            # Model path
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

            # Update status
            if hasattr(self.ui, 'label_status'):
                self.ui.label_status.setText("✅ System gotowy")

        except Exception as e:
            self.log_message(f"❌ Krytyczny błąd ładowania modelu: {e}")

    def refresh_device_list(self):
        """Refresh list of available devices"""
        try:
            devices = sd.query_devices()

            if hasattr(self.ui, 'comboBox_input_devices'):
                combo_box = self.ui.comboBox_input_devices
                combo_box.clear()

                # Add default device
                default_input = sd.default.device[0]
                default_device = devices[default_input] if default_input < len(devices) else None

                if default_device and default_device['max_input_channels'] > 0:
                    name = default_device['name']
                    if len(name) > 40:
                        name = name[:37] + "..."
                    combo_box.addItem(f"⭐ {name} (domyślne)", default_input)

                # Add other input devices
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
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        if hasattr(self.ui, 'textEdit_log'):
            self.ui.textEdit_log.append(formatted_message)

            # Scroll to bottom
            cursor = self.ui.textEdit_log.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.ui.textEdit_log.setTextCursor(cursor)

    def start_recording(self):
        """Start recording"""
        # Check speaker name
        if not hasattr(self.ui, 'lineEdit_speaker_name') or not self.ui.lineEdit_speaker_name.text().strip():
            QMessageBox.warning(self, "Brak nazwy", "Proszę wprowadzić imię/nazwę mówcy.")
            return

        # Get settings
        mode = "Nagrywanie przez czas"  # Default
        if hasattr(self.ui, 'comboBox_recording_mode'):
            mode = self.ui.comboBox_recording_mode.currentText()

        duration = None
        if hasattr(self.ui, 'spinBox_duration'):
            if "przez czas" in mode:
                duration = self.ui.spinBox_duration.value()

        # Get device
        if hasattr(self.ui, 'comboBox_input_devices'):
            device_id = self.ui.comboBox_input_devices.currentData()
            if device_id is None:
                QMessageBox.warning(self, "Brak urządzenia", "Nie wybrano urządzenia wejściowego.")
                return

        # Disable start button
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(False)

        # Enable stop button
        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(True)

        # Reset progress bar
        if hasattr(self.ui, 'progressBar_recording'):
            self.ui.progressBar_recording.setValue(0)

        # Update status
        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText("Rozpoczynam nagrywanie...")

        self.log_message("🎤 Rozpoczynam nagrywanie...")

        # Start recording thread
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
        """Stop recording"""
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop_recording()
            self.log_message("⏹️ Zatrzymywanie nagrywania...")

    def update_recording_progress(self, progress):
        """Update recording progress bar"""
        if hasattr(self.ui, 'progressBar_recording'):
            self.ui.progressBar_recording.setValue(progress)

    def update_recording_status(self, status):
        """Update recording status"""
        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText(status)

    def recording_finished(self, audio_segments):
        """Called when recording finishes"""
        self.speaker_audio_segments = audio_segments

        # Enable start button
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(True)

        # Disable stop button
        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(False)

        # Enable preview button
        if hasattr(self.ui, 'pushButton_preview'):
            self.ui.pushButton_preview.setEnabled(len(audio_segments) > 0)

        # Update audio info
        if hasattr(self.ui, 'label_audio_info'):
            total_duration = sum(len(seg) for seg in audio_segments) / 16000
            self.ui.label_audio_info.setText(
                f"Nagrane dane: {len(audio_segments)} segmentów, "
                f"całkowity czas: {total_duration:.1f}s"
            )

        self.log_message(f"✅ Nagrano {len(audio_segments)} segmentów audio")
        self.update_register_button()

    def recording_error(self, error_message):
        """Called on recording error"""
        self.log_message(f"❌ {error_message}")

        # Restore button states
        if hasattr(self.ui, 'pushButton_start_recording'):
            self.ui.pushButton_start_recording.setEnabled(True)

        if hasattr(self.ui, 'pushButton_stop_recording'):
            self.ui.pushButton_stop_recording.setEnabled(False)

        if hasattr(self.ui, 'label_recording_status'):
            self.ui.label_recording_status.setText(f"Błąd: {error_message}")

    def preview_audio(self):
        """Preview recorded audio"""
        if not self.speaker_audio_segments:
            QMessageBox.information(self, "Brak danych", "Nie ma nagranych danych do odtworzenia.")
            return

        try:
            # Combine segments
            full_audio = np.concatenate(self.speaker_audio_segments)

            # Normalize
            if np.max(np.abs(full_audio)) > 0:
                full_audio = full_audio / np.max(np.abs(full_audio))

            # Play
            sd.play(full_audio, samplerate=16000)
            sd.wait()

            self.log_message("🔊 Odtworzono podgląd audio")

        except Exception as e:
            self.log_message(f"❌ Błąd odtwarzania: {e}")

    def add_audio_files(self):
        """Add audio files from disk"""
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
        """Remove selected audio files from list"""
        if hasattr(self.ui, 'listWidget_audio_files'):
            list_widget = self.ui.listWidget_audio_files

            for item in list_widget.selectedItems():
                list_widget.takeItem(list_widget.row(item))
                self.log_message(f"🗑️ Usunięto plik: {item.text()}")

    def clear_audio_files(self):
        """Clear all audio files from list"""
        if hasattr(self.ui, 'listWidget_audio_files'):
            self.ui.listWidget_audio_files.clear()
            self.log_message("🧹 Wyczyszczono listę plików")

    def load_audio_file(self, filepath):
        """Load audio file"""
        try:
            audio, sr = librosa.load(filepath, sr=16000, mono=True)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return audio

        except Exception as e:
            self.log_message(f"❌ Błąd wczytywania {os.path.basename(filepath)}: {e}")
            return None

    def update_register_button(self):
        """Update register button state"""
        has_name = hasattr(self.ui, 'lineEdit_speaker_name') and self.ui.lineEdit_speaker_name.text().strip()
        has_audio = len(self.speaker_audio_segments) > 0

        if hasattr(self.ui, 'listWidget_audio_files'):
            has_audio = has_audio or (self.ui.listWidget_audio_files.count() > 0)

        if hasattr(self.ui, 'pushButton_register'):
            self.ui.pushButton_register.setEnabled(has_name and has_audio)

        if hasattr(self.ui, 'pushButton_play_samples'):
            self.ui.pushButton_play_samples.setEnabled(has_audio)

    def register_speaker(self):
        """Register new speaker"""
        # Get speaker name
        if not hasattr(self.ui, 'lineEdit_speaker_name'):
            QMessageBox.warning(self, "Błąd", "Pole nazwy mówcy nie istnieje.")
            return

        speaker_name = self.ui.lineEdit_speaker_name.text().strip()
        if not speaker_name:
            QMessageBox.warning(self, "Brak nazwy", "Proszę wprowadzić imię/nazwę mówcy.")
            return

        # Get audio segments
        audio_segments = self.speaker_audio_segments.copy()

        # Check if files were added
        if hasattr(self.ui, 'listWidget_audio_files'):
            list_widget = self.ui.listWidget_audio_files

            for i in range(list_widget.count()):
                file_path = list_widget.item(i).text()
                audio = self.load_audio_file(file_path)

                if audio is not None:
                    # Split into 3-second segments
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

        # Load or create database
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

        # Find new ID
        existing_ids = list(database['speakers'].keys())
        if existing_ids:
            # Simple ID
            new_id = str(len(existing_ids))
        else:
            new_id = "0"

        self.log_message(f"🆔 ID mówcy: {new_id}")

        # Collect embeddings from segments
        embeddings = []
        device = next(self.model.parameters()).device

        # USTAW MODEL W TRYB EWALUACJI - KLUCZOWA ZMIANA
        self.model.eval()

        for i, audio_segment in enumerate(audio_segments):
            try:
                # Extract features
                features = extract_features_for_recognition(audio_segment)
                features = features.to(device)

                # Get embedding - UŻYJ torch.no_grad()
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

        # Average embeddings
        stacked = torch.stack(embeddings, dim=0)
        final_embedding = torch.mean(stacked, dim=0)
        final_embedding = F.normalize(final_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Save to database
        database['speakers'][new_id] = final_embedding
        database['speaker_names'][new_id] = speaker_name

        # Save database
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(database, f)

            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

            # EMITUJ SYGNAŁ ŻE BAZA DANYCH ZOSTAŁA ZAKTUALIZOWANA
            self.database_updated.emit()

            # Show success dialog
            QMessageBox.information(
                self,
                "Rejestracja zakończona",
                success_msg
            )

            # Reset data
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
        """Play all audio samples"""
        if not self.speaker_audio_segments:
            QMessageBox.information(self, "Brak danych", "Nie ma danych audio do odtworzenia.")
            return

        try:
            # Combine all segments
            all_audio = []
            for segment in self.speaker_audio_segments:
                all_audio.append(segment)
                # Add short silence between segments
                all_audio.append(np.zeros(int(0.5 * 16000)))

            if all_audio:
                full_audio = np.concatenate(all_audio)

                # Normalize
                if np.max(np.abs(full_audio)) > 0:
                    full_audio = full_audio / np.max(np.abs(full_audio))

                # Play
                self.log_message("🔊 Odtwarzanie wszystkich próbek...")
                sd.play(full_audio, samplerate=16000)
                sd.wait()
                self.log_message("✅ Zakończono odtwarzanie")

        except Exception as e:
            self.log_message(f"❌ Błąd odtwarzania: {e}")

    def closeEvent(self, event):
        """Handle window close"""
        # Stop recording if running
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop_recording()
            self.recording_thread.wait(1000)

        event.accept()


# ============================================
# CONFIGURATION WINDOW
# ============================================

class ConfigWindow(QMainWindow):
    """Configuration window for adjusting parameters"""
    config_changed = Signal(dict)  # Signal emitted when config changes
    config_closed = Signal()  # Signal emitted when window closes

    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.params = current_params.copy()
        self.load_ui()
        self.setup_ui()
        self.setup_connections()

    def load_ui(self):
        """Load UI from Config.ui file"""
        try:
            loader = QUiLoader()
            ui_file = "Config.ui"

            if not os.path.exists(ui_file):
                log(f"ERROR: Nie znaleziono pliku {ui_file}")
                self.create_fallback_ui()
                return

            file = QFile(ui_file)
            if not file.open(QFile.ReadOnly):
                log(f"ERROR: Nie można otworzyć pliku {ui_file}")
                self.create_fallback_ui()
                return

            self.ui = loader.load(file, self)
            file.close()

            if self.ui:
                self.setCentralWidget(self.ui.centralwidget)
                log("SUCCESS: Config UI załadowane")
            else:
                log("ERROR: loader.load() zwrócił None")
                self.create_fallback_ui()

        except Exception as e:
            log(f"ERROR w load_ui Config: {e}")
            self.create_fallback_ui()

    def create_fallback_ui(self):
        """Create fallback UI if loading fails"""
        self.ui = QWidget()
        self.setCentralWidget(self.ui)
        layout = QVBoxLayout(self.ui)

        self.comboBox = QComboBox()
        layout.addWidget(self.comboBox)

        self.label = QLabel("Wartość: ")
        layout.addWidget(self.label)

        button_layout = QHBoxLayout()
        self.subtract_btn = QPushButton("-")
        self.add_btn = QPushButton("+")
        button_layout.addWidget(self.subtract_btn)
        button_layout.addWidget(self.add_btn)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Gotowe")
        layout.addWidget(self.status_label)

        # Map names for compatibility
        self.ui.comboBox = self.comboBox
        self.ui.label = self.label
        self.ui.Subtract = self.subtract_btn
        self.ui.Add = self.add_btn

    def setup_ui(self):
        """Setup UI elements with current parameters"""
        # Populate combo box with parameters
        self.ui.comboBox.clear()
        self.ui.comboBox.addItem("Próg podobieństwa", "similarity_threshold")
        self.ui.comboBox.addItem("Wzmocnienie wejścia", "input_gain")
        self.ui.comboBox.addItem("Wzmocnienie wyjścia", "output_gain")
        self.ui.comboBox.addItem("Wzmocnienie mówcy", "speaker_gain")
        self.ui.comboBox.addItem("Fail-safe matches", "fail_safe_matches")
        self.ui.comboBox.addItem("Fail-safe duration", "fail_safe_duration")

        # Update label with current value
        self.update_display()

    def setup_connections(self):
        """Setup signal connections"""
        self.ui.comboBox.currentIndexChanged.connect(self.update_display)
        self.ui.Add.clicked.connect(self.increase_value)
        self.ui.Subtract.clicked.connect(self.decrease_value)

    def update_display(self):
        """Update display with current parameter value"""
        param_key = self.ui.comboBox.currentData()
        if param_key and param_key in self.params:
            value = self.params[param_key]

            # Format value based on parameter type
            if param_key in ['similarity_threshold']:
                self.ui.label.setText(f"Wartość: {value:.3f}")
            elif param_key in ['fail_safe_matches', 'fail_safe_duration']:
                self.ui.label.setText(f"Wartość: {int(value)}")
            else:
                self.ui.label.setText(f"Wartość: {value:.2f}")

    def increase_value(self):
        """Increase current parameter value"""
        param_key = self.ui.comboBox.currentData()
        if param_key and param_key in self.params:
            # Determine step size based on parameter
            if param_key in ['similarity_threshold']:
                step = 0.05
                self.params[param_key] = min(1.0, self.params[param_key] + step)
            elif param_key in ['input_gain', 'output_gain', 'speaker_gain']:
                step = 0.5
                self.params[param_key] = min(10.0, self.params[param_key] + step)
            elif param_key == 'fail_safe_matches':
                step = 1
                self.params[param_key] = min(10, self.params[param_key] + step)
            elif param_key == 'fail_safe_duration':
                step = 1
                self.params[param_key] = min(20, self.params[param_key] + step)

            self.update_display()
            self.config_changed.emit(self.params)
            log(f"Zwiększono {param_key}: {self.params[param_key]:.3f}")

    def decrease_value(self):
        """Decrease current parameter value"""
        param_key = self.ui.comboBox.currentData()
        if param_key and param_key in self.params:
            # Determine step size based on parameter
            if param_key in ['similarity_threshold']:
                step = 0.05
                self.params[param_key] = max(0.0, self.params[param_key] - step)
            elif param_key in ['input_gain', 'output_gain', 'speaker_gain']:
                step = 0.5
                self.params[param_key] = max(0.0, self.params[param_key] - step)
            elif param_key == 'fail_safe_matches':
                step = 1
                self.params[param_key] = max(1, self.params[param_key] - step)
            elif param_key == 'fail_safe_duration':
                step = 1
                self.params[param_key] = max(1, self.params[param_key] - step)

            self.update_display()
            self.config_changed.emit(self.params)
            log(f"Zmniejszono {param_key}: {self.params[param_key]:.3f}")

    def closeEvent(self, event):
        """Handle window close event"""
        self.config_closed.emit()
        event.accept()


# ============================================
# PROCESSOR THREAD FOR AUDIO PROCESSING
# ============================================

class ProcessorThread(QThread):
    """Thread for running audio processor without blocking UI"""
    processor_started = Signal()
    processor_stopped = Signal()
    processor_error = Signal(str)

    def __init__(self, processor, input_device=None, output_device=None):
        super().__init__()
        self.processor = processor
        self.input_device = input_device
        self.output_device = output_device

    def run(self):
        """Run the processor in a separate thread"""
        try:
            self.processor_started.emit()
            self.processor.run(self.input_device, self.output_device)
        except Exception as e:
            error_msg = f"Error in processor thread: {e}"
            log(error_msg)
            self.processor_error.emit(error_msg)
        finally:
            self.processor_stopped.emit()


# ============================================
# AI ENHANCED SPEAKER TRACKER WITH FAIL-SAFE
# ============================================

class AIEnhancedSpeakerTracker:
    """Speaker tracker with fail-safe mechanism"""

    def __init__(self):
        # Use speaker recognizer
        self.recognizer = SpeakerRecognizer()

        # Speaker selection mode
        self.selected_speaker_id = -1  # -1 = auto (similarity-based), specific ID = only that speaker
        self.block_other_speakers = False

        # FAIL-SAFE MECHANISM
        self.fail_safe_enabled = True
        self.consecutive_matches = 0  # Counter for consecutive matches
        self.consecutive_misses = 0  # Counter for consecutive misses
        self.fail_safe_counter = 0  # Fail-safe counter (how many samples to pass after match series)

        # Fail-safe parameters
        self.fail_safe_match_threshold = 3  # After how many matches to activate fail-safe
        self.fail_safe_duration = 5  # How many samples to pass in fail-safe mode
        self.max_fail_safe_attempts = 3  # Max number of fail-safe activations in a row

        # Activity tracking with history for better stability
        self.speaker_activity = defaultdict(float)
        self.speaker_history = defaultdict(list)
        self.last_recognized_speaker = None
        self.last_recognized_time = 0
        self.last_similarity = 0.0

        # Buffer for accumulating audio for recognition
        self.recognition_buffer = np.array([], dtype=np.float32)
        self.recognition_buffer_size = 32000  # 2 seconds at 16kHz

        # Decision smoothing
        self.decision_buffer = deque(maxlen=5)  # Store last 5 decisions
        self.decision_threshold = 3  # Need 3/5 decisions to change state

        # Debug mode
        self.debug = False

        # Energy threshold to avoid processing silence
        self.energy_threshold = 0.01

        # Fail-safe state tracking
        self.fail_safe_active = False
        self.fail_safe_activations = 0
        self.last_fail_safe_time = 0

        # Statistics
        self.stats = {
            'total_frames': 0,
            'matched_frames': 0,
            'failed_frames': 0,
            'fail_safe_activations': 0,
            'fail_safe_frames': 0
        }

    def set_selected_speaker(self, speaker_id):
        """Set which speaker to allow through"""
        self.selected_speaker_id = speaker_id
        if speaker_id == -1:
            self.block_other_speakers = False
        else:
            self.block_other_speakers = True
        # Clear decision buffer when changing speaker selection
        self.decision_buffer.clear()
        # Reset fail-safe counters
        self.consecutive_matches = 0
        self.consecutive_misses = 0
        self.fail_safe_counter = 0
        self.fail_safe_active = False

    def update_fail_safe_state(self, is_match):
        """Update fail-safe state based on current match"""
        self.stats['total_frames'] += 1

        if is_match:
            self.consecutive_matches += 1
            self.consecutive_misses = 0
            self.stats['matched_frames'] += 1

            # If we have a series of matches, activate fail-safe
            if (self.consecutive_matches >= self.fail_safe_match_threshold and
                    not self.fail_safe_active and
                    self.fail_safe_enabled):

                # Check if we exceed activation limit
                if (time.time() - self.last_fail_safe_time > 10 or
                        self.fail_safe_activations < self.max_fail_safe_attempts):

                    self.fail_safe_counter = self.fail_safe_duration
                    self.fail_safe_active = True
                    self.fail_safe_activations += 1
                    self.last_fail_safe_time = time.time()
                    self.stats['fail_safe_activations'] += 1

                    if self.debug:
                        log(f"[FAIL-SAFE] 🔄 Activation! Passing next {self.fail_safe_duration} samples")
        else:
            self.consecutive_misses += 1
            self.consecutive_matches = 0
            self.stats['failed_frames'] += 1

        # Update fail-safe counter
        if self.fail_safe_counter > 0:
            self.fail_safe_counter -= 1
            self.stats['fail_safe_frames'] += 1
            if self.debug and self.fail_safe_counter == 0:
                log(f"[FAIL-SAFE] ✅ Fail-safe mode ended")
                self.fail_safe_active = False

    def accumulate_audio_for_recognition(self, audio):
        """Accumulate audio for better speaker recognition"""
        # Check if audio has enough energy
        energy = np.mean(audio ** 2)
        if energy < self.energy_threshold:
            return False

        # Add new audio to buffer
        self.recognition_buffer = np.concatenate([self.recognition_buffer, audio])

        # Keep only the last recognition_buffer_size samples
        if len(self.recognition_buffer) > self.recognition_buffer_size:
            self.recognition_buffer = self.recognition_buffer[-self.recognition_buffer_size:]

        return True

    def process_audio(self, audio):
        """Process audio batch with fail-safe mechanism"""
        # First check if audio has enough energy
        energy = np.mean(audio ** 2)
        if energy < self.energy_threshold:
            # If silence and we have a recently recognized speaker, assume it's still the same speaker
            if self.last_recognized_speaker is not None and (time.time() - self.last_recognized_time) < 1.5:
                return self.last_recognized_speaker, True, self.last_similarity, self.recognizer.speaker_names.get(
                    str(self.last_recognized_speaker), f"Speaker_{self.last_recognized_speaker}")
            return None, False, 0.0, "Unknown"

        # Accumulate audio for recognition
        if not self.accumulate_audio_for_recognition(audio):
            return None, False, 0.0, "Unknown"

        # We need at least 1.5 seconds for reliable recognition
        if len(self.recognition_buffer) < 24000:  # 1.5 seconds at 16kHz
            return None, False, 0.0, "Unknown"

        # Use the last 2 seconds for recognition (or whatever is available)
        recognition_audio = self.recognition_buffer[-min(len(self.recognition_buffer), self.recognition_buffer_size):]

        # Normalize audio for better recognition
        if np.max(np.abs(recognition_audio)) > 0:
            recognition_audio = recognition_audio / np.max(np.abs(recognition_audio))

        # Recognize speaker in the accumulated audio
        speaker_id, similarity, speaker_name = self.recognizer.recognize_speaker(recognition_audio)

        # Update last recognized speaker
        if speaker_id is not None:
            self.last_recognized_speaker = speaker_id
            self.last_recognized_time = time.time()
            self.last_similarity = similarity
            self.speaker_activity[speaker_id] = time.time()
            # Keep history of similarities for stability
            self.speaker_history[speaker_id].append(similarity)
            if len(self.speaker_history[speaker_id]) > 5:
                self.speaker_history[speaker_id].pop(0)

        # Determine if this is a match with selected speaker
        is_match = False
        if self.block_other_speakers and self.selected_speaker_id != -1:
            if speaker_id is not None and str(speaker_id) == str(self.selected_speaker_id):
                is_match = True

        # Update fail-safe state
        self.update_fail_safe_state(is_match)

        # Determine if we should pass this audio
        should_pass = False

        # FAIL-SAFE LOGIC: If in fail-safe mode, pass audio
        if self.fail_safe_counter > 0:
            should_pass = True
            if self.debug:
                log(f"[FAIL-SAFE] 🛡️  Passing (remaining: {self.fail_safe_counter})")

        # Normal logic (if not in fail-safe)
        elif self.block_other_speakers and self.selected_speaker_id != -1:
            # Mode: only selected speaker
            if is_match:
                # Add to decision buffer
                self.decision_buffer.append(True)
                should_pass = True
                if self.debug:
                    avg_similarity = np.mean(self.speaker_history.get(speaker_id, [similarity]))
                    log(f"[TRACKER] ✅ Passing audio from '{speaker_name}' (sim={similarity:.3f}, avg={avg_similarity:.3f})")
            else:
                # Add to decision buffer
                self.decision_buffer.append(False)
                if self.debug:
                    if speaker_id is not None:
                        log(f"[TRACKER] ❌ Blocking '{speaker_name}' (not selected, sim={similarity:.3f})")
                    else:
                        log(f"[TRACKER] ❌ Blocking (no speaker recognized)")
        else:
            # Mode: all speakers or auto
            should_pass = True
            if speaker_id is not None and self.debug:
                avg_similarity = np.mean(self.speaker_history.get(speaker_id, [similarity]))
                log(f"[TRACKER] 🔊 Passing '{speaker_name}' (sim={similarity:.3f}, avg={avg_similarity:.3f})")

        # Apply decision smoothing if we have enough history
        if len(self.decision_buffer) == self.decision_buffer.maxlen and not self.fail_safe_active:
            true_count = sum(1 for decision in self.decision_buffer if decision)
            if true_count >= self.decision_threshold:
                should_pass = True
            else:
                should_pass = False

        return speaker_id, should_pass, similarity, speaker_name

    def get_active_speakers(self):
        """Get list of active speakers"""
        active = []
        for speaker_id, last_active in self.speaker_activity.items():
            if time.time() - last_active < 10.0:  # Active within 10s
                name = self.recognizer.speaker_names.get(str(speaker_id), f"Speaker_{speaker_id}")
                avg_similarity = np.mean(self.speaker_history.get(speaker_id, [0])) if self.speaker_history.get(
                    speaker_id) else 0
                active.append({
                    'id': speaker_id,
                    'name': name,
                    'last_active': last_active,
                    'avg_similarity': avg_similarity
                })
        return active

    def list_registered_speakers(self):
        """List all registered speakers"""
        return self.recognizer.list_speakers()

    def get_fail_safe_stats(self):
        """Get fail-safe statistics"""
        if self.stats['total_frames'] == 0:
            return "No data"

        match_rate = self.stats['matched_frames'] / self.stats['total_frames'] * 100
        fail_safe_rate = self.stats['fail_safe_frames'] / self.stats['total_frames'] * 100

        return (f"Match rate: {match_rate:.1f}%, "
                f"Fail-safe: {self.stats['fail_safe_activations']} activations, "
                f"Fail-safe frames: {fail_safe_rate:.1f}%")

    def toggle_fail_safe(self):
        """Toggle fail-safe mode"""
        self.fail_safe_enabled = not self.fail_safe_enabled
        return self.fail_safe_enabled

    def get_speaker_name(self, speaker_id):
        """Get speaker name by ID"""
        return self.recognizer.get_speaker_name(speaker_id)

    def get_all_speakers(self):
        """Get all speaker IDs and names"""
        return self.recognizer.get_all_speakers()


# ============================================
# NON-BLOCKING INPUT - WINDOWS COMPATIBLE
# ============================================

class NonBlockingInput:
    """Non-blocking stdin input for Windows and Unix"""

    def __init__(self):
        self.old_settings = None
        if not IS_WINDOWS:
            self.setup_unix()

    def setup_unix(self):
        """Configure stdin for non-blocking read on Unix"""
        import termios, tty, fcntl, os, sys
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        fd = sys.stdin.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def restore(self):
        """Restore terminal settings"""
        if not IS_WINDOWS and self.old_settings:
            import termios, sys
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """Get key if pressed, otherwise None"""
        if IS_WINDOWS:
            return self._get_key_windows()
        else:
            return self._get_key_unix()

    def _get_key_windows(self):
        """Windows implementation"""
        import msvcrt
        if msvcrt.kbhit():
            try:
                char = msvcrt.getch()
                if isinstance(char, bytes):
                    # Try to decode as UTF-8, fallback to latin-1
                    try:
                        char = char.decode('utf-8')
                    except UnicodeDecodeError:
                        char = char.decode('latin-1')

                # Handle special keys
                if char == '\t':
                    return '\t'
                elif char == '\r':
                    return '\n'
                elif char == '\x03':  # Ctrl+C
                    return 'q'
                elif char == '\x1b':  # ESC
                    return None  # Ignore ESC
                # Handle Shift+key combinations
                elif char in ['G', 'A', 'N', 'I', 'D', 'L', 'S', 'F']:
                    return char.lower()  # Convert to lowercase for consistency
                return char
            except Exception as e:
                # Silent fail on Windows to avoid console spam
                return None
        return None

    def _get_key_unix(self):
        """Unix implementation"""
        import sys, select
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if ord(char) == 9:
                    return '\t'
                return char
        except Exception:
            pass
        return None


# ============================================
# MAIN REAL-TIME PROCESSOR WITH FAIL-SAFE
# ============================================

class RealTimeSpeakerFilter:
    def __init__(self, input_gain=1.0, output_gain=1.0, speaker_gain=1.0, debug_passthrough=False):
        """
        Real-time speaker filtering with fail-safe (NO DENOISING)
        """
        self.input_gain = input_gain
        self.output_gain = output_gain
        self.speaker_gain = speaker_gain
        self.debug_passthrough = debug_passthrough

        # Speaker selection
        self.selected_speaker_id = -1  # -1 = auto/all speakers, specific ID = only that speaker

        # AI speaker tracker with fail-safe
        self.speaker_tracker = AIEnhancedSpeakerTracker()

        # Fail-safe parameters
        self.show_fail_safe_stats = True
        self.last_stats_time = time.time()

        # Debug flag
        self.debug_speech_detection = False

        # Counter for limiting display
        self.process_counter = 0
        self.display_interval = 20  # Display info every 20 processes

        # Audio processing parameters for smooth output
        self.window_size = int(0.5 * SAMPLE_RATE)  # 0.5 second window for processing
        self.hop_size = int(0.25 * SAMPLE_RATE)  # 0.25 second hop (50% overlap)

        log(f"Platform: {platform.system()} ({'Windows' if IS_WINDOWS else 'Unix/Linux'})")
        log(f"Debug passthrough mode: {debug_passthrough}")
        log(f"Using device: {DEVICE}")
        log(f"Speaker gain: {speaker_gain}")
        log(f"Speaker recognition: YOUR MODEL INTEGRATED")
        log(f"Selection mode: BY SPEAKER ID")
        log(f"NO DENOISING - only speaker filtering")
        log(f"FAIL-SAFE: ENABLED ({self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples)")
        log(f"Processing window: {self.window_size} samples ({self.window_size / SAMPLE_RATE * 1000:.0f}ms)")
        log(f"Hop size: {self.hop_size} samples ({self.hop_size / SAMPLE_RATE * 1000:.0f}ms)")
        log(f"Overlap: 50% for smooth audio output")

        # Input buffer for smooth processing (overlap-add)
        self.input_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.input_buffer_ptr = 0

        # Output buffer for smooth output (overlap-add)
        self.output_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.output_ready = np.zeros(self.window_size, dtype=np.float32)
        self.output_ptr = 0

        # Window function for smooth overlap-add
        self.window = np.hanning(self.window_size)

        # Queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=100)

        # Flags
        self.is_running = True
        self.samples_processed = 0
        self.last_audio_time = time.time()

        # Filters
        self.setup_simple_filters()

        # Energy threshold
        self.energy_threshold = 0.005

        log(f"Stream: {SAMPLE_RATE}Hz, Model: {MODEL_SAMPLE_RATE}Hz")

    def setup_simple_filters(self):
        """Simple filters"""
        nyquist = 0.5 * SAMPLE_RATE
        self.b_hp, self.a_hp = signal.butter(2, 80 / nyquist, btype='high')
        self.filter_state = None

    def apply_filter(self, audio):
        """Apply filter"""
        if len(audio) == 0:
            return audio

        if self.filter_state is None:
            filtered, self.filter_state = signal.lfilter(self.b_hp, self.a_hp, audio,
                                                         zi=np.zeros(max(len(self.a_hp), len(self.b_hp)) - 1))
        else:
            filtered, self.filter_state = signal.lfilter(self.b_hp, self.a_hp, audio, zi=self.filter_state)

        return filtered

    def select_speaker(self, speaker_id):
        """Select which speaker to allow through"""
        if speaker_id == -1:
            self.selected_speaker_id = -1
            self.speaker_tracker.set_selected_speaker(-1)
            speaker_name = "Wszyscy"
            log(f"\n🎤 Mode: ALL SPEAKERS (no filtering)")
        else:
            self.selected_speaker_id = speaker_id
            self.speaker_tracker.set_selected_speaker(speaker_id)

            # Find speaker name
            speaker_name = self.speaker_tracker.get_speaker_name(speaker_id)

            log(f"\n🎤 Mode: ONLY SPEAKER '{speaker_name}' (ID: {speaker_id})")
            log(f"   Fail-safe: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")

        return speaker_name

    def select_next_speaker(self):
        """Select next active speaker"""
        active_speakers = self.speaker_tracker.get_active_speakers()
        if not active_speakers:
            log("[INFO] No active speakers")
            return None

        # Find currently selected speaker in active list
        current_index = -1
        if self.selected_speaker_id != -1:
            for i, spk in enumerate(active_speakers):
                if str(spk['id']) == str(self.selected_speaker_id):
                    current_index = i
                    break

        # Select next
        if current_index == -1 or current_index >= len(active_speakers) - 1:
            next_speaker = active_speakers[0]
        else:
            next_speaker = active_speakers[current_index + 1]

        speaker_name = self.select_speaker(next_speaker['id'])
        return speaker_name

    def show_speaker_info(self):
        """Show information about speakers"""
        # Registered speakers
        registered_info = self.speaker_tracker.list_registered_speakers()
        if registered_info:
            log("\n[SPEAKER INFO] Registered speakers:")
            for line in registered_info:
                log(f"  {line}")

        # Active speakers
        active_speakers = self.speaker_tracker.get_active_speakers()
        if active_speakers:
            log("\n[SPEAKER INFO] Currently active speakers:")
            for spk in active_speakers:
                status = "✓ SELECTED" if str(spk['id']) == str(self.selected_speaker_id) else ""
                log(f"  ID: {spk['id']}, Name: '{spk['name']}', "
                    f"Avg similarity: {spk['avg_similarity']:.3f}, "
                    f"Active: {time.time() - spk['last_active']:.1f}s ago {status}")
        else:
            log("\n[SPEAKER INFO] No active speakers")

        # Fail-safe stats
        log(f"\n[FAIL-SAFE] Stats: {self.speaker_tracker.get_fail_safe_stats()}")
        log(f"[FAIL-SAFE] Enabled: {self.speaker_tracker.fail_safe_enabled}")
        log(f"[FAIL-SAFE] Active: {self.speaker_tracker.fail_safe_active}")
        log(f"[FAIL-SAFE] Counter: {self.speaker_tracker.fail_safe_counter}")

    def toggle_speech_debug(self):
        """Toggle debug mode"""
        self.debug_speech_detection = not self.debug_speech_detection
        self.speaker_tracker.debug = self.debug_speech_detection
        log(f"[DEBUG] Debug mode: {'ENABLED' if self.debug_speech_detection else 'DISABLED'}")

    def toggle_fail_safe(self):
        """Toggle fail-safe mode"""
        enabled = self.speaker_tracker.toggle_fail_safe()
        log(f"[FAIL-SAFE] Fail-safe mode: {'ENABLED' if enabled else 'DISABLED'}")

    def adjust_fail_safe_params(self):
        """Adjust fail-safe parameters"""
        log("\n[FAIL-SAFE] Current parameters:")
        log(f"   Matches to activate: {self.speaker_tracker.fail_safe_match_threshold}")
        log(f"   Samples to pass: {self.speaker_tracker.fail_safe_duration}")

        try:
            log(f"[FAIL-SAFE] Keeping current values: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")
        except Exception as e:
            log(f"[FAIL-SAFE] Error: {e}")
            log(f"[FAIL-SAFE] Keeping current values")

    def process_audio_with_speaker_filter(self, audio_chunk_48k):
        """Process audio: speaker recognition + filtering with fail-safe (NO DENOISING)"""
        if self.debug_passthrough:
            return audio_chunk_48k

        # Increase counter and check if we should display info
        self.process_counter += 1
        should_display = (self.process_counter % self.display_interval == 0)

        # Check audio energy
        energy = np.mean(audio_chunk_48k ** 2)
        if energy < self.energy_threshold:
            # Very low energy - return silence
            return np.zeros_like(audio_chunk_48k)

        # 1. Resample to 16kHz
        try:
            audio_16k = resampy.resample(
                audio_chunk_48k,
                SAMPLE_RATE,
                MODEL_SAMPLE_RATE,
                filter='kaiser_fast'
            )
        except Exception as e:
            log(f"Error in resampling to 16k: {e}")
            return audio_chunk_48k

        # 2. SPEAKER RECOGNITION WITH FAIL-SAFE
        if should_display:
            log(f"[PROCESS] Step 1: Speaker recognition (fail-safe: {self.speaker_tracker.fail_safe_enabled})")

        # Recognize speaker in this audio batch (with fail-safe logic)
        speaker_id, should_pass, similarity, speaker_name = self.speaker_tracker.process_audio(audio_16k)

        # 3. FILTER BASED ON SELECTED SPEAKER (with fail-safe consideration)
        if should_pass:
            # Pass the audio (it's from the selected speaker or fail-safe is active)
            selected_audio = audio_16k  # NO DENOISING - use original audio
            if should_display:
                if speaker_id is not None:
                    status = "FAIL-SAFE" if self.speaker_tracker.fail_safe_counter > 0 else "MATCH"
                    log(f"[FILTER] ✅ {status}: Passing audio from '{speaker_name}' (similarity={similarity:.3f})")
                else:
                    if self.speaker_tracker.fail_safe_counter > 0:
                        log(f"[FILTER] 🛡️  FAIL-SAFE: Passing audio (counter: {self.speaker_tracker.fail_safe_counter})")
                    else:
                        log(f"[FILTER] 🔊 Passing audio (no speaker filtering)")
        else:
            # Block audio (not from selected speaker)
            selected_audio = np.zeros_like(audio_16k)
            if should_display:
                if speaker_id is not None:
                    log(f"[FILTER] 🔇 Blocking audio from '{speaker_name}' (not selected)")
                else:
                    log(f"[FILTER] 🔇 Blocking audio (no speaker recognized)")

        # 4. APPLY SPEAKER GAIN
        if should_pass and speaker_id is not None:
            audio_peak = np.max(np.abs(selected_audio))
            if audio_peak > 0:
                selected_audio = selected_audio * min(self.speaker_gain, 0.5 / audio_peak)

        selected_audio = np.clip(selected_audio, -1.0, 1.0)

        # 5. Match length
        if len(selected_audio) > len(audio_16k):
            selected_audio = selected_audio[:len(audio_16k)]
        elif len(selected_audio) < len(audio_16k):
            selected_audio = np.pad(selected_audio,
                                    (0, len(audio_16k) - len(selected_audio)),
                                    mode='constant')

        # 6. Resample back to 48kHz
        try:
            output_48k = resampy.resample(
                selected_audio,
                MODEL_SAMPLE_RATE,
                SAMPLE_RATE,
                filter='kaiser_fast'
            )
        except Exception as e:
            log(f"Error in resampling to 48k: {e}")
            return audio_chunk_48k

        # 7. Apply identity filter (high-pass)
        if should_display:
            log(f"[PROCESS] Step 2: Identity filter")

        output_48k = self.apply_filter(output_48k)

        # 8. Match size
        if len(output_48k) > len(audio_chunk_48k):
            output_48k = output_48k[:len(audio_chunk_48k)]
        elif len(output_48k) < len(audio_chunk_48k):
            output_48k = np.pad(output_48k,
                                (0, len(audio_chunk_48k) - len(output_48k)),
                                mode='constant')

        if should_display:
            fail_safe_status = "ACTIVE" if self.speaker_tracker.fail_safe_counter > 0 else "INACTIVE"
            log(f"[PROCESS] Completed: Recognize → Filter | Fail-safe: {fail_safe_status}")

        return output_48k

    def input_callback(self, indata, frames, time_info, status):
        """Input callback - collect audio chunks"""
        if status:
            log(f"Input status: {status}")

        chunk = indata.copy().flatten()
        chunk = chunk * self.input_gain

        audio_level = np.max(np.abs(chunk))
        if audio_level > 0.01:
            self.last_audio_time = time.time()

        # Add to input queue for processing
        self.input_queue.put(chunk)

    def output_callback(self, outdata, frames, time_info, status):
        """Output callback - send smooth audio output"""
        if status:
            log(f"Output status: {status}")

        try:
            # Get processed audio from output queue
            output_chunk = self.output_queue.get_nowait()

            if len(output_chunk) < frames:
                # Pad if needed
                output_chunk = np.pad(output_chunk, (0, frames - len(output_chunk)), mode='constant')
            elif len(output_chunk) > frames:
                # Trim if needed
                output_chunk = output_chunk[:frames]

            # Apply output gain and clipping
            audio_level = np.max(np.abs(output_chunk))
            if audio_level > 0.01:
                output_chunk = output_chunk * self.output_gain
                max_val = np.max(np.abs(output_chunk))
                if max_val > 1.0:
                    output_chunk = output_chunk / max_val * 0.95

            outdata[:, 0] = output_chunk

        except queue.Empty:
            # No data available, output silence
            outdata.fill(0)

    def keyboard_listener(self, input_handler):
        """Keyboard listener with speaker selection and fail-safe controls"""
        if not self.debug_passthrough:
            log("\n[KEYBOARD] Speaker Recognition Commands:")
            log("[KEYBOARD] 'n' - select next active speaker")
            log("[KEYBOARD] 'a' - ALL speakers mode (no filtering)")
            log("[KEYBOARD] 'i' - show speaker information")
            log("[KEYBOARD] 'd' - toggle debug mode")
            log("[KEYBOARD] 'f' - toggle fail-safe mode")
            log("[KEYBOARD] 'F' (Shift+f) - adjust fail-safe parameters")
            log("[KEYBOARD] 'l' - list registered speakers")
            log("[KEYBOARD] '0'-'9' - select speaker by ID")
            log("[KEYBOARD] 'g' - increase input gain (+0.5)")
            log("[KEYBOARD] 'G' (Shift+g) - decrease input gain (-0.5)")
            log("[KEYBOARD] '+' - increase speaker gain (+1.0)")
            log("[KEYBOARD] '-' - decrease speaker gain (-1.0)")
            log("[KEYBOARD] 't' - set similarity threshold")
            log("[KEYBOARD] 's' - show fail-safe statistics")
            log("[KEYBOARD] 'q' - quit program")
        else:
            log("\n[KEYBOARD] Debug Passthrough Mode")
            log("[KEYBOARD] 'q' - quit program")

        last_key_time = 0
        key_debounce = 0.3

        while self.is_running:
            try:
                key = input_handler.get_key()

                if key:
                    current_time = time.time()

                    if current_time - last_key_time > key_debounce:
                        if key == 'n' and not self.debug_passthrough:
                            speaker_name = self.select_next_speaker()
                            if speaker_name:
                                self.update_speaker_name_callback(speaker_name)
                            last_key_time = current_time
                        elif key == 'a' and not self.debug_passthrough:
                            speaker_name = self.select_speaker(-1)  # All speakers mode
                            self.update_speaker_name_callback(speaker_name)
                            last_key_time = current_time
                        elif key == 'i' and not self.debug_passthrough:
                            self.show_speaker_info()
                            last_key_time = current_time
                        elif key == 'd' and not self.debug_passthrough:
                            self.toggle_speech_debug()
                            last_key_time = current_time
                        elif key == 'f' and not self.debug_passthrough:
                            self.toggle_fail_safe()
                            last_key_time = current_time
                        elif key == 'F' and not self.debug_passthrough:
                            self.adjust_fail_safe_params()
                            last_key_time = current_time
                        elif key == 'l' and not self.debug_passthrough:
                            log("\n[SPEAKERS] Registered speakers:")
                            speakers = self.speaker_tracker.list_registered_speakers()
                            for line in speakers:
                                log(f"  {line}")
                            last_key_time = current_time
                        elif key == 's' and not self.debug_passthrough:
                            log(f"\n[STATS] Fail-safe statistics:")
                            log(f"  {self.speaker_tracker.get_fail_safe_stats()}")
                            log(f"  Consecutive matches: {self.speaker_tracker.consecutive_matches}")
                            log(f"  Fail-safe counter: {self.speaker_tracker.fail_safe_counter}")
                            last_key_time = current_time
                        elif key.isdigit() and not self.debug_passthrough:
                            speaker_id = int(key)
                            speaker_name = self.select_speaker(speaker_id)
                            self.update_speaker_name_callback(speaker_name)
                            last_key_time = current_time
                        elif key == 'g':
                            self.input_gain = min(10.0, self.input_gain + 0.5)
                            log(f"[GAIN] Increased input gain to: {self.input_gain:.1f}")
                            last_key_time = current_time
                        elif key == 'G':
                            self.input_gain = max(0.5, self.input_gain - 0.5)
                            log(f"[GAIN] Decreased input gain to: {self.input_gain:.1f}")
                            last_key_time = current_time
                        elif key == '+':
                            self.speaker_gain = min(10.0, self.speaker_gain + 1.0)
                            log(f"[GAIN] Increased speaker gain to: {self.speaker_gain:.1f}")
                            last_key_time = current_time
                        elif key == '-':
                            self.speaker_gain = max(1.0, self.speaker_gain - 1.0)
                            log(f"[GAIN] Decreased speaker gain to: {self.speaker_gain:.1f}")
                            last_key_time = current_time
                        elif key == 't' and not self.debug_passthrough:
                            try:
                                log(f"\n[THRESHOLD] Current similarity threshold: {self.speaker_tracker.recognizer.similarity_threshold:.2f}")
                                # For simplicity, we'll just cycle through some values
                                if self.speaker_tracker.recognizer.similarity_threshold < 0.4:
                                    new_threshold = 0.4
                                elif self.speaker_tracker.recognizer.similarity_threshold < 0.6:
                                    new_threshold = 0.6
                                else:
                                    new_threshold = 0.3

                                self.speaker_tracker.recognizer.similarity_threshold = new_threshold
                                log(f"[THRESHOLD] Set similarity threshold to: {new_threshold:.2f}")
                            except:
                                pass
                            last_key_time = current_time
                        elif key == 'q':
                            log("\n[KEYBOARD] Detected 'q' - quitting...")
                            self.is_running = False
                            break

                time.sleep(0.01)

            except Exception as e:
                log(f"[KEYBOARD] Listener error: {e}")
                time.sleep(0.1)

    def set_update_speaker_name_callback(self, callback):
        """Set callback to update speaker name in UI"""
        self.update_speaker_name_callback = callback

    def processing_loop(self):
        """Main processing loop with overlap-add for smooth audio"""
        log("Processing loop started...")

        if self.debug_passthrough:
            log("DEBUG PASSTHROUGH MODE: Direct audio passthrough")
            while self.is_running:
                try:
                    raw_chunk = self.input_queue.get(timeout=0.1)
                    self.samples_processed += len(raw_chunk)

                    try:
                        self.output_queue.put_nowait(raw_chunk)
                    except queue.Full:
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(raw_chunk)
                        except queue.Empty:
                            pass

                except queue.Empty:
                    continue
                except Exception as e:
                    log(f"Processing loop error: {e}")
        else:
            # Buffer for overlap-add processing
            input_buffer = np.zeros(self.window_size, dtype=np.float32)
            output_buffer = np.zeros(self.window_size, dtype=np.float32)
            output_accumulator = np.zeros(self.window_size, dtype=np.float32)

            # Pointer in the buffer
            buffer_ptr = 0

            # Counter for hop processing
            samples_since_last_process = 0

            while self.is_running:
                try:
                    # Get input chunk
                    try:
                        raw_chunk = self.input_queue.get(timeout=0.1)
                    except queue.Empty:
                        # If no data, continue to process any remaining audio in buffer
                        if buffer_ptr > 0 and samples_since_last_process >= self.hop_size:
                            # Process what we have in buffer
                            windowed = input_buffer * self.window
                            processed = self.process_audio_with_speaker_filter(windowed)

                            # Overlap-add to output accumulator
                            output_accumulator[:] += processed * self.window

                            # Send first hop_size samples to output
                            output_to_send = output_accumulator[:self.hop_size].copy()

                            # Shift buffers
                            input_buffer[:-self.hop_size] = input_buffer[self.hop_size:]
                            input_buffer[-self.hop_size:] = 0

                            output_accumulator[:-self.hop_size] = output_accumulator[self.hop_size:]
                            output_accumulator[-self.hop_size:] = 0

                            buffer_ptr = max(0, buffer_ptr - self.hop_size)
                            samples_since_last_process = 0

                            # Send output in smaller chunks for smooth playback
                            chunk_size = CHUNK_SIZE
                            for i in range(0, len(output_to_send), chunk_size):
                                chunk = output_to_send[i:i + chunk_size]
                                if len(chunk) > 0:
                                    try:
                                        self.output_queue.put_nowait(chunk)
                                    except queue.Full:
                                        try:
                                            self.output_queue.get_nowait()
                                            self.output_queue.put_nowait(chunk)
                                        except queue.Empty:
                                            pass
                        continue

                    self.samples_processed += len(raw_chunk)

                    # Add chunk to buffer
                    if buffer_ptr + len(raw_chunk) <= self.window_size:
                        input_buffer[buffer_ptr:buffer_ptr + len(raw_chunk)] = raw_chunk
                        buffer_ptr += len(raw_chunk)
                    else:
                        # Buffer overflow, shift and add
                        shift = len(raw_chunk)
                        input_buffer[:-shift] = input_buffer[shift:]
                        input_buffer[-shift:] = raw_chunk
                        buffer_ptr = self.window_size

                    samples_since_last_process += len(raw_chunk)

                    # Process when we have enough samples for a hop
                    if samples_since_last_process >= self.hop_size and buffer_ptr >= self.window_size:
                        # Apply window and process
                        windowed = input_buffer * self.window
                        processed = self.process_audio_with_speaker_filter(windowed)

                        # Overlap-add to output accumulator
                        output_accumulator[:] += processed * self.window

                        # Send first hop_size samples to output
                        output_to_send = output_accumulator[:self.hop_size].copy()

                        # Shift buffers
                        input_buffer[:-self.hop_size] = input_buffer[self.hop_size:]
                        input_buffer[-self.hop_size:] = 0

                        output_accumulator[:-self.hop_size] = output_accumulator[self.hop_size:]
                        output_accumulator[-self.hop_size:] = 0

                        buffer_ptr = max(0, buffer_ptr - self.hop_size)
                        samples_since_last_process = 0

                        # Send output in smaller chunks for smooth playback
                        chunk_size = CHUNK_SIZE
                        for i in range(0, len(output_to_send), chunk_size):
                            chunk = output_to_send[i:i + chunk_size]
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    try:
                                        self.output_queue.get_nowait()
                                        self.output_queue.put_nowait(chunk)
                                    except queue.Empty:
                                        pass

                except Exception as e:
                    log(f"Processing loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)

    def run(self, input_device=None, output_device=None):
        """Run the processor"""
        log("\n" + "=" * 60)
        log("Real-time Speaker Filtering with FAIL-SAFE")
        log("=" * 60)
        log(f"Platform: {platform.system()} ({'Windows' if IS_WINDOWS else 'Unix/Linux'})")
        log(f"Speaker recognition: YOUR MODEL INTEGRATED")
        log(f"Processing: Recognize → Filter (NO DENOISING)")
        log(f"Fail-safe: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")
        log(f"Current speaker: {'ALL (no filtering)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
        log(f"Speaker gain: {self.speaker_gain}x (auto-adjusted)")
        log(f"Input gain: {self.input_gain}")
        log(f"Sample rate: {SAMPLE_RATE} Hz")
        log(f"Processing window: {self.window_size} samples ({self.window_size / SAMPLE_RATE * 1000:.0f}ms)")
        log(f"Hop size: {self.hop_size} samples ({self.hop_size / SAMPLE_RATE * 1000:.0f}ms)")
        log(f"Overlap: 50% for smooth audio")
        log(f"Input device: {input_device or 'default'}")
        log(f"Output device: {output_device or 'default'}")
        log("\nPress Ctrl+C or 'q' to stop\n")

        input_handler = NonBlockingInput()

        keyboard_thread = threading.Thread(target=self.keyboard_listener, args=(input_handler,), daemon=True)
        keyboard_thread.start()

        processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        processing_thread.start()

        try:
            with sd.InputStream(
                    device=input_device,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=CHUNK_SIZE,
                    dtype='float32',
                    callback=self.input_callback
            ), sd.OutputStream(
                device=output_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype='float32',
                callback=self.output_callback
            ):
                if self.debug_passthrough:
                    log("🎤 DEBUG: Passthrough mode running!")
                    log("   You should hear your microphone input directly")
                else:
                    log("🎤 AI Speaker Filtering System running!")
                    log(f"   Selected speaker: {'ALL (no filtering)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
                    log(f"   Fail-safe: ENABLED ({self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples)")
                    log(f"   Processing pipeline: Recognize → Filter (NO DENOISING)")
                    log(f"   Display interval: {self.display_interval} windows")
                    log(f"   Audio processing: 50% overlap-add for smooth output")
                    log(f"   Use 'n' to select next active speaker")
                    log(f"   Use 'a' for ALL speakers mode (no filtering)")
                    log(f"   Use '0'-'9' to select speaker by ID")
                    log(f"   Use 'i' to show speaker information")
                    log(f"   Use 'l' to list registered speakers")
                    log(f"   Use 'd' to toggle debug mode")
                    log(f"   Use 'f' to toggle fail-safe mode")
                    log(f"   Use 'F' to adjust fail-safe parameters")
                    log(f"   Use '+'/'-' to adjust speaker gain")
                    log(f"   Use 't' to adjust similarity threshold (currently: {self.speaker_tracker.recognizer.similarity_threshold:.2f})")
                    log(f"   Use 's' to show fail-safe statistics")
                    if IS_WINDOWS:
                        log(f"   Note: On Windows, press keys directly (no Enter needed)")

                last_status_time = time.time()

                while self.is_running:
                    time.sleep(0.1)
                    now = time.time()

                    if now - last_status_time >= 5:
                        time_since_audio = now - self.last_audio_time

                        if time_since_audio < 5:
                            audio_status = "🎤 Audio active"
                        else:
                            audio_status = "🔇 No audio"

                        # Active speakers
                        active_speakers = self.speaker_tracker.get_active_speakers()
                        active_count = len(active_speakers)

                        selected_info = ""
                        if self.selected_speaker_id == -1:
                            selected_info = "Selected: ALL (no filtering)"
                        else:
                            # Find speaker name
                            speaker_name = f"Speaker_{self.selected_speaker_id}"
                            for spk in active_speakers:
                                if str(spk['id']) == str(self.selected_speaker_id):
                                    speaker_name = spk['name']
                                    break
                            selected_info = f"Selected: {speaker_name}"

                        # Fail-safe status
                        fail_safe_status = ""
                        if self.speaker_tracker.fail_safe_counter > 0:
                            fail_safe_status = f"FAIL-SAFE: {self.speaker_tracker.fail_safe_counter}"

                        log(f"Status: {audio_status}, "
                            f"active_speakers={active_count}, "
                            f"{selected_info}, "
                            f"{fail_safe_status}, "
                            f"threshold={self.speaker_tracker.recognizer.similarity_threshold:.2f}, "
                            f"input_gain={self.input_gain:.1f}, "
                            f"processed={self.samples_processed}")
                        last_status_time = now

        except KeyboardInterrupt:
            log("\n\nStopping processor...")
        except Exception as e:
            log(f"Audio stream error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            processing_thread.join(timeout=1.0)
            keyboard_thread.join(timeout=0.5)
            input_handler.restore()

            # Show final statistics
            log(f"\n📊 FINAL STATISTICS:")
            log(f"   Total samples processed: {self.samples_processed}")
            log(f"   {self.speaker_tracker.get_fail_safe_stats()}")
            log(f"   Consecutive matches: {self.speaker_tracker.consecutive_matches}")
            log(f"   Fail-safe activations: {self.speaker_tracker.stats['fail_safe_activations']}")

            log(f"\nProcessor stopped.")


# ============================================
# MAIN WINDOW CLASS
# ============================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # First set window size
        self.setWindowTitle("Wybierz Mówcę")
        self.setGeometry(100, 100, 520, 661)

        # Current parameters
        self.params = DEFAULT_PARAMS.copy()

        # Configuration window
        self.config_window = None

        # Registration window
        self.registration_window = None

        # Audio processor thread
        self.processor_thread = None
        self.is_restarting = False

        # Load UI
        self.ui = None
        self.load_ui()

        # If UI didn't load, create fallback
        if self.ui is None:
            self.create_fallback_ui()
        else:
            self.fix_ui_problems()

        self.setup_connections()

        # Set logger to send messages to UI
        ui_logger.set_main_window(self)
        ui_logger.log_signal.connect(self.append_to_console)

        # Initialize speaker recognizer for checking IDs
        self.speaker_recognizer = SpeakerRecognizer()

        # Current speaker name
        self.current_speaker_name = "Wszyscy"

        # Teraz mamy menu i akcje, więc możemy podpiąć RegisterSpeaker
        self.connect_menu_actions()

    def connect_menu_actions(self):
        """Connect menu actions to their handlers"""
        # Szukamy akcji RegisterSpeaker
        for action in self.ui.menubar.actions():
            # Przejdź przez wszystkie menu
            if action.menu():
                menu = action.menu()
                # Szukaj menu "Rejestracja"
                if menu.title() == "Rejestracja":
                    # Znajdź akcję "RegisterSpeaker" w tym menu
                    for act in menu.actions():
                        if act.text() == "Rejestrój Mówcę":
                            act.triggered.connect(self.open_register_speaker)
                            log("✅ Podłączono akcję 'Rejestrój Mówcę' do okna rejestracji")
                            break
                    break

    def open_register_speaker(self):
        """Open speaker registration window"""
        try:
            if self.registration_window is None or not self.registration_window.isVisible():
                self.registration_window = RegistrationWindow(self)

                # PODŁĄCZENIE SYGNAŁU DO PRZEŁADOWANIA BAZY DANYCH
                self.registration_window.database_updated.connect(self.reload_speaker_database)

                self.registration_window.show()
                log("Okno rejestracji mówcy otwarte")
            else:
                self.registration_window.raise_()
                self.registration_window.activateWindow()
        except Exception as e:
            log(f"Błąd otwierania okna rejestracji: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie można otworzyć okna rejestracji:\n{e}")

    def reload_speaker_database(self):
        """Reload speaker database after registration - DODANA METODA"""
        log("🔄 Przeładowywanie bazy danych mówców...")

        # Przeładuj bazę w głównym recognizerze
        if self.speaker_recognizer.reload_database():
            log("✅ Baza danych mówców przeładowana w głównym systemie")
        else:
            log("❌ Nie udało się przeładować bazy danych mówców")

        # Jeśli procesor jest uruchomiony, przeładuj też jego bazę
        if self.processor_thread and self.processor_thread.processor:
            try:
                # Przeładuj bazę w trackerze procesora
                if self.processor_thread.processor.speaker_tracker.recognizer.reload_database():
                    log("✅ Baza danych mówców przeładowana w procesorze audio")
                else:
                    log("❌ Nie udało się przeładować bazy danych w procesorze")

                # Zaktualizuj listę dostępnych mówców w UI
                all_speakers = self.speaker_recognizer.get_all_speakers()
                if all_speakers:
                    log(f"📋 Aktualna lista mówców ({len(all_speakers)}):")
                    for speaker_id, name in all_speakers.items():
                        log(f"  ID {speaker_id}: {name}")

                    # Jeśli mamy lineEdit, zaktualizuj sugestie
                    if hasattr(self.ui, 'lineEdit'):
                        current_text = self.ui.lineEdit.text().strip()
                        if current_text and current_text != "Wszyscy":
                            # Sprawdź czy obecnie wybrany mówca nadal istnieje
                            found = False
                            for speaker_id, name in all_speakers.items():
                                if name == current_text or speaker_id == current_text:
                                    found = True
                                    break

                            if not found:
                                log(f"⚠️ Obecnie wybrany mówca '{current_text}' nie istnieje w bazie")
                                self.ui.lineEdit.setText("Wszyscy")
                                self.current_speaker_name = "Wszyscy"

                                # Jeśli procesor działa, ustaw tryb "wszyscy"
                                if self.processor_thread and self.processor_thread.processor:
                                    self.processor_thread.processor.select_speaker(-1)
            except Exception as e:
                log(f"❌ Błąd przy przeładowywaniu bazy w procesorze: {e}")

    def append_to_console(self, message):
        """Append message to ConsoleOutput"""
        if hasattr(self.ui, 'ConsoleOutput'):
            self.ui.ConsoleOutput.append(message)

    def load_ui(self):
        """Load .ui file with QUiLoader"""
        try:
            loader = QUiLoader()
            ui_file = "menu.ui"

            if not os.path.exists(ui_file):
                log(f"ERROR: Nie znaleziono pliku {ui_file}")
                return

            file = QFile(ui_file)
            if not file.open(QFile.ReadOnly):
                log(f"ERROR: Nie można otworzyć pliku")
                return

            self.ui = loader.load(file, self)
            file.close()

            if self.ui:
                log("SUCCESS: UI załadowane")
            else:
                log("ERROR: loader.load() zwrócił None")

        except Exception as e:
            log(f"ERROR w load_ui: {e}")

    def fix_ui_problems(self):
        """Fix common problems with loaded UI"""
        # 1. Make sure centralwidget exists
        if hasattr(self.ui, 'centralwidget'):
            central = self.ui.centralwidget

            # 2. Check if centralwidget has geometry set
            if central.geometry().width() == 0:
                central.setGeometry(QRect(0, 0, 520, 661))

            # 3. Check if centralwidget is visible
            if not central.isVisible():
                central.setVisible(True)

            # 4. Set as central widget
            if self.centralWidget() != central:
                self.setCentralWidget(central)

        # 5. Check if menubar exists and set it
        if hasattr(self.ui, 'menubar'):
            log(f"Znaleziono menubar: {self.ui.menubar}")
            # Set main window menubar
            self.setMenuBar(self.ui.menubar)

            # Check if menu exists
            if hasattr(self.ui, 'menuParametry'):
                log(f"Znaleziono menuParametry: {self.ui.menubar}")
                # Now we need to find actions in menu
                # This is key part - QUiLoader may not load actions properly
                # So we create them manually
                self.create_menu_actions()

        # 6. Force layout refresh
        self.centralWidget().updateGeometry()

    def create_menu_actions(self):
        """Create menu actions manually since QUiLoader doesn't load them properly"""
        try:
            # Find Parametry menu
            menu_parametry = self.ui.menubar.findChild(QMenu, "menuParametry")
            if not menu_parametry:
                # If not found, try another way
                for action in self.ui.menubar.actions():
                    if action.menu() and action.menu().title() == "Parametry":
                        menu_parametry = action.menu()
                        break

            if menu_parametry:
                log(f"Znaleziono menu Parametry: {menu_parametry}")

                # Clear existing actions (may be incorrectly loaded)
                menu_parametry.clear()

                # Create new actions
                configure_action = QAction("Konfiguruj Parametry", self)
                configure_action.triggered.connect(self.open_config_window)
                menu_parametry.addAction(configure_action)

                restore_action = QAction("Przywróć Domyślne", self)
                restore_action.triggered.connect(self.restore_default_params)
                menu_parametry.addAction(restore_action)

                log("Utworzono akcje menu Parametry ręcznie")
            else:
                log("Nie znaleziono menu Parametry, tworzę nowe")
                # Create new menu
                menu_parametry = QMenu("Parametry", self)

                configure_action = QAction("Konfiguruj Parametry", self)
                configure_action.triggered.connect(self.open_config_window)
                menu_parametry.addAction(configure_action)

                restore_action = QAction("Przywróć Domyślne", self)
                restore_action.triggered.connect(self.restore_default_params)
                menu_parametry.addAction(restore_action)

                # Add menu to menubar
                self.ui.menubar.addMenu(menu_parametry)

            # Znajdź lub utwórz menu Rejestracja
            menu_rejestracja = self.ui.menubar.findChild(QMenu, "menuRejestracja")
            if not menu_rejestracja:
                for action in self.ui.menubar.actions():
                    if action.menu() and action.menu().title() == "Rejestracja":
                        menu_rejestracja = action.menu()
                        break

            if menu_rejestracja:
                log(f"Znaleziono menu Rejestracja: {menu_rejestracja}")

                # Clear existing actions
                menu_rejestracja.clear()

                # Create register speaker action
                register_action = QAction("Rejestrój Mówcę", self)
                register_action.triggered.connect(self.open_register_speaker)
                menu_rejestracja.addAction(register_action)

                log("Utworzono akcję 'Rejestrój Mówcę' w menu Rejestracja")
            else:
                log("Nie znaleziono menu Rejestracja, tworzę nowe")
                # Create new menu
                menu_rejestracja = QMenu("Rejestracja", self)

                register_action = QAction("Rejestrój Mówcę", self)
                register_action.triggered.connect(self.open_register_speaker)
                menu_rejestracja.addAction(register_action)

                # Add menu to menubar
                self.ui.menubar.addMenu(menu_rejestracja)

        except Exception as e:
            log(f"Błąd tworzenia akcji menu: {e}")

    def create_fallback_ui(self):
        """Create fallback UI if loading fails"""
        log("Tworzenie awaryjnego UI...")
        self.ui = QWidget()
        self.setCentralWidget(self.ui)
        layout = QVBoxLayout(self.ui)

        # Create all necessary widgets
        grid_layout = QVBoxLayout()

        # Input device
        self.InputDv = QComboBox()
        grid_layout.addWidget(QLabel("Urządzenie Wejścia"))
        grid_layout.addWidget(self.InputDv)

        # Output device
        self.OutputDv = QComboBox()
        grid_layout.addWidget(QLabel("Urządzenie Wyjścia"))
        grid_layout.addWidget(self.OutputDv)

        layout.addLayout(grid_layout)

        # Console output
        self.ConsoleOutput = QTextEdit()
        self.ConsoleOutput.setReadOnly(True)
        layout.addWidget(self.ConsoleOutput)

        # Status label and speaker selection
        status_layout = QHBoxLayout()
        self.Status = QLabel("INFO")
        self.Status.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.Status)

        # Add spacer
        status_layout.addStretch()

        # Speaker selection
        status_layout.addWidget(QLabel("Obecny Mówca:"))
        self.lineEdit = QLineEdit()
        self.lineEdit.setPlaceholderText("Wpisz ID mówcy")
        self.lineEdit.returnPressed.connect(self.on_speaker_id_entered)
        status_layout.addWidget(self.lineEdit)

        layout.addLayout(status_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.OnButton = QPushButton("Włącz")
        self.OffButton = QPushButton("Wyłącz")
        self.checkBox = QCheckBox("DebugMode")
        buttons_layout.addWidget(self.OnButton)
        buttons_layout.addWidget(self.checkBox)
        buttons_layout.addWidget(self.OffButton)
        layout.addLayout(buttons_layout)

        # Map names for compatibility
        self.ui.ConsoleOutput = self.ConsoleOutput
        self.ui.Status = self.Status
        self.ui.OnButton = self.OnButton
        self.ui.OffButton = self.OffButton
        self.ui.InputDv = self.InputDv
        self.ui.OutputDv = self.OutputDv
        self.ui.checkBox = self.checkBox
        self.ui.lineEdit = self.lineEdit

        # Create menu bar
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # Create Parametry menu
        menu_parametry = QMenu("Parametry", self)
        menubar.addMenu(menu_parametry)

        # Create actions
        configure_action = QAction("Konfiguruj Parametry", self)
        configure_action.triggered.connect(self.open_config_window)
        menu_parametry.addAction(configure_action)

        restore_action = QAction("Przywróć Domyślne", self)
        restore_action.triggered.connect(self.restore_default_params)
        menu_parametry.addAction(restore_action)

        # Create Rejestracja menu
        menu_rejestracja = QMenu("Rejestracja", self)
        menubar.addMenu(menu_rejestracja)

        # Create register action
        register_action = QAction("Rejestrój Mówcę", self)
        register_action.triggered.connect(self.open_register_speaker)
        menu_rejestracja.addAction(register_action)

        self.ui.menubar = menubar

        log("Awaryjne UI utworzone")

    def get_clean_display_name(self, device_name):
        """
        Cleans device name but preserves information in parentheses
        """
        if not device_name:
            return ""

        name = str(device_name)

        # 1. Remove ONLY technical information about channels, frequencies etc.
        patterns_to_remove = [
            r'\(in:\s*\d+ch[^)]*\)',  # (in: 2ch, 44100Hz)
            r'\(out:\s*\d+ch[^)]*\)',  # (out: 2ch, 44100Hz)
            r'\(channels:\s*\d+[^)]*\)',
            r'\(\d+ch[^)]*\)',
            r'\(\d+Hz[^)]*\)',
            r'\(samplerate[^)]*\)',
            r'\(latency[^)]*\)',
            r'#\d+',  # #0, #4, etc - but only if separate token
        ]

        for pattern in patterns_to_remove:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        # 2. Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # 3. If name contains " - X - " (for monitors), simplify
        if ' - ' in name:
            parts = name.split(' - ')
            if len(parts) >= 2 and parts[0].strip().isdigit():
                # Format: "1 - MSI MP242A E2 (AMD High Definition Audio Device)"
                # Keep only part after number
                rest = ' - '.join(parts[1:])
                name = rest

        # 4. Keep rest of parentheses - these are important device info
        # 5. Remove endings like "Device", "Audio" if redundant
        name = re.sub(r'\s+Device$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+Audio$', '', name, flags=re.IGNORECASE)

        # 6. Capitalize first letter
        if name:
            name = name[0].upper() + name[1:]

        return name

    def get_base_device_key(self, device_name, device_type):
        """
        Returns base key for grouping devices
        """
        if not device_name:
            return ""

        name_lower = str(device_name).lower()

        # For virtual/unuseful devices
        virtual_keywords = ['microsoft sound mapper', 'primary', 'podstawowy', 'default', 'virtual']
        if any(keyword in name_lower for keyword in virtual_keywords):
            return "virtual"

        # For audio outputs - group similar monitors
        if device_type == 'output':
            # Recognize monitors by format "X - Name"
            if re.match(r'^\d+\s*-\s*', device_name, re.IGNORECASE):
                # Extract monitor name
                monitor_match = re.search(r'\d+\s*-\s*(.*?)\s*\(', device_name)
                if monitor_match:
                    monitor_name = monitor_match.group(1).strip()
                    # Remove redundant words
                    monitor_name = re.sub(r'\b(AMD|NVIDIA|High Definition|Audio|Device)\b', '', monitor_name,
                                          flags=re.IGNORECASE)
                    monitor_name = re.sub(r'\s+', ' ', monitor_name).strip()
                    if monitor_name:
                        return f"monitor_{monitor_name}"

            # For output ports (HDMI, DP)
            if 'hdmi' in name_lower:
                return "hdmi"
            elif 'dp' in name_lower:
                return "dp"
            elif 'headphone' in name_lower or 'słuchawki' in name_lower:
                return "headphones"
            elif 'speaker' in name_lower or 'głośnik' in name_lower:
                # Distinguish Realtek and Buzzard
                if 'realtek' in name_lower:
                    return "speakers_realtek"
                elif 'buzzard' in name_lower:
                    return "speakers_buzzard"
                else:
                    return "speakers_other"

        # For inputs
        if device_type == 'input':
            if 'microphone' in name_lower or 'mikrofon' in name_lower:
                # Distinguish Buzzard and Realtek
                if 'buzzard' in name_lower:
                    return "mic_buzzard"
                elif 'realtek' in name_lower:
                    return "mic_realtek"
                else:
                    return "mic_other"
            elif 'line' in name_lower or 'liniowe' in name_lower:
                return "line_input"

        # Default: use cleaned name as key
        return self.get_clean_display_name(device_name)

    def is_useful_device(self, device_name, device_type):
        """
        Checks if device is useful
        """
        name_lower = str(device_name).lower()

        # List of devices that are NOT useful
        useless_keywords = [
            'microsoft sound mapper',
            'primary sound driver',
            'podstawowy sterownik',
            'default',
            'mapper',
            'sound driver',
            'audio driver',
            'virtual',
            'loopback',
            'stereo mix',
            'miks stereo',
            'what you hear',
            'waveout',
            'wdma',
            'disabled',
            'nieaktywne'
        ]

        # Check if name contains any useless words
        for keyword in useless_keywords:
            if keyword in name_lower:
                return False

        # Additional filters for types
        if device_type == 'input':
            # Only devices with microphone or line input
            useful_inputs = ['mic', 'microphone', 'mikrofon', 'line', 'liniowe']
            if not any(keyword in name_lower for keyword in useful_inputs):
                return False

        return True

    def setup_connections(self):
        """Configure signal connections"""
        if not self.ui:
            return

        # Make sure widgets exist and are accessible
        widgets_to_check = ['OnButton', 'OffButton', 'InputDv', 'OutputDv',
                            'ConsoleOutput', 'Status', 'checkBox', 'lineEdit']

        for widget_name in widgets_to_check:
            if hasattr(self.ui, widget_name):
                widget = getattr(self.ui, widget_name)
                log(f"✓ Znaleziono: {widget_name} - {type(widget).__name__}")

                # Set visibility for sure
                widget.setVisible(True)
                widget.setEnabled(True)
            else:
                log(f"✗ BRAK: {widget_name}")

        # Connections (only if widgets exist)
        if hasattr(self.ui, 'OnButton'):
            self.ui.OnButton.clicked.connect(self.on_on_clicked)

        if hasattr(self.ui, 'OffButton'):
            self.ui.OffButton.clicked.connect(self.on_off_clicked)

        if hasattr(self.ui, 'InputDv'):
            self.setup_input_devices()

        if hasattr(self.ui, 'OutputDv'):
            self.setup_output_devices()

        if hasattr(self.ui, 'checkBox'):
            self.ui.checkBox.stateChanged.connect(self.on_debug_mode_changed)
            log(f"CheckBox znaleziony, stan początkowy: {'zaznaczony' if self.ui.checkBox.isChecked() else 'odznaczony'}")

        # Connect lineEdit for speaker ID input
        if hasattr(self.ui, 'lineEdit'):
            self.ui.lineEdit.returnPressed.connect(self.on_speaker_id_entered)
            self.ui.lineEdit.setPlaceholderText("Wpisz ID mówcy")

    def setup_input_devices(self):
        """Configure and populate InputDv ComboBox"""
        try:
            devices = sd.query_devices()

            # Clear ComboBox
            self.ui.InputDv.clear()

            # Add default device at the top (if exists)
            default_input_id = sd.default.device[0]
            default_added = False

            if default_input_id is not None and default_input_id < len(devices):
                default_device = devices[default_input_id]
                if default_device['max_input_channels'] > 0:
                    display_name = self.get_clean_display_name(default_device['name'])
                    display_text = f"⭐ {display_name} (domyślnie)"
                    self.ui.InputDv.addItem(display_text, default_input_id)
                    default_added = True

            # Add other input devices
            for i, device in enumerate(devices):
                # Skip default device if already added
                if default_added and i == default_input_id:
                    continue

                channels = device['max_input_channels']

                # Only input devices
                if channels == 0:
                    continue

                # Check if device is useful
                if not self.is_useful_device(device['name'], 'input'):
                    continue

                display_name = self.get_clean_display_name(device['name'])
                self.ui.InputDv.addItem(display_name, i)

            # If empty, add information
            if self.ui.InputDv.count() == 0:
                self.ui.InputDv.addItem("Brak urządzeń wejściowych", None)

            # Connect signal
            self.ui.InputDv.currentTextChanged.connect(self.on_input_changed)

        except Exception as e:
            log(f"Błąd konfiguracji InputDv: {e}")
            self.ui.InputDv.clear()
            self.ui.InputDv.addItem(f"Błąd: {str(e)}", None)

    def setup_output_devices(self):
        """Configure and populate OutputDv ComboBox"""
        try:
            devices = sd.query_devices()

            # Clear ComboBox
            self.ui.OutputDv.clear()

            # Add default device at the top (if exists)
            default_output_id = sd.default.device[1]
            default_added = False

            if default_output_id is not None and default_output_id < len(devices):
                default_device = devices[default_output_id]
                if default_device['max_output_channels'] > 0:
                    display_name = self.get_clean_display_name(default_device['name'])
                    display_text = f"⭐ {display_name} (domyślnie)"
                    self.ui.OutputDv.addItem(display_text, default_output_id)
                    default_added = True

            # Add other output devices
            for i, device in enumerate(devices):
                # Skip default device if already added
                if default_added and i == default_output_id:
                    continue

                channels = device['max_output_channels']

                # Only output devices
                if channels == 0 and "VB" not in device['name']:
                    continue

                # Check if device is useful
                if not self.is_useful_device(device['name'], 'output'):
                    continue

                display_name = self.get_clean_display_name(device['name'])
                self.ui.OutputDv.addItem(display_name, i)

            # If empty, add information
            if self.ui.OutputDv.count() == 0:
                self.ui.OutputDv.addItem("Brak urządzeń wyjściowych", None)

            # Connect signal
            self.ui.OutputDv.currentTextChanged.connect(self.on_output_changed)

        except Exception as e:
            log(f"Błąd konfiguracji OutputDv: {e}")
            self.ui.OutputDv.clear()
            self.ui.OutputDv.addItem(f"Błąd: {str(e)}", None)

    def on_speaker_id_entered(self):
        """Handle speaker ID entered in lineEdit"""
        if not hasattr(self.ui, 'lineEdit'):
            return

        speaker_id_text = self.ui.lineEdit.text().strip()

        if not speaker_id_text:
            # Empty input - set to all speakers
            if self.processor_thread and self.processor_thread.processor:
                speaker_name = self.processor_thread.processor.select_speaker(-1)
                self.ui.lineEdit.setText(speaker_name)
                self.current_speaker_name = speaker_name
                log(f"Ustawiono wszystkich mówców")
            return

        # Check if input is a number (speaker ID)
        if speaker_id_text.isdigit():
            speaker_id = int(speaker_id_text)

            # Check if speaker exists in database
            all_speakers = self.speaker_recognizer.get_all_speakers()

            if str(speaker_id) in all_speakers:
                # Speaker exists - select it
                speaker_name = all_speakers[str(speaker_id)]

                if self.processor_thread and self.processor_thread.processor:
                    selected_name = self.processor_thread.processor.select_speaker(speaker_id)
                    self.ui.lineEdit.setText(selected_name)
                    self.current_speaker_name = selected_name
                    log(f"Wybrano mówcę: {selected_name} (ID: {speaker_id})")
                else:
                    self.ui.lineEdit.setText(speaker_name)
                    self.current_speaker_name = speaker_name
                    log(f"Znaleziono mówcę: {speaker_name} (ID: {speaker_id}) - uruchom system aby aktywować")
            else:
                # Speaker doesn't exist
                QMessageBox.warning(self, "Nie znaleziono mówcy",
                                    f"Mówca o ID {speaker_id} nie istnieje w bazie danych.\n"
                                    f"Dostępni mówcy: {', '.join([f'{k}: {v}' for k, v in all_speakers.items()])}")
                self.ui.lineEdit.setText(self.current_speaker_name)
        else:
            # Not a number - check if it's "all" or "wszyscy"
            lower_text = speaker_id_text.lower()
            if lower_text in ["all", "wszyscy", "wszystkich", "każdy"]:
                if self.processor_thread and self.processor_thread.processor:
                    speaker_name = self.processor_thread.processor.select_speaker(-1)
                    self.ui.lineEdit.setText(speaker_name)
                    self.current_speaker_name = speaker_name
                    log(f"Ustawiono wszystkich mówców")
            else:
                # Try to find speaker by name
                all_speakers = self.speaker_recognizer.get_all_speakers()
                found_id = None

                for speaker_id, name in all_speakers.items():
                    if name.lower() == speaker_id_text.lower():
                        found_id = int(speaker_id)
                        break

                if found_id is not None:
                    if self.processor_thread and self.processor_thread.processor:
                        selected_name = self.processor_thread.processor.select_speaker(found_id)
                        self.ui.lineEdit.setText(selected_name)
                        self.current_speaker_name = selected_name
                        log(f"Wybrano mówcę: {selected_name} (ID: {found_id})")
                    else:
                        self.ui.lineEdit.setText(speaker_id_text)
                        self.current_speaker_name = speaker_id_text
                else:
                    QMessageBox.warning(self, "Nie znaleziono mówcy",
                                        f"Mówca o nazwie '{speaker_id_text}' nie istnieje w bazie danych.\n"
                                        f"Dostępni mówcy: {', '.join([f'{k}: {v}' for k, v in all_speakers.items()])}")
                    self.ui.lineEdit.setText(self.current_speaker_name)

    def update_speaker_name(self, speaker_name):
        """Update speaker name in lineEdit"""
        if hasattr(self.ui, 'lineEdit'):
            self.ui.lineEdit.setText(speaker_name)
            self.current_speaker_name = speaker_name

    def on_on_clicked(self):
        log("WŁĄCZONO")
        if hasattr(self.ui, 'ConsoleOutput'):
            self.ui.ConsoleOutput.append("System włączony")
        if hasattr(self.ui, 'Status'):
            self.ui.Status.setText("Status: WŁĄCZONY")

        # Start audio processor thread
        self.start_audio_processor()

    def on_off_clicked(self):
        log("WYŁĄCZONO")
        if hasattr(self.ui, 'ConsoleOutput'):
            self.ui.ConsoleOutput.append("System wyłączony")
        if hasattr(self.ui, 'Status'):
            self.ui.Status.setText("Status: WYŁĄCZONY")

        # Stop audio processor thread
        self.stop_audio_processor()

    def on_input_changed(self, text):
        selected_id = self.ui.InputDv.currentData()
        log(f"Input zmieniony na: {text}")
        log(f"ID urządzenia: {selected_id}")

    def on_output_changed(self, text):
        selected_id = self.ui.OutputDv.currentData()
        log(f"Output zmieniony na: {text}")
        log(f"ID urządzenia: {selected_id}")

    def on_debug_mode_changed(self, state):
        """Handle debug mode checkbox change"""
        debug_enabled = (state == 2)  # Qt.CheckState.Checked == 2
        log(f"DebugMode zmieniony: {'włączony' if debug_enabled else 'wyłączony'}")

        # If processor is currently running, stop it and restart with new settings
        if self.processor_thread and self.processor_thread.isRunning():
            log("Procesor jest uruchomiony, restartuję z nowymi ustawieniami...")
            self.stop_audio_processor()
            # Use QTimer to give time for processor to stop
            QTimer.singleShot(500, self.start_audio_processor)

    def open_config_window(self):
        """Open configuration window"""
        if self.config_window is None or not self.config_window.isVisible():
            self.config_window = ConfigWindow(self.params, self)
            self.config_window.config_changed.connect(self.on_config_changed)
            self.config_window.config_closed.connect(self.on_config_closed)
            self.config_window.show()
            log("Okno konfiguracji otwarte")
        else:
            self.config_window.raise_()
            self.config_window.activateWindow()

    def on_config_changed(self, new_params):
        """Handle configuration changes"""
        self.params = new_params.copy()
        log("Parametry zaktualizowane:")
        for key, value in self.params.items():
            log(f"  {key}: {value}")

        # Restart processor if running
        if self.processor_thread and self.processor_thread.isRunning():
            log("Restartowanie procesora z nowymi parametrami...")
            self.stop_audio_processor()
            QTimer.singleShot(500, self.start_audio_processor)

    def on_config_closed(self):
        """Handle configuration window closing"""
        self.config_window = None
        log("Okno konfiguracji zamknięte")

    def restore_default_params(self):
        """Restore default parameters"""
        self.params = DEFAULT_PARAMS.copy()
        log("Przywrócono domyślne parametry:")
        for key, value in self.params.items():
            log(f"  {key}: {value}")

        # Restart processor if running
        if self.processor_thread and self.processor_thread.isRunning():
            log("Restartowanie procesora z domyślnymi parametrami...")
            self.stop_audio_processor()
            QTimer.singleShot(500, self.start_audio_processor)

    def start_audio_processor(self):
        """Start audio processor in a separate thread"""
        if self.is_restarting:
            return

        if self.processor_thread and self.processor_thread.isRunning():
            log("Processor już działa")
            return

        self.is_restarting = True

        try:
            # Get selected devices
            input_device = self.ui.InputDv.currentData() if hasattr(self.ui, 'InputDv') else None
            output_device = self.ui.OutputDv.currentData() if hasattr(self.ui, 'OutputDv') else None

            # Get debug passthrough state from checkbox
            debug_passthrough = False
            if hasattr(self.ui, 'checkBox'):
                debug_passthrough = self.ui.checkBox.isChecked()
                log(f"Debug passthrough ustawiony na: {debug_passthrough}")

            # Create processor with arguments from params
            processor = RealTimeSpeakerFilter(
                input_gain=self.params['input_gain'],
                output_gain=self.params['output_gain'],
                speaker_gain=self.params['speaker_gain'],
                debug_passthrough=debug_passthrough  # Use checkbox state
            )

            # Set callback to update speaker name in UI
            processor.set_update_speaker_name_callback(self.update_speaker_name)

            # Set the similarity threshold
            processor.speaker_tracker.recognizer.similarity_threshold = self.params['similarity_threshold']

            # Set fail-safe parameters
            processor.speaker_tracker.fail_safe_match_threshold = self.params['fail_safe_matches']
            processor.speaker_tracker.fail_safe_duration = self.params['fail_safe_duration']

            # Check if we have a speaker ID in lineEdit
            if hasattr(self.ui, 'lineEdit'):
                current_text = self.ui.lineEdit.text().strip()
                if current_text and current_text != "Wszyscy":
                    # Try to parse speaker ID from current text
                    # First check if it's a known speaker name
                    all_speakers = self.speaker_recognizer.get_all_speakers()
                    speaker_id = -1

                    # Check if text is a number
                    if current_text.isdigit():
                        speaker_id = int(current_text)
                        if str(speaker_id) not in all_speakers:
                            speaker_id = -1
                    else:
                        # Try to find by name
                        for spk_id, name in all_speakers.items():
                            if name == current_text:
                                speaker_id = int(spk_id)
                                break

                    if speaker_id != -1:
                        processor.select_speaker(speaker_id)
                        log(f"Ustawiono mówcę z lineEdit: ID {speaker_id}")

            # Create and start thread
            self.processor_thread = ProcessorThread(processor, input_device, output_device)
            self.processor_thread.processor_started.connect(self.on_processor_started)
            self.processor_thread.processor_stopped.connect(self.on_processor_stopped)
            self.processor_thread.processor_error.connect(self.on_processor_error)
            self.processor_thread.start()

        except Exception as e:
            log(f"Błąd przy uruchamianiu procesora: {e}")
            self.is_restarting = False

    def stop_audio_processor(self):
        """Stop audio processor thread"""
        if self.processor_thread and self.processor_thread.isRunning():
            try:
                # Stop the processor
                if hasattr(self.processor_thread, 'processor'):
                    self.processor_thread.processor.is_running = False

                # Wait for thread to stop
                self.processor_thread.quit()
                if not self.processor_thread.wait(2000):  # Wait max 2 seconds
                    log("Wątek procesora nie odpowiada, przerywam...")
                    self.processor_thread.terminate()
                    self.processor_thread.wait()

                log("Procesor audio zatrzymany")
                self.processor_thread = None
            except Exception as e:
                log(f"Błąd przy zatrzymywaniu procesora: {e}")
        self.is_restarting = False

    def on_processor_started(self):
        """Called when processor thread starts"""
        log("Procesor audio uruchomiony")
        self.is_restarting = False

    def on_processor_stopped(self):
        """Called when processor thread stops"""
        log("Procesor audio zakończony")
        self.processor_thread = None
        self.is_restarting = False

    def on_processor_error(self, error_msg):
        """Called when processor thread encounters an error"""
        log(f"Błąd procesora: {error_msg}")
        self.processor_thread = None
        self.is_restarting = False

    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_audio_processor()
        if self.config_window:
            self.config_window.close()
        if self.registration_window:
            self.registration_window.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time speaker filtering with UI - WINDOWS VERSION")
    parser.add_argument("--input-device", type=int, default=None, help="Input device ID")
    parser.add_argument("--output-device", type=int, default=None, help="Output device ID")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size")
    parser.add_argument("--input-gain", type=float, default=3.0,
                        help="Input gain (0.5-10.0)")
    parser.add_argument("--output-gain", type=float, default=1.0,
                        help="Output gain (0.5-5.0)")
    parser.add_argument("--speaker-gain", type=float, default=2.0,
                        help="Gain applied to recognized speaker (1.0-10.0)")
    parser.add_argument("--similarity-threshold", type=float, default=0.55,
                        help="Similarity threshold for speaker recognition (0.1-0.9)")
    parser.add_argument("--fail-safe-matches", type=int, default=3,
                        help="Number of consecutive matches to activate fail-safe")
    parser.add_argument("--fail-safe-duration", type=int, default=5,
                        help="Number of samples to pass in fail-safe mode")
    parser.add_argument("--debug-passthrough", action="store_true",
                        help="Debug mode: skip all processing")

    args = parser.parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

    log(f"Configuration:")
    log(f"  Platform: {platform.system()}")
    log(f"  Speaker recognition: YOUR MODEL INTEGRATED")
    log(f"  Selection: BY SPEAKER ID")
    log(f"  NO DENOISING - only speaker filtering")
    log(f"  Similarity threshold: {args.similarity_threshold}")
    log(f"  FAIL-SAFE: {args.fail_safe_matches} matches → {args.fail_safe_duration} samples")
    log(f"  Input device: {args.input_device or 'default'}")
    log(f"  Output device: {args.output_device or 'default'}")
    log(f"  Sample rate: {SAMPLE_RATE} Hz")
    log(f"  Chunk size: {args.chunk_size} samples")
    log(f"  Input gain: {args.input_gain}")
    log(f"  Output gain: {args.output_gain}")
    log(f"  Speaker gain: {args.speaker_gain}")
    log(f"  Debug passthrough: {args.debug_passthrough}")

    # Run the application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Execute application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()