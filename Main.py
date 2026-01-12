#!/usr/bin/env python3
"""
Real-time Audio Denoiser + Separator + Speaker Recognition System
Integrates your trained speaker recognition model for speaker selection
WINDOWS COMPATIBLE VERSION
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

# Try to import asteroid models
try:
    from asteroid.models import ConvTasNet
except ImportError:
    print("Warning: Could not import ConvTasNet from asteroid")


    # Create a dummy class for fallback
    class ConvTasNet(nn.Module):
        def __init__(self, n_src=3):
            super().__init__()
            self.n_src = n_src
            print(f"Warning: Using dummy ConvTasNet for {n_src} sources")

        def forward(self, x):
            # Return the same audio for each source
            batch_size = x.shape[0]
            return torch.stack([x] * self.n_src, dim=1)

# Try to import your training module
try:
    from train_denoiser import UNet1D
except ImportError:
    print("Warning: Could not import UNet1D from train_denoiser")


    # Define a simple fallback
    class UNet1D(nn.Module):
        def __init__(self, in_chan=1, base=32):
            super().__init__()
            print("Warning: Using dummy UNet1D")
            self.conv1 = nn.Conv1d(in_chan, base, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(base, in_chan, kernel_size=3, padding=1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return x

warnings.filterwarnings('ignore')

# Global variables
SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# YOUR SPEAKER RECOGNITION MODEL (from test_speaker_recognition.py)
# ============================================

class SpeakerEncoder(nn.Module):
    """Your working speaker recognition model"""

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
    """Extract MFCC features for speaker recognition (identical to test_speaker_recognition.py)"""
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

    # Normalization
    features_tensor = (features_tensor - features_tensor.mean(dim=2, keepdim=True)) / \
                      (features_tensor.std(dim=2, keepdim=True) + 1e-8)

    return features_tensor


def load_audio_for_recognition(filepath, target_sr=16000, duration=3.0):
    """Load audio for recognition"""
    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

        # Normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Trim/padding
        target_len = int(duration * target_sr)
        if len(audio) > target_len:
            # Random segment
            start = np.random.randint(0, len(audio) - target_len)
            audio = audio[start:start + target_len]
        else:
            padding = np.zeros(target_len - len(audio))
            audio = np.concatenate([audio, padding])

        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# ============================================
# SPEAKER RECOGNIZER WITH YOUR MODEL
# ============================================

class SpeakerRecognizer:
    """Speaker recognition using YOUR trained model"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

        # Load YOUR model
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
                print("✅ Speaker recognition model loaded")
            except Exception as e:
                print(f"❌ Error loading speaker model: {e}")
                print("⚠️  Using untrained model")
        else:
            print("⚠️  Warning: No trained speaker model found")

        self.model.eval()

        # Speaker database
        self.speaker_database_path = "./speaker_database.pkl"
        self.speaker_embeddings = {}  # speaker_id -> embedding tensor
        self.speaker_names = {}  # speaker_id -> name

        # Similarity threshold
        self.similarity_threshold = 0.6

        # Load existing database
        self.load_database()

    def load_database(self):
        """Load speaker database from file"""
        if os.path.exists(self.speaker_database_path):
            try:
                with open(self.speaker_database_path, 'rb') as f:
                    db = pickle.load(f)

                if 'speakers' in db and 'speaker_names' in db:
                    self.speaker_embeddings = db['speakers']
                    self.speaker_names = db['speaker_names']
                    print(f"📊 Speaker database: {len(self.speaker_embeddings)} speakers loaded")
                else:
                    print("⚠️  Invalid database format")
            except Exception as e:
                print(f"❌ Error loading database: {e}")
        else:
            print("📊 Creating new speaker database")

    def extract_embedding(self, audio):
        """Extract speaker embedding from audio using YOUR model"""
        if len(audio) < self.sample_rate:  # Need at least 1 second
            return None

        with torch.no_grad():
            # Extract features
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
        """Recognize speaker in audio using YOUR model and database"""
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
                    if 'debug' in locals() and debug:
                        print(f"  Error calculating similarity for {speaker_id}: {e}")

        # Check threshold
        if best_similarity >= self.similarity_threshold:
            name = self.speaker_names.get(best_speaker_id, f"Speaker_{best_speaker_id}")
            return best_speaker_id, best_similarity, name

        return None, best_similarity, "Unknown"

    def get_top_similarities(self, embedding, top_k=5):
        """Get top K most similar speakers"""
        similarities = []

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

                try:
                    similarity = F.cosine_similarity(embedding_2d, ref_embedding_2d, dim=1).item()
                    name = self.speaker_names.get(speaker_id, f"Speaker_{speaker_id}")
                    similarities.append((speaker_id, name, similarity))
                except:
                    continue

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def list_speakers(self):
        """List all registered speakers"""
        speakers = []
        for speaker_id, name in self.speaker_names.items():
            speakers.append(f"ID: {speaker_id} -> '{name}'")
        return speakers


# ============================================
# AI ENHANCED SPEAKER TRACKER
# ============================================

class AIEnhancedSpeakerTracker:
    """Speaker tracker with YOUR recognition model"""

    def __init__(self, max_speakers=3):
        self.max_speakers = max_speakers

        # Use YOUR speaker recognizer
        self.recognizer = SpeakerRecognizer()

        # Tracking assignments {source_idx: speaker_id}
        self.source_to_speaker = {}
        self.speaker_to_source = {}
        self.speaker_activity = defaultdict(float)

        # Decision history for stability
        self.assignment_history = defaultdict(lambda: deque(maxlen=5))

        # Speaker selection mode
        self.selected_speaker_id = -1  # -1 = auto (similarity-based), specific ID = only that speaker
        self.block_other_speakers = False

        # Debug mode - domyślnie wyłączone
        self.debug = False

    def set_selected_speaker(self, speaker_id):
        """Set which speaker to allow through"""
        self.selected_speaker_id = speaker_id
        if speaker_id == -1:
            self.block_other_speakers = False
        else:
            self.block_other_speakers = True

    def process_batch(self, separated_sources):
        """Process separated sources, recognize speakers"""
        current_assignments = {}
        speaker_assignments = {}

        # Update activity timeout
        current_time = time.time()
        for speaker_id in list(self.speaker_activity.keys()):
            if current_time - self.speaker_activity[speaker_id] > 10.0:  # 10s timeout
                if speaker_id in self.speaker_to_source:
                    source_idx = self.speaker_to_source[speaker_id]
                    del self.source_to_speaker[source_idx]
                    del self.speaker_to_source[speaker_id]

        # For each separated source
        for source_idx in range(separated_sources.shape[0]):
            audio = separated_sources[source_idx].cpu().numpy()

            # Need enough audio for recognition (at least 1 second)
            if len(audio) >= 16000:
                # Recognize speaker using YOUR model
                speaker_id, similarity, speaker_name = self.recognizer.recognize_speaker(audio)

                if self.debug:
                    energy = 10 * np.log10(np.mean(audio ** 2) + 1e-10)
                    print(f"[TRACKER] Source {source_idx}: energy={energy:.1f}dB, "
                          f"speaker={speaker_name}, similarity={similarity:.3f}")

                if speaker_id is not None:
                    current_assignments[source_idx] = speaker_id
                    speaker_assignments[source_idx] = {
                        'speaker_id': speaker_id,
                        'name': speaker_name,
                        'similarity': similarity,
                        'energy': 10 * np.log10(np.mean(audio ** 2) + 1e-10)
                    }

                    # Update activity
                    self.speaker_activity[speaker_id] = current_time

                    # Update assignment tracking
                    if speaker_id not in self.speaker_to_source:
                        # New speaker, assign to this source
                        self._assign_speaker_to_source(source_idx, speaker_id)
                        print(
                            f"[TRACKER] 🎯 Speaker '{speaker_name}' (ID: {speaker_id}) assigned to source {source_idx}")
                    elif self.speaker_to_source[speaker_id] != source_idx:
                        print(
                            f"[TRACKER] 🔄 Speaker '{speaker_name}' (ID: {speaker_id}) switched from source {self.speaker_to_source[speaker_id]} to {source_idx}")
                        self._assign_speaker_to_source(source_idx, speaker_id)

        # Determine which sources to allow based on selected speaker
        allowed_sources = []

        if self.block_other_speakers and self.selected_speaker_id != -1:
            # Mode: only selected speaker
            for source_idx, speaker_id in current_assignments.items():
                if str(speaker_id) == str(self.selected_speaker_id):
                    allowed_sources.append(source_idx)
            # DODANE: Komunikat o znalezionych źródłach
            if allowed_sources and self.debug:
                print(f"[TRACKER] ✅ Selected speaker found in source(s): {allowed_sources}")
        else:
            # Mode: all speakers or auto (similarity-based)
            allowed_sources = list(current_assignments.keys())

        return speaker_assignments, allowed_sources

    def _assign_speaker_to_source(self, source_idx, speaker_id):
        """Assign speaker to source"""
        # Remove old assignments
        if speaker_id in self.speaker_to_source:
            old_source = self.speaker_to_source[speaker_id]
            del self.source_to_speaker[old_source]

        if source_idx in self.source_to_speaker:
            old_speaker = self.source_to_speaker[source_idx]
            del self.speaker_to_source[old_speaker]

        # Set new assignment
        self.source_to_speaker[source_idx] = speaker_id
        self.speaker_to_source[speaker_id] = source_idx

    def get_active_speakers(self):
        """Get list of active speakers"""
        active = []
        for speaker_id, last_active in self.speaker_activity.items():
            if time.time() - last_active < 5.0:  # Active within 5s
                name = self.recognizer.speaker_names.get(str(speaker_id), f"Speaker_{speaker_id}")
                active.append({
                    'id': speaker_id,
                    'name': name,
                    'last_active': last_active
                })
        return active

    def list_registered_speakers(self):
        """List all registered speakers"""
        return self.recognizer.list_speakers()


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
                elif char in ['G', 'A', 'N', 'I', 'D', 'L', 'S']:
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
# MAIN REAL-TIME PROCESSOR WITH SPEAKER RECOGNITION
# ============================================

class RealTimeDenoiserSeparator:
    def __init__(self, denoise_model_path=None, separation_model_path=None,
                 separation_model_path2=None, vad_model_path=None,
                 denoise_strength=0.5, second_denoise_strength=0.3, input_gain=1.0, output_gain=1.0,
                 separation_gain=50.0, num_sources=3, debug_passthrough=False):
        """
        Real-time denoiser + separator with YOUR speaker recognition
        """
        self.denoise_strength = max(0.1, min(0.9, denoise_strength))
        self.second_denoise_strength = max(0.1, min(0.9, second_denoise_strength))
        self.input_gain = input_gain
        self.output_gain = output_gain
        self.separation_gain = separation_gain
        self.debug_passthrough = debug_passthrough
        self.num_sources = num_sources

        # Speaker selection
        self.selected_speaker_id = -1  # -1 = auto/similarity-based, specific ID = only that speaker

        # Model paths
        self.separation_model_path = separation_model_path
        self.separation_model_path2 = separation_model_path2
        self.vad_model_path = vad_model_path
        self.current_model_type = "3-source"

        # AI speaker tracker with YOUR model
        self.speaker_tracker = AIEnhancedSpeakerTracker(max_speakers=3)

        # Debug flag - domyślnie wyłączone
        self.debug_speech_detection = False

        # Dodane: śledzenie ostatnio użytych źródeł
        self.last_allowed_sources = []

        # Licznik dla ograniczenia wyświetlania
        self.process_counter = 0
        self.display_interval = 5  # Wyświetlaj informacje co 5 procesowań

        print(f"Debug passthrough mode: {debug_passthrough}")
        print(f"Using device: {DEVICE}")
        print(f"Number of sources: {num_sources}")
        print(f"First denoise strength: {denoise_strength}")
        print(f"Second denoise strength: {second_denoise_strength}")
        print(f"Separation gain: {separation_gain}")
        print(f"Main separation model: {separation_model_path}")
        print(f"Alternative separation model (2-source): {separation_model_path2}")
        print(f"Speaker recognition: YOUR MODEL INTEGRATED")
        print(f"Selection mode: BY SPEAKER ID")
        print(f"Selection method: SIMILARITY-BASED (not loudest)")

        if debug_passthrough:
            print("DEBUG: Running in passthrough mode - no processing")
            self.denoise_model = None
            self.sep_model = None
            self.sep_model2 = None
            # Dla trybu passthrough ustaw prostsze wartości
            self.window_size = CHUNK_SIZE * 16  # Większy buffer dla płynności
            self.hop_size = CHUNK_SIZE * 4  # Przesunięcie 4 chunków
            print(f"Window size: {self.window_size} samples ({self.window_size / SAMPLE_RATE * 1000:.0f}ms)")
            print(f"Hop size: {self.hop_size} samples ({self.hop_size / SAMPLE_RATE * 1000:.0f}ms)")
        else:
            # Window settings dla separacji 1.5 sekundy
            self.window_size = int(1.5 * SAMPLE_RATE)  # 1.5 sekundy przy 48kHz = 72000 samples
            self.hop_size = int(0.25 * SAMPLE_RATE)  # 0.25 sekundy = 12000 samples

            # Window settings dla odszumiania (krótsze)
            self.denoise_window_size = int(1.0 * SAMPLE_RATE)  # 1 sekunda dla odszumiania

            print(f"Processing window: {self.window_size} samples ({self.window_size / SAMPLE_RATE * 1000:.0f}ms)")
            print(
                f"Denoise window: {self.denoise_window_size} samples ({self.denoise_window_size / SAMPLE_RATE * 1000:.0f}ms)")
            print(f"Hop size: {self.hop_size} samples ({self.hop_size / SAMPLE_RATE * 1000:.0f}ms)")

            # Load denoising model
            if denoise_model_path and os.path.exists(denoise_model_path):
                print(f"Loading denoising model from: {denoise_model_path}")
                self.load_denoising_model(denoise_model_path)
            else:
                print("WARNING: No denoising model, skipping denoising steps")
                self.denoise_model = None

            # Load main separation model
            if separation_model_path and os.path.exists(separation_model_path):
                print(f"Loading main separation model (3-source) from: {separation_model_path}")
                self.load_separation_model(separation_model_path, 3)
                self.current_sep_model = self.sep_model
            else:
                print("WARNING: No main separation model, skipping separation step")
                self.sep_model = None
                self.current_sep_model = None

            # Load alternative separation model
            if separation_model_path2 and os.path.exists(separation_model_path2):
                print(f"Loading alternative separation model (2-source) from: {separation_model_path2}")
                self.load_separation_model2(separation_model_path2, 2)
            else:
                print("WARNING: No alternative separation model")
                self.sep_model2 = None

        # Input buffer
        self.input_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.buffer_ptr = 0

        # Output buffer dla płynności
        self.output_buffer = np.zeros(self.hop_size * 2, dtype=np.float32)
        self.output_buffer_pos = 0

        # Queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=100)  # Mniejsza kolejka

        # Flags
        self.is_running = True
        self.samples_processed = 0
        self.last_audio_time = time.time()
        self.last_speech_time = 0

        # Filters
        self.setup_simple_filters()

        print(f"Stream: {SAMPLE_RATE}Hz, Model: {MODEL_SAMPLE_RATE}Hz")
        print(f"Platform: {platform.system()} ({'Windows' if IS_WINDOWS else 'Unix/Linux'})")

    def load_denoising_model(self, model_path):
        """Load denoising model"""
        try:
            self.denoise_model = UNet1D(in_chan=1, base=32)
            checkpoint = torch.load(model_path, map_location=DEVICE)

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.denoise_model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.denoise_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.denoise_model.load_state_dict(checkpoint)
            else:
                self.denoise_model.load_state_dict(checkpoint)

            self.denoise_model.to(DEVICE)
            self.denoise_model.eval()
            print("Denoising model loaded successfully")
        except Exception as e:
            print(f"Error loading denoising model: {e}")
            self.denoise_model = None

    def load_separation_model(self, model_path, num_sources):
        """Load main separation model"""
        try:
            self.sep_model = ConvTasNet(n_src=num_sources)
            checkpoint = torch.load(model_path, map_location=DEVICE)

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.sep_model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.sep_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.sep_model.load_state_dict(checkpoint)
            else:
                self.sep_model.load_state_dict(checkpoint)

            self.sep_model.to(DEVICE)
            self.sep_model.eval()
            print(f"Main separation model loaded successfully for {num_sources} sources")
        except Exception as e:
            print(f"Error loading main separation model: {e}")
            self.sep_model = None

    def load_separation_model2(self, model_path, num_sources):
        """Load alternative separation model"""
        try:
            self.sep_model2 = ConvTasNet(n_src=num_sources)
            checkpoint = torch.load(model_path, map_location=DEVICE)

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.sep_model2.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.sep_model2.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.sep_model2.load_state_dict(checkpoint)
            else:
                self.sep_model2.load_state_dict(checkpoint)

            self.sep_model2.to(DEVICE)
            self.sep_model2.eval()
            print(f"Alternative separation model loaded successfully for {num_sources} sources")
        except Exception as e:
            print(f"Error loading alternative separation model: {e}")
            self.sep_model2 = None

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

    def denoise_audio(self, audio_chunk_16k, denoise_strength=0.5):
        """Denoise audio"""
        if self.denoise_model is None or self.debug_passthrough or len(audio_chunk_16k) == 0:
            return audio_chunk_16k

        try:
            with torch.no_grad():
                # Model expects specific input size
                model_input_size = 16384  # 1 sekunda przy 16kHz

                if len(audio_chunk_16k) != model_input_size:
                    # Resize or pad as needed
                    if len(audio_chunk_16k) < model_input_size:
                        padding = model_input_size - len(audio_chunk_16k)
                        audio_padded = np.pad(audio_chunk_16k, (0, padding), mode='reflect')
                    else:
                        audio_padded = audio_chunk_16k[:model_input_size]
                else:
                    audio_padded = audio_chunk_16k

                audio_tensor = torch.from_numpy(audio_padded).float()
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

                denoised = self.denoise_model(audio_tensor)
                denoised = denoised.squeeze().cpu().numpy()

                if len(audio_chunk_16k) != model_input_size:
                    if len(audio_chunk_16k) < model_input_size:
                        denoised = denoised[:len(audio_chunk_16k)]
                    else:
                        denoised = np.pad(denoised, (0, len(audio_chunk_16k) - len(denoised)), mode='constant')

                if denoise_strength < 1.0:
                    denoised = (
                            denoised * denoise_strength +
                            audio_chunk_16k * (1 - denoise_strength)
                    )

                return denoised
        except Exception as e:
            print(f"Error in denoising: {e}")
            return audio_chunk_16k

    def separate_sources(self, audio_chunk_16k):
        """Separate audio sources - używa dłuższego okna dla lepszej separacji"""
        if self.current_sep_model is None or self.debug_passthrough or self.num_sources <= 1 or len(
                audio_chunk_16k) == 0:
            # Jeśli brak modelu, zwróć oryginalny dźwięk jako pojedyncze źródło
            return torch.tensor([audio_chunk_16k])

        try:
            with torch.no_grad():
                # Model separacji wymaga konkretnej długości wejścia
                # Zazwyczaj modele ConvTasNet wymagają 16384 próbek (1 sekunda) lub podobnie
                model_input_size = 16384  # 1 sekunda przy 16kHz

                if len(audio_chunk_16k) != model_input_size:
                    # Dopasuj długość do modelu
                    if len(audio_chunk_16k) < model_input_size:
                        pad_len = model_input_size - len(audio_chunk_16k)
                        audio_padded = np.pad(audio_chunk_16k, (0, pad_len), mode='reflect')
                    else:
                        # Weź środek dłuższego segmentu
                        start = (len(audio_chunk_16k) - model_input_size) // 2
                        audio_padded = audio_chunk_16k[start:start + model_input_size]
                else:
                    audio_padded = audio_chunk_16k

                audio_tensor = torch.from_numpy(audio_padded).float()
                audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)

                separated = self.current_sep_model(audio_tensor)
                separated = separated.squeeze(0)

                # Jeśli źródła są zbyt ciche, wzmocnij je
                for i in range(separated.shape[0]):
                    src = separated[i].cpu().numpy()
                    rms = np.sqrt(np.mean(src ** 2))
                    if rms > 0:
                        separated[i] = separated[i] * min(5.0, 0.1 / rms)

                return separated
        except Exception as e:
            print(f"Error in separation: {e}")
            return torch.tensor([audio_chunk_16k])

    def switch_separation_model(self):
        """Switch between separation models"""
        if self.sep_model is None or self.sep_model2 is None:
            print("[INFO] Cannot switch models: missing one model")
            return

        if self.current_model_type == "3-source":
            self.current_sep_model = self.sep_model2
            self.num_sources = 2
            self.current_model_type = "2-source"
            print(f"\n[INFO] Switched to 2-source separation model")
        else:
            self.current_sep_model = self.sep_model
            self.num_sources = 3
            self.current_model_type = "3-source"
            print(f"\n[INFO] Switched to 3-source separation model")

        print(f"[INFO] Number of sources: {self.num_sources}")

    def select_speaker(self, speaker_id):
        """Select which speaker to allow through"""
        if speaker_id == -1:
            self.selected_speaker_id = -1
            self.speaker_tracker.set_selected_speaker(-1)
            print(f"\n🎤 Mode: ALL SPEAKERS (similarity-based)")
        else:
            self.selected_speaker_id = speaker_id
            self.speaker_tracker.set_selected_speaker(speaker_id)

            # Find speaker name
            speaker_name = f"Speaker_{speaker_id}"
            for spk_id, name in self.speaker_tracker.recognizer.speaker_names.items():
                if str(spk_id) == str(speaker_id):
                    speaker_name = name
                    break

            print(f"\n🎤 Mode: ONLY SPEAKER '{speaker_name}' (ID: {speaker_id})")

    def select_next_speaker(self):
        """Select next active speaker"""
        active_speakers = self.speaker_tracker.get_active_speakers()
        if not active_speakers:
            print("[INFO] No active speakers")
            return

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

        self.select_speaker(next_speaker['id'])

    def show_speaker_info(self):
        """Show information about speakers"""
        # Registered speakers
        registered_info = self.speaker_tracker.list_registered_speakers()
        if registered_info:
            print("\n[SPEAKER INFO] Registered speakers:")
            for line in registered_info:
                print(f"  {line}")

        # Active speakers
        active_speakers = self.speaker_tracker.get_active_speakers()
        if active_speakers:
            print("\n[SPEAKER INFO] Currently active speakers:")
            for spk in active_speakers:
                status = "✓ SELECTED" if str(spk['id']) == str(self.selected_speaker_id) else ""
                print(f"  ID: {spk['id']}, Name: '{spk['name']}', "
                      f"Active: {time.time() - spk['last_active']:.1f}s ago {status}")
        else:
            print("\n[SPEAKER INFO] No active speakers")

    def toggle_speech_debug(self):
        """Toggle debug mode"""
        self.debug_speech_detection = not self.debug_speech_detection
        self.speaker_tracker.debug = self.debug_speech_detection
        print(f"[DEBUG] Debug mode: {'ENABLED' if self.debug_speech_detection else 'DISABLED'}")

    def process_denoise_separate_denoise(self, audio_chunk_48k):
        """Processing with speaker recognition and blocking"""
        if self.debug_passthrough:
            return audio_chunk_48k

        # Zwiększ licznik i sprawdź czy wyświetlać informacje
        self.process_counter += 1
        should_display = (self.process_counter % self.display_interval == 0)

        # 1. Resample to 16kHz
        try:
            audio_16k = resampy.resample(
                audio_chunk_48k,
                SAMPLE_RATE,
                MODEL_SAMPLE_RATE,
                filter='kaiser_fast'
            )
        except Exception as e:
            print(f"Error in resampling to 16k: {e}")
            return audio_chunk_48k

        # 2. FIRST DENOISING
        if should_display:
            print(f"[PROCESS] Step 1: Denoising (strength={self.denoise_strength})")

        denoised_16k = self.denoise_audio(audio_16k, self.denoise_strength)

        # 3. SEPARATION
        if should_display:
            print(f"[PROCESS] Step 2: Separation ({self.current_model_type})")

        separated_sources = self.separate_sources(denoised_16k)

        # Sprawdź czy separacja działa
        if separated_sources.shape[0] > 1 and should_display:
            print(f"[PROCESS] Separated into {separated_sources.shape[0]} sources")

        # 4. SPEAKER RECOGNITION FOR EACH SOURCE
        if should_display:
            print(f"[PROCESS] Step 3: Speaker recognition")

        speaker_assignments, allowed_sources = self.speaker_tracker.process_batch(separated_sources)

        # 5. SELECT SOURCES BASED ON SELECTED SPEAKER
        selected_audio = None

        if self.selected_speaker_id == -1 or not self.speaker_tracker.block_other_speakers:
            # Mode: ALL speakers or AUTO (similarity-based)
            if separated_sources.shape[0] > 0:
                # Find source with highest similarity to any recognized speaker
                best_idx = -1
                best_similarity = -1.0
                best_speaker_name = "Unknown"

                for i in range(separated_sources.shape[0]):
                    if i in speaker_assignments:
                        assignment = speaker_assignments[i]
                        similarity = assignment['similarity']
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_idx = i
                            best_speaker_name = assignment['name']

                if best_idx != -1:
                    selected_audio = separated_sources[best_idx].cpu().numpy()
                    if should_display:
                        print(
                            f"[SELECT] 🔓 Using source {best_idx}: {best_speaker_name} (similarity={best_similarity:.3f})")
                else:
                    # No recognized speakers, use first source
                    selected_audio = separated_sources[0].cpu().numpy() if separated_sources.shape[
                                                                               0] > 0 else denoised_16k
                    if should_display:
                        print(f"[SELECT] 🔊 Using source 0 (no speaker recognized)")
            else:
                selected_audio = denoised_16k
        else:
            # Mode: ONLY selected speaker
            if allowed_sources:
                # Znajdź źródło z najwyższym podobieństwem do wybranego mówcy
                best_source_idx = -1
                best_similarity = -1.0
                best_speaker_name = "Unknown"

                for source_idx in allowed_sources:
                    if source_idx in speaker_assignments:
                        assignment = speaker_assignments[source_idx]
                        if assignment['speaker_id'] == self.selected_speaker_id:
                            similarity = assignment['similarity']
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_source_idx = source_idx
                                best_speaker_name = assignment['name']

                if best_source_idx != -1:
                    selected_audio = separated_sources[best_source_idx].cpu().numpy()
                    if should_display:
                        print(
                            f"[SELECT] 🔒 Using source {best_source_idx}: {best_speaker_name} (similarity={best_similarity:.3f})")
                else:
                    # Jeśli nie znaleziono, weź pierwsze
                    selected_audio = separated_sources[allowed_sources[0]].cpu().numpy()
                    if should_display:
                        print(f"[SELECT] 🔒 Using source {allowed_sources[0]} (fallback)")
            else:
                # No match - silence
                selected_audio = np.zeros_like(denoised_16k)
                if should_display:
                    print(f"[SELECT] 🔇 Outputting silence (no matching speaker)")

        # 6. POST-SEPARATION GAIN
        source_peak = np.max(np.abs(selected_audio))
        if source_peak > 0:
            auto_gain = min(0.5 / source_peak, self.separation_gain)
            effective_gain = min(self.separation_gain, auto_gain)
            selected_audio = selected_audio * effective_gain

        selected_audio = np.clip(selected_audio, -1.0, 1.0)

        # 7. SECOND DENOISING
        if should_display:
            print(f"[PROCESS] Step 4: Second denoising (strength={self.second_denoise_strength})")

        second_denoised_source = self.denoise_audio(selected_audio, self.second_denoise_strength)

        # 8. Match length
        if len(second_denoised_source) > len(audio_16k):
            second_denoised_source = second_denoised_source[:len(audio_16k)]
        elif len(second_denoised_source) < len(audio_16k):
            second_denoised_source = np.pad(second_denoised_source,
                                            (0, len(audio_16k) - len(second_denoised_source)),
                                            mode='constant')

        # 9. Resample back to 48kHz
        try:
            output_48k = resampy.resample(
                second_denoised_source,
                MODEL_SAMPLE_RATE,
                SAMPLE_RATE,
                filter='kaiser_fast'
            )
        except Exception as e:
            print(f"Error in resampling to 48k: {e}")
            return audio_chunk_48k

        # 10. Apply identity filter (high-pass)
        if should_display:
            print(f"[PROCESS] Step 5: Identity filter")

        output_48k = self.apply_filter(output_48k)

        # 11. Match size
        if len(output_48k) > len(audio_chunk_48k):
            output_48k = output_48k[:len(audio_chunk_48k)]
        elif len(output_48k) < len(audio_chunk_48k):
            output_48k = np.pad(output_48k,
                                (0, len(audio_chunk_48k) - len(output_48k)),
                                mode='constant')

        if should_display:
            print(f"[PROCESS] Completed pipeline: Denoise → Separate → Recognize → Denoise → Filter")

        return output_48k

    def input_callback(self, indata, frames, time_info, status):
        """Input callback"""
        if status:
            print(f"Input status: {status}")

        chunk = indata.copy().flatten()
        chunk = chunk * self.input_gain

        audio_level = np.max(np.abs(chunk))
        if audio_level > 0.01:
            self.last_audio_time = time.time()

        self.input_queue.put(chunk)

    def output_callback(self, outdata, frames, time_info, status):
        """Output callback"""
        if status:
            print(f"Output status: {status}")

        try:
            output_chunk = self.output_queue.get_nowait()

            if len(output_chunk) < frames:
                output_chunk = np.pad(output_chunk, (0, frames - len(output_chunk)), mode='constant')
            elif len(output_chunk) > frames:
                output_chunk = output_chunk[:frames]

            audio_level = np.max(np.abs(output_chunk))
            if audio_level > 0.01:
                output_chunk = output_chunk * self.output_gain
                max_val = np.max(np.abs(output_chunk))
                if max_val > 1.0:
                    output_chunk = output_chunk / max_val * 0.95

            outdata[:, 0] = output_chunk

        except queue.Empty:
            outdata.fill(0)

    def keyboard_listener(self, input_handler):
        """Keyboard listener with speaker selection"""
        print("\n[KEYBOARD] Speaker Recognition Commands:")
        print("[KEYBOARD] 'Tab' - switch separation models")
        print("[KEYBOARD] 'n' - select next active speaker")
        print("[KEYBOARD] 'a' - ALL speakers mode (similarity-based)")
        print("[KEYBOARD] 'i' - show speaker information")
        print("[KEYBOARD] 'd' - toggle debug mode")
        print("[KEYBOARD] 'l' - list registered speakers")
        print("[KEYBOARD] '0'-'9' - select speaker by ID")
        print("[KEYBOARD] 's' - show current source assignments")
        print("[KEYBOARD] 'g' - increase input gain (+0.5)")
        print("[KEYBOARD] 'G' (Shift+g) - decrease input gain (-0.5)")
        print("[KEYBOARD] 'q' - quit program")
        print("[KEYBOARD] Note: On Windows, press keys without Enter")

        last_key_time = 0
        key_debounce = 0.3

        while self.is_running:
            try:
                key = input_handler.get_key()

                if key:
                    current_time = time.time()

                    if current_time - last_key_time > key_debounce:
                        if key == '\t':
                            self.switch_separation_model()
                            last_key_time = current_time
                        elif key == 'n':
                            self.select_next_speaker()
                            last_key_time = current_time
                        elif key == 'a':
                            self.select_speaker(-1)  # All speakers mode
                            last_key_time = current_time
                        elif key == 'i':
                            self.show_speaker_info()
                            last_key_time = current_time
                        elif key == 'd':
                            self.toggle_speech_debug()
                            last_key_time = current_time
                        elif key == 'l':
                            print("\n[SPEAKERS] Registered speakers:")
                            speakers = self.speaker_tracker.list_registered_speakers()
                            for line in speakers:
                                print(f"  {line}")
                            last_key_time = current_time
                        elif key == 's':
                            print("\n[CURRENT] Source assignments:")
                            for src_idx, spk_id in self.speaker_tracker.source_to_speaker.items():
                                name = self.speaker_tracker.recognizer.speaker_names.get(str(spk_id),
                                                                                         f"Speaker_{spk_id}")
                                status = "✓ ACTIVE" if src_idx in (
                                    self.last_allowed_sources if hasattr(self, 'last_allowed_sources') else []) else ""
                                print(f"  Source {src_idx} -> Speaker '{name}' (ID: {spk_id}) {status}")
                            last_key_time = current_time
                        elif key.isdigit():
                            speaker_id = int(key)
                            self.select_speaker(speaker_id)
                            last_key_time = current_time
                        elif key == 'g':
                            self.input_gain = min(10.0, self.input_gain + 0.5)
                            print(f"[GAIN] Increased input gain to: {self.input_gain:.1f}")
                            last_key_time = current_time
                        elif key == 'G':
                            self.input_gain = max(0.5, self.input_gain - 0.5)
                            print(f"[GAIN] Decreased input gain to: {self.input_gain:.1f}")
                            last_key_time = current_time
                        elif key == 'q':
                            print("\n[KEYBOARD] Detected 'q' - quitting...")
                            self.is_running = False
                            break

                time.sleep(0.01)

            except Exception as e:
                print(f"[KEYBOARD] Listener error: {e}")
                time.sleep(0.1)

    def processing_loop(self):
        """Main processing loop - poprawione dla płynności"""
        print("Processing loop started...")

        if self.debug_passthrough:
            print("DEBUG PASSTHROUGH MODE: Direct audio passthrough")
            while self.is_running:
                try:
                    raw_chunk = self.input_queue.get(timeout=0.1)
                    self.samples_processed += len(raw_chunk)

                    # Po prostu przekaż dźwięk
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
                    print(f"Processing loop error: {e}")
        else:
            # Improved processing dla trybu z modelami
            processing_buffer = np.zeros(self.window_size, dtype=np.float32)
            buffer_fill = 0

            # Overlap-add window dla płynności
            window = np.hanning(self.window_size)

            while self.is_running:
                try:
                    # Pobierz dane wejściowe
                    try:
                        raw_chunk = self.input_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Jeśli nie ma danych, ale mamy coś w buforze, przetwórz
                        if buffer_fill > 0:
                            # Przetwórz to co mamy
                            window_to_process = processing_buffer[:buffer_fill].copy()
                            if len(window_to_process) >= int(0.5 * SAMPLE_RATE):  # Minimum 0.5s
                                # Zastosuj okno tylko do części z danymi
                                window_part = window[:buffer_fill]
                                windowed = window_to_process * window_part

                                processed = self.process_denoise_separate_denoise(windowed)
                                # Podziel na chunki i wyślij
                                for i in range(0, len(processed), CHUNK_SIZE):
                                    chunk = processed[i:i + CHUNK_SIZE].copy()
                                    if len(chunk) > 0:
                                        try:
                                            self.output_queue.put_nowait(chunk)
                                        except queue.Full:
                                            pass
                                buffer_fill = 0
                        continue

                    self.samples_processed += len(raw_chunk)

                    # Dodaj do bufora
                    if buffer_fill + len(raw_chunk) <= self.window_size:
                        processing_buffer[buffer_fill:buffer_fill + len(raw_chunk)] = raw_chunk
                        buffer_fill += len(raw_chunk)
                    else:
                        # Przetwórz pełne okno
                        window_to_process = processing_buffer.copy()
                        windowed = window_to_process * window

                        processed = self.process_denoise_separate_denoise(windowed)

                        # Wyślij przetworzone dane
                        # Wysyłamy tylko pierwszą część (hop_size) aby zachować timing
                        samples_to_send = min(self.hop_size, len(processed))
                        for i in range(0, samples_to_send, CHUNK_SIZE):
                            chunk = processed[i:i + CHUNK_SIZE].copy()
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass

                        # Przesuń bufor o hop_size
                        shift = self.hop_size
                        processing_buffer = np.roll(processing_buffer, -shift)
                        processing_buffer[-shift:] = 0
                        buffer_fill = max(0, buffer_fill - shift)

                        # Dodaj nowe dane
                        if buffer_fill + len(raw_chunk) <= self.window_size:
                            processing_buffer[buffer_fill:buffer_fill + len(raw_chunk)] = raw_chunk
                            buffer_fill += len(raw_chunk)
                        else:
                            # Jeśli nadal nie ma miejsca, nadpisz końcówkę
                            processing_buffer[-len(raw_chunk):] = raw_chunk
                            buffer_fill = self.window_size

                    # Przetwórz jeśli bufor jest pełny
                    if buffer_fill >= self.window_size:
                        window_to_process = processing_buffer.copy()
                        windowed = window_to_process * window

                        processed = self.process_denoise_separate_denoise(windowed)

                        # Wyślij przetworzone dane
                        samples_to_send = min(self.hop_size, len(processed))
                        for i in range(0, samples_to_send, CHUNK_SIZE):
                            chunk = processed[i:i + CHUNK_SIZE].copy()
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass

                        # Przesuń bufor
                        shift = self.hop_size
                        processing_buffer = np.roll(processing_buffer, -shift)
                        processing_buffer[-shift:] = 0
                        buffer_fill = max(0, buffer_fill - shift)

                except Exception as e:
                    print(f"Processing loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)

    def run(self, input_device=None, output_device=None):
        """Run the processor"""
        print("\n" + "=" * 60)
        print("Real-time Audio Processor with Speaker Recognition")
        print("=" * 60)
        print(f"Speaker recognition: YOUR MODEL INTEGRATED")
        print(f"Selection mode: BY SPEAKER ID")
        print(f"Selection method: SIMILARITY-BASED (not loudest)")
        print(
            f"Current speaker: {'ALL (similarity-based)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
        print(f"Model type: {self.current_model_type}")
        print(f"Number of sources: {self.num_sources}")
        print(f"First denoise strength: {self.denoise_strength}")
        print(f"Second denoise strength: {self.second_denoise_strength}")
        print(f"Separation gain: {self.separation_gain}x (auto-adjusted)")
        print(f"Input gain: {self.input_gain} (adjust with 'g'/'G')")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Processing window: {self.window_size} samples ({self.window_size / SAMPLE_RATE * 1000:.0f}ms)")
        print(
            f"Denoise window: {self.denoise_window_size} samples ({self.denoise_window_size / SAMPLE_RATE * 1000:.0f}ms)")
        print(f"Hop size: {self.hop_size} samples ({self.hop_size / SAMPLE_RATE * 1000:.0f}ms)")
        print(f"Input device: {input_device or 'default'}")
        print(f"Output device: {output_device or 'default'}")
        print(f"Platform: {platform.system()} ({'Windows' if IS_WINDOWS else 'Unix/Linux'})")
        print("\nPress Ctrl+C or 'q' to stop\n")

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
                    print("🎤 DEBUG: Passthrough mode running!")
                    print("   You should hear your microphone input directly")
                else:
                    print("🎤 AI Speaker Recognition System running!")
                    print(f"   Model: {self.current_model_type}")
                    print(
                        f"   Selected speaker: {'ALL (similarity-based)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
                    print(f"   Processing pipeline: Denoise → Separate → Recognize → Denoise → Filter")
                    print(f"   Display interval: {self.display_interval} windows")
                    print(f"   Use 'n' to select next active speaker")
                    print(f"   Use 'a' for ALL speakers mode")
                    print(f"   Use '0'-'9' to select speaker by ID")
                    print(f"   Use 'i' to show speaker information")
                    print(f"   Use 's' to show current source assignments")
                    print(f"   Use 'l' to list registered speakers")
                    print(f"   Use 'd' to toggle debug mode")
                    print(f"   Use 'Tab' to switch separation models")
                    if IS_WINDOWS:
                        print(f"   Note: On Windows, press keys directly (no Enter needed)")

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
                            selected_info = "Selected: ALL (similarity-based)"
                        else:
                            # Find speaker name
                            speaker_name = f"Speaker_{self.selected_speaker_id}"
                            for spk in active_speakers:
                                if str(spk['id']) == str(self.selected_speaker_id):
                                    speaker_name = spk['name']
                                    break
                            selected_info = f"Selected: {speaker_name}"

                        print(f"Status: {audio_status}, "
                              f"model={self.current_model_type}, "
                              f"active_speakers={active_count}, "
                              f"{selected_info}, "
                              f"input_gain={self.input_gain:.1f}, "
                              f"processed={self.samples_processed}")
                        last_status_time = now

        except KeyboardInterrupt:
            print("\n\nStopping processor...")
        except Exception as e:
            print(f"Audio stream error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            processing_thread.join(timeout=1.0)
            keyboard_thread.join(timeout=0.5)
            input_handler.restore()
            print(f"Processor stopped. Total samples processed: {self.samples_processed}")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Real-time audio processor with speaker recognition - WINDOWS VERSION")
    parser.add_argument("--denoise-model", default="denoiser_ckpt.pt", help="Path to denoising model")
    parser.add_argument("--separation-model", default="sep_model_new.pt",
                        help="Path to main separation model (3 sources)")
    parser.add_argument("--separation-model2", default="sep_model2.pt",
                        help="Path to alternative separation model (2 sources)")
    parser.add_argument("--vad-model", default=None, help="Path to VAD model (optional)")
    parser.add_argument("--num-sources", type=int, default=3, help="Initial number of sources in separation model")
    parser.add_argument("--input-device", type=int, default=None, help="Input device ID")
    parser.add_argument("--output-device", type=int, default=None, help="Output device ID")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size")
    parser.add_argument("--denoise-strength", type=float, default=0.7,
                        help="First denoising strength (before separation, 0.3-0.9)")
    parser.add_argument("--second-denoise-strength", type=float, default=0.8,
                        help="Second denoising strength (after separation, 0.2-0.7)")
    parser.add_argument("--input-gain", type=float, default=3.0,
                        help="Input gain (0.5-10.0)")
    parser.add_argument("--output-gain", type=float, default=1.0,
                        help="Output gain (0.5-5.0)")
    parser.add_argument("--separation-gain", type=float, default=50.0,
                        help="Max gain applied after separation (1.0-100.0)")
    parser.add_argument("--debug-passthrough", action="store_true",
                        help="Debug mode: skip all processing")

    args = parser.parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

    print(f"Configuration:")
    print(f"  Platform: {platform.system()}")
    print(f"  Denoising model: {args.denoise_model}")
    print(f"  Main separation model (3-source): {args.separation_model}")
    print(f"  Alternative separation model (2-source): {args.separation_model2}")
    print(f"  Speaker recognition: YOUR MODEL INTEGRATED")
    print(f"  Selection: BY SPEAKER ID")
    print(f"  Selection method: SIMILARITY-BASED (not loudest)")
    print(f"  Initial number of sources: {args.num_sources}")
    print(f"  Input device: {args.input_device or 'default'}")
    print(f"  Output device: {args.output_device or 'default'}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_SIZE} samples")
    print(f"  First denoise strength: {args.denoise_strength}")
    print(f"  Second denoise strength: {args.second_denoise_strength}")
    print(f"  Input gain: {args.input_gain}")
    print(f"  Output gain: {args.output_gain}")
    print(f"  Max separation gain: {args.separation_gain}")
    print(f"  Debug passthrough: {args.debug_passthrough}")

    try:
        processor = RealTimeDenoiserSeparator(
            denoise_model_path=args.denoise_model,
            separation_model_path=args.separation_model,
            separation_model_path2=args.separation_model2,
            vad_model_path=args.vad_model,
            denoise_strength=args.denoise_strength,
            second_denoise_strength=args.second_denoise_strength,
            input_gain=args.input_gain,
            output_gain=args.output_gain,
            separation_gain=args.separation_gain,
            num_sources=args.num_sources,
            debug_passthrough=args.debug_passthrough
        )
        processor.run(args.input_device, args.output_device)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()