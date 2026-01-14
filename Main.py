#!/usr/bin/env python3
"""
Real-time Audio Denoiser + Separator + Speaker Recognition System
Integrates your trained speaker recognition model for speaker selection
WITH THRESHOLD-BASED OUTPUT
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
from asteroid.models import ConvTasNet
import os
import sys
import warnings
from collections import defaultdict, deque
import json
import pickle
from datetime import datetime
import librosa
import platform

warnings.filterwarnings('ignore')

# Global variables
SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Importy zależne od systemu operacyjnego
if platform.system() == 'Windows':
    import msvcrt
else:
    import select
    import termios
    import tty
    import fcntl


# ============================================
# NON-BLOCKING INPUT HANDLER (Windows/Linux)
# ============================================

class NonBlockingInput:
    """Non-blocking keyboard input handler for Windows and Linux"""

    def __init__(self):
        self.is_windows = platform.system() == 'Windows'

        if not self.is_windows:
            # Linux/Unix setup
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            # Set non-blocking mode
            old_flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, old_flags | os.O_NONBLOCK)

        self.buffer = []

    def get_key(self):
        """Get a key if available"""
        if self.is_windows:
            # Windows implementation using msvcrt
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getch()
                    # Handle special keys
                    if char == b'\xe0':  # Special function key
                        ch2 = msvcrt.getch()
                        if ch2 == b'H':
                            return 'UP'
                        elif ch2 == b'P':
                            return 'DOWN'
                        elif ch2 == b'M':
                            return 'RIGHT'
                        elif ch2 == b'K':
                            return 'LEFT'
                        return None
                    elif char == b'\x00':  # Other special key
                        msvcrt.getch()  # Skip second byte
                        return None
                    elif char == b'\r':  # Enter
                        return '\n'
                    elif char == b'\t':  # Tab
                        return '\t'
                    elif char == b'\x1b':  # Escape
                        return 'ESC'
                    else:
                        # Decode byte to string
                        try:
                            return char.decode('utf-8', errors='ignore')
                        except:
                            return None
                except Exception:
                    return None
            return None
        else:
            # Linux/Unix implementation
            try:
                # Read all available characters
                while True:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.buffer.append(ch)
                    else:
                        break
            except (IOError, TypeError):
                pass

            # Return the first character in buffer if available
            if self.buffer:
                key = self.buffer.pop(0)
                # Handle special keys
                if key == '\x1b':  # Escape sequence
                    try:
                        # Try to read more for arrow keys
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A':
                                return 'UP'
                            elif ch3 == 'B':
                                return 'DOWN'
                            elif ch3 == 'C':
                                return 'RIGHT'
                            elif ch3 == 'D':
                                return 'LEFT'
                    except (IOError, TypeError):
                        pass
                    return 'ESC'
                elif key == '\t':
                    return '\t'
                elif key == '\n':
                    return '\n'
                return key
            return None

    def restore(self):
        """Restore terminal settings (Linux only)"""
        if not self.is_windows and hasattr(self, 'old_settings'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


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
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✅ Speaker recognition model loaded")
        else:
            print("⚠️  Warning: No trained speaker model found")

        self.model.eval()

        # Speaker database
        self.speaker_database_path = "./speaker_database.pkl"
        self.speaker_embeddings = {}  # speaker_id -> embedding tensor
        self.speaker_names = {}  # speaker_id -> name

        # Similarity threshold - DODANO: możliwość konfiguracji
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
                    print(f"  Error calculating similarity for {speaker_id}: {e}")

        # Check threshold
        if best_similarity >= self.similarity_threshold:
            name = self.speaker_names.get(best_speaker_id, f"Speaker_{best_speaker_id}")
            return best_speaker_id, best_similarity, name

        return None, best_similarity, "Unknown"

    def set_similarity_threshold(self, threshold):
        """Set similarity threshold"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))


# ============================================
# AI ENHANCED SPEAKER TRACKER WITH THRESHOLD
# ============================================

class AIEnhancedSpeakerTracker:
    """Speaker tracker with YOUR recognition model and threshold control"""

    def __init__(self, max_speakers=3, similarity_threshold=0.6):
        self.max_speakers = max_speakers

        # Use YOUR speaker recognizer
        self.recognizer = SpeakerRecognizer()
        self.recognizer.set_similarity_threshold(similarity_threshold)

        # Tracking assignments {source_idx: speaker_id}
        self.source_to_speaker = {}
        self.speaker_to_source = {}
        self.speaker_activity = defaultdict(float)

        # Decision history for stability
        self.assignment_history = defaultdict(lambda: deque(maxlen=5))

        # Output control
        self.enable_output = True
        self.min_similarity_for_output = 0.4  # Minimalne podobieństwo aby przepuścić dźwięk
        self.min_energy_for_output = -50.0  # Minimalna energia w dB

        # Debug mode
        self.debug = False

        print(f"🎯 Speaker tracker initialized:")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Min similarity for output: {self.min_similarity_for_output}")
        print(f"   Min energy for output: {self.min_energy_for_output}dB")

    def set_similarity_threshold(self, threshold):
        """Set similarity threshold"""
        self.recognizer.set_similarity_threshold(threshold)

    def set_output_thresholds(self, min_similarity=0.4, min_energy=-50.0):
        """Set thresholds for output control"""
        self.min_similarity_for_output = min_similarity
        self.min_energy_for_output = min_energy

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

                # Calculate energy
                energy = 10 * np.log10(np.mean(audio ** 2) + 1e-10)

                if self.debug:
                    print(f"[TRACKER] Source {source_idx}: energy={energy:.1f}dB, "
                          f"speaker={speaker_name}, similarity={similarity:.3f}")

                if speaker_id is not None:
                    current_assignments[source_idx] = speaker_id
                    speaker_assignments[source_idx] = {
                        'speaker_id': speaker_id,
                        'name': speaker_name,
                        'similarity': similarity,
                        'energy': energy
                    }

                    # Update activity
                    self.speaker_activity[speaker_id] = current_time

                    # Update assignment tracking
                    if speaker_id not in self.speaker_to_source:
                        # New speaker, assign to this source
                        self._assign_speaker_to_source(source_idx, speaker_id)
                        if self.debug:
                            print(
                                f"[TRACKER] 🎯 Speaker '{speaker_name}' (ID: {speaker_id}) assigned to source {source_idx}")
                    elif self.speaker_to_source[speaker_id] != source_idx:
                        if self.debug:
                            print(
                                f"[TRACKER] 🔄 Speaker '{speaker_name}' (ID: {speaker_id}) switched from source {self.speaker_to_source[speaker_id]} to {source_idx}")
                        self._assign_speaker_to_source(source_idx, speaker_id)

        # Determine which sources to allow based on thresholds
        allowed_sources = []
        best_sources = []

        for source_idx, assignment in speaker_assignments.items():
            # Check if meets minimum thresholds
            if (assignment['similarity'] >= self.min_similarity_for_output and
                    assignment['energy'] >= self.min_energy_for_output):
                allowed_sources.append(source_idx)
                best_sources.append((source_idx, assignment['similarity'], assignment['energy']))

        # Sort by similarity (highest first)
        if best_sources:
            best_sources.sort(key=lambda x: x[1], reverse=True)
            # Take only the best source (closest to registered speaker)
            if best_sources:
                best_source = best_sources[0][0]
                allowed_sources = [best_source]

                if self.debug and len(best_sources) > 1:
                    print(
                        f"[TRACKER] Selected source {best_source} (similarity: {best_sources[0][1]:.3f}) from {len(best_sources)} valid sources")

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


# ============================================
# DENOISING MODEL
# ============================================

class UNet1D(nn.Module):
    """Simple 1D U-Net for denoising"""

    def __init__(self, in_chan=1, base=32):
        super().__init__()
        self.encoder1 = nn.Conv1d(in_chan, base, kernel_size=4, stride=2, padding=1)
        self.encoder2 = nn.Conv1d(base, base * 2, kernel_size=4, stride=2, padding=1)
        self.encoder3 = nn.Conv1d(base * 2, base * 4, kernel_size=4, stride=2, padding=1)

        self.decoder1 = nn.ConvTranspose1d(base * 4, base * 2, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose1d(base * 2 * 2, base, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose1d(base * 2, in_chan, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.encoder1(x))
        e2 = self.relu(self.encoder2(e1))
        e3 = self.relu(self.encoder3(e2))

        # Decoder with skip connections
        d1 = self.relu(self.decoder1(e3))
        d1 = torch.cat([d1, e2], dim=1)

        d2 = self.relu(self.decoder2(d1))
        d2 = torch.cat([d2, e1], dim=1)

        d3 = self.decoder3(d2)

        return self.sigmoid(d3) * x


# ============================================
# MAIN REAL-TIME PROCESSOR WITH THRESHOLD CONTROL
# ============================================

class RealTimeDenoiserSeparator:
    def __init__(self, denoise_model_path=None, separation_model_path=None,
                 separation_model_path2=None, vad_model_path=None,
                 denoise_strength=0.5, second_denoise_strength=0.3, input_gain=1.0, output_gain=1.0,
                 separation_gain=50.0, num_sources=3, debug_passthrough=False,
                 similarity_threshold=0.6, min_similarity_for_output=0.4):
        """
        Real-time denoiser + separator with speaker recognition and threshold control
        """
        self.denoise_strength = max(0.1, min(0.9, denoise_strength))
        self.second_denoise_strength = max(0.1, min(0.9, second_denoise_strength))
        self.input_gain = input_gain
        self.output_gain = output_gain
        self.separation_gain = separation_gain
        self.debug_passthrough = debug_passthrough
        self.num_sources = num_sources
        self.similarity_threshold = similarity_threshold
        self.min_similarity_for_output = min_similarity_for_output

        # Speaker selection
        self.selected_speaker_id = -1

        # Model paths
        self.separation_model_path = separation_model_path
        self.separation_model_path2 = separation_model_path2
        self.vad_model_path = vad_model_path
        self.current_model_type = "3-source"

        # AI speaker tracker with threshold control
        self.speaker_tracker = AIEnhancedSpeakerTracker(
            max_speakers=3,
            similarity_threshold=similarity_threshold
        )
        self.speaker_tracker.set_output_thresholds(
            min_similarity=min_similarity_for_output,
            min_energy=-50.0
        )

        # Debug flag
        self.debug_speech_detection = False

        # Output control
        self.silence_output = False  # Manual override to silence output

        # Statistics
        self.total_chunks_processed = 0
        self.chunks_passed = 0
        self.chunks_blocked = 0

        print(f"Debug passthrough mode: {debug_passthrough}")
        print(f"Using device: {DEVICE}")
        print(f"Number of sources: {num_sources}")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Min similarity for output: {min_similarity_for_output}")
        print(f"Selection method: CLOSEST SPEAKER (highest similarity)")

        if debug_passthrough:
            print("DEBUG: Running in passthrough mode - no processing")
            self.denoise_model = None
            self.sep_model = None
            self.sep_model2 = None
            self.window_size = CHUNK_SIZE * 16
            self.hop_size = CHUNK_SIZE * 4
        else:
            # Window settings
            self.window_size = int(1.5 * SAMPLE_RATE)
            self.hop_size = int(0.25 * SAMPLE_RATE)
            self.denoise_window_size = int(1.0 * SAMPLE_RATE)

            print(f"Processing window: {self.window_size} samples")

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

        # Output buffer
        self.output_buffer = np.zeros(self.hop_size * 2, dtype=np.float32)
        self.output_buffer_pos = 0

        # Queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=100)

        # Flags
        self.is_running = True
        self.samples_processed = 0
        self.last_audio_time = time.time()

        # Filters
        self.setup_simple_filters()

        # Display interval
        self.display_interval = 3

        print(f"Stream: {SAMPLE_RATE}Hz, Model: {MODEL_SAMPLE_RATE}Hz")

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
                model_input_size = 16384

                if len(audio_chunk_16k) != model_input_size:
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
        """Separate audio sources"""
        if self.current_sep_model is None or self.debug_passthrough or self.num_sources <= 1 or len(
                audio_chunk_16k) == 0:
            return torch.tensor([audio_chunk_16k])

        try:
            with torch.no_grad():
                model_input_size = 16384

                if len(audio_chunk_16k) != model_input_size:
                    if len(audio_chunk_16k) < model_input_size:
                        pad_len = model_input_size - len(audio_chunk_16k)
                        audio_padded = np.pad(audio_chunk_16k, (0, pad_len), mode='reflect')
                    else:
                        start = (len(audio_chunk_16k) - model_input_size) // 2
                        audio_padded = audio_chunk_16k[start:start + model_input_size]
                else:
                    audio_padded = audio_chunk_16k

                audio_tensor = torch.from_numpy(audio_padded).float()
                audio_tensor = audio_tensor.unsqueeze(0).to(DEVICE)

                separated = self.current_sep_model(audio_tensor)
                separated = separated.squeeze(0)

                return separated
        except Exception as e:
            print(f"Error in separation: {e}")
            return torch.tensor([audio_chunk_16k])

    def process_denoise_separate_denoise(self, audio_chunk_48k):
        """Processing pipeline with threshold control"""
        if self.debug_passthrough:
            return audio_chunk_48k

        self.total_chunks_processed += 1
        should_display = (self.total_chunks_processed % self.display_interval == 0)

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
        denoised_16k = self.denoise_audio(audio_16k, self.denoise_strength)

        # 3. SEPARATION
        separated_sources = self.separate_sources(denoised_16k)

        # 4. SPEAKER RECOGNITION FOR EACH SOURCE
        speaker_assignments, allowed_sources = self.speaker_tracker.process_batch(separated_sources)

        # 5. SELECT SOURCE WITH HIGHEST SIMILARITY (CLOSEST SPEAKER)
        selected_audio = None
        selected_info = ""

        if allowed_sources:
            # Find source with highest similarity
            best_source_idx = -1
            best_similarity = -1.0
            best_speaker_name = "Unknown"

            for source_idx in allowed_sources:
                if source_idx in speaker_assignments:
                    assignment = speaker_assignments[source_idx]
                    similarity = assignment['similarity']
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_source_idx = source_idx
                        best_speaker_name = assignment['name']

            if best_source_idx != -1 and not self.silence_output:
                selected_audio = separated_sources[best_source_idx].cpu().numpy()
                selected_info = f"🎤 Source {best_source_idx}: {best_speaker_name} (similarity={best_similarity:.3f})"
                self.chunks_passed += 1

                if should_display:
                    energy = 10 * np.log10(np.mean(selected_audio ** 2) + 1e-10)
                    print(f"[SELECT] ✅ {selected_info}, energy={energy:.1f}dB")
            else:
                # Below threshold or manual silence
                selected_audio = np.zeros_like(denoised_16k)
                selected_info = "🔇 Below threshold or silenced"
                self.chunks_blocked += 1

                if should_display:
                    print(f"[SELECT] {selected_info}")
        else:
            # No valid sources
            selected_audio = np.zeros_like(denoised_16k)
            selected_info = "🔇 No valid sources"
            self.chunks_blocked += 1

            if should_display:
                print(f"[SELECT] {selected_info}")

        # 6. POST-SEPARATION GAIN
        source_peak = np.max(np.abs(selected_audio))
        if source_peak > 0:
            auto_gain = min(0.5 / source_peak, self.separation_gain)
            effective_gain = min(self.separation_gain, auto_gain)
            selected_audio = selected_audio * effective_gain

        selected_audio = np.clip(selected_audio, -1.0, 1.0)

        # 7. SECOND DENOISING
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

        # 10. Apply filter
        output_48k = self.apply_filter(output_48k)

        # 11. Match size
        if len(output_48k) > len(audio_chunk_48k):
            output_48k = output_48k[:len(audio_chunk_48k)]
        elif len(output_48k) < len(audio_chunk_48k):
            output_48k = np.pad(output_48k,
                                (0, len(audio_chunk_48k) - len(output_48k)),
                                mode='constant')

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
        """Keyboard listener with new commands"""
        print("\n[KEYBOARD] Speaker Recognition Commands:")
        print("[KEYBOARD] 'Tab' - switch separation models")
        print("[KEYBOARD] 'm' - toggle output (silence/all)")
        print("[KEYBOARD] '+' - increase similarity threshold (+0.05)")
        print("[KEYBOARD] '-' - decrease similarity threshold (-0.05)")
        print("[KEYBOARD] 'd' - toggle debug mode")
        print("[KEYBOARD] 's' - show statistics")
        print("[KEYBOARD] 'q' - quit program")

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
                        elif key == 'm' or key == 'M':
                            self.silence_output = not self.silence_output
                            status = "🔇 SILENCED" if self.silence_output else "🔊 ENABLED"
                            print(f"\n[OUTPUT] {status}")
                            last_key_time = current_time
                        elif key == '+':
                            new_threshold = self.speaker_tracker.recognizer.similarity_threshold + 0.05
                            if new_threshold <= 1.0:
                                self.speaker_tracker.set_similarity_threshold(new_threshold)
                                print(f"\n[THRESHOLD] Increased to: {new_threshold:.2f}")
                            last_key_time = current_time
                        elif key == '-':
                            new_threshold = self.speaker_tracker.recognizer.similarity_threshold - 0.05
                            if new_threshold >= 0.0:
                                self.speaker_tracker.set_similarity_threshold(new_threshold)
                                print(f"\n[THRESHOLD] Decreased to: {new_threshold:.2f}")
                            last_key_time = current_time
                        elif key == 'd' or key == 'D':
                            self.debug_speech_detection = not self.debug_speech_detection
                            self.speaker_tracker.debug = self.debug_speech_detection
                            print(f"\n[DEBUG] Debug mode: {'ENABLED' if self.debug_speech_detection else 'DISABLED'}")
                            last_key_time = current_time
                        elif key == 's' or key == 'S':
                            if self.total_chunks_processed > 0:
                                pass_rate = (self.chunks_passed / self.total_chunks_processed) * 100
                                print(f"\n[STATS] Chunks processed: {self.total_chunks_processed}")
                                print(f"[STATS] Chunks passed: {self.chunks_passed} ({pass_rate:.1f}%)")
                                print(f"[STATS] Chunks blocked: {self.chunks_blocked} ({100 - pass_rate:.1f}%)")
                                print(
                                    f"[STATS] Similarity threshold: {self.speaker_tracker.recognizer.similarity_threshold:.2f}")
                            last_key_time = current_time
                        elif key == 'q' or key == 'Q':
                            print("\n[KEYBOARD] Detected 'q' - quitting...")
                            self.is_running = False
                            break

                time.sleep(0.01)

            except Exception as e:
                print(f"[KEYBOARD] Listener error: {e}")
                time.sleep(0.1)

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

    def processing_loop(self):
        """Main processing loop"""
        print("Processing loop started...")

        if self.debug_passthrough:
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
        else:
            processing_buffer = np.zeros(self.window_size, dtype=np.float32)
            buffer_fill = 0
            window = np.hanning(self.window_size)

            while self.is_running:
                try:
                    # Get input data
                    try:
                        raw_chunk = self.input_queue.get(timeout=0.1)
                    except queue.Empty:
                        if buffer_fill > 0:
                            window_to_process = processing_buffer[:buffer_fill].copy()
                            if len(window_to_process) >= int(0.5 * SAMPLE_RATE):
                                window_part = window[:buffer_fill]
                                windowed = window_to_process * window_part

                                processed = self.process_denoise_separate_denoise(windowed)
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

                    # Add to buffer
                    if buffer_fill + len(raw_chunk) <= self.window_size:
                        processing_buffer[buffer_fill:buffer_fill + len(raw_chunk)] = raw_chunk
                        buffer_fill += len(raw_chunk)
                    else:
                        # Process full window
                        window_to_process = processing_buffer.copy()
                        windowed = window_to_process * window

                        processed = self.process_denoise_separate_denoise(windowed)

                        # Send processed data
                        samples_to_send = min(self.hop_size, len(processed))
                        for i in range(0, samples_to_send, CHUNK_SIZE):
                            chunk = processed[i:i + CHUNK_SIZE].copy()
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass

                        # Shift buffer
                        shift = self.hop_size
                        processing_buffer = np.roll(processing_buffer, -shift)
                        processing_buffer[-shift:] = 0
                        buffer_fill = max(0, buffer_fill - shift)

                        # Add new data
                        if buffer_fill + len(raw_chunk) <= self.window_size:
                            processing_buffer[buffer_fill:buffer_fill + len(raw_chunk)] = raw_chunk
                            buffer_fill += len(raw_chunk)
                        else:
                            processing_buffer[-len(raw_chunk):] = raw_chunk
                            buffer_fill = self.window_size

                    # Process if buffer is full
                    if buffer_fill >= self.window_size:
                        window_to_process = processing_buffer.copy()
                        windowed = window_to_process * window

                        processed = self.process_denoise_separate_denoise(windowed)

                        samples_to_send = min(self.hop_size, len(processed))
                        for i in range(0, samples_to_send, CHUNK_SIZE):
                            chunk = processed[i:i + CHUNK_SIZE].copy()
                            if len(chunk) > 0:
                                try:
                                    self.output_queue.put_nowait(chunk)
                                except queue.Full:
                                    pass

                        # Shift buffer
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
        print(f"Selection method: CLOSEST SPEAKER (highest similarity)")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Min similarity for output: {self.min_similarity_for_output}")
        print(f"Model type: {self.current_model_type}")
        print(f"Number of sources: {self.num_sources}")
        print(f"First denoise strength: {self.denoise_strength}")
        print(f"Second denoise strength: {self.second_denoise_strength}")
        print(f"Max separation gain: {self.separation_gain}x")
        print(f"Input gain: {self.input_gain}")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Processing window: {self.window_size} samples")
        print(f"Hop size: {self.hop_size} samples")
        print(f"Input device: {input_device or 'default'}")
        print(f"Output device: {output_device or 'default'}")
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
                else:
                    print("🎤 AI Speaker Recognition System running!")
                    print(f"   Processing pipeline: Denoise → Separate → Recognize → Denoise → Filter")
                    print(f"   Use '+'/'-' to adjust similarity threshold")
                    print(f"   Use 'm' to toggle output")
                    print(f"   Use 's' to show statistics")
                    print(f"   Use 'd' to toggle debug mode")

                last_status_time = time.time()
                last_stat_time = time.time()

                while self.is_running:
                    time.sleep(0.1)
                    now = time.time()

                    # Show status every 5 seconds
                    if now - last_status_time >= 5:
                        time_since_audio = now - self.last_audio_time

                        if time_since_audio < 5:
                            audio_status = "🎤 Audio active"
                        else:
                            audio_status = "🔇 No audio"

                        status_msg = f"Status: {audio_status}, "
                        status_msg += f"model={self.current_model_type}, "
                        status_msg += f"threshold={self.speaker_tracker.recognizer.similarity_threshold:.2f}, "
                        status_msg += f"output={'🔇' if self.silence_output else '🔊'}"

                        print(status_msg)
                        last_status_time = now

                    # Show statistics every 30 seconds
                    if now - last_stat_time >= 30 and self.total_chunks_processed > 0:
                        pass_rate = (self.chunks_passed / self.total_chunks_processed) * 100
                        print(
                            f"[STATS] Pass rate: {pass_rate:.1f}% ({self.chunks_passed}/{self.total_chunks_processed})")
                        last_stat_time = now

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

            # Final statistics
            if self.total_chunks_processed > 0:
                pass_rate = (self.chunks_passed / self.total_chunks_processed) * 100
                print(f"\n📊 FINAL STATISTICS:")
                print(f"   Total chunks processed: {self.total_chunks_processed}")
                print(f"   Chunks passed: {self.chunks_passed} ({pass_rate:.1f}%)")
                print(f"   Chunks blocked: {self.chunks_blocked} ({100 - pass_rate:.1f}%)")

            print(f"Processor stopped. Total samples processed: {self.samples_processed}")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Real-time audio processor with speaker recognition")
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
    # New arguments for threshold control
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="Similarity threshold for speaker recognition (0.0-1.0)")
    parser.add_argument("--min-similarity-output", type=float, default=0.4,
                        help="Minimum similarity to pass audio to output (0.0-1.0)")

    args = parser.parse_args()

    global CHUNK_SIZE
    CHUNK_SIZE = args.chunk_size

    print(f"Configuration:")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  Min similarity for output: {args.min_similarity_output}")
    print(f"  Selection: CLOSEST SPEAKER (highest similarity)")
    print(f"  Dźwięk będzie przepuszczany tylko jeśli podobieństwo >= {args.min_similarity_output}")

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
            debug_passthrough=args.debug_passthrough,
            similarity_threshold=args.similarity_threshold,
            min_similarity_for_output=args.min_similarity_output
        )
        processor.run(args.input_device, args.output_device)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()