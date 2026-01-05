#!/usr/bin/env python3
"""
Real-time Audio Denoiser + Speaker Recognition System z fail-safe mechanizmem
Uses denoising model and speaker recognition to pass only selected speaker's audio
with smooth audio output using overlap-add method and fail-safe protection
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
from train_denoiser import UNet1D
import os
import sys
import select
import termios
import tty
import fcntl
import warnings
from collections import defaultdict, deque
import json
import pickle
from datetime import datetime
import librosa
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
        self.speaker_names = {}       # speaker_id -> name
        
        # Similarity threshold
        self.similarity_threshold = 0.3
        
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
                    # Print speaker names
                    for speaker_id, name in self.speaker_names.items():
                        print(f"  ID {speaker_id}: {name}")
                else:
                    print("⚠️  Invalid database format")
            except Exception as e:
                print(f"❌ Error loading database: {e}")
        else:
            print("📊 Creating new speaker database")
    
    def extract_embedding(self, audio):
        """Extract speaker embedding from audio using YOUR model"""
        if len(audio) < 24000:  # Need at least 1.5 second for better recognition
            return None
        
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
# AI ENHANCED SPEAKER TRACKER WITH FAIL-SAFE
# ============================================

class AIEnhancedSpeakerTracker:
    """Speaker tracker with fail-safe mechanism"""
    
    def __init__(self):
        # Use YOUR speaker recognizer
        self.recognizer = SpeakerRecognizer()
        
        # Speaker selection mode
        self.selected_speaker_id = -1  # -1 = auto (similarity-based), specific ID = only that speaker
        self.block_other_speakers = False
        
        # FAIL-SAFE MECHANISM
        self.fail_safe_enabled = True
        self.consecutive_matches = 0  # Licznik kolejnych dopasowań
        self.consecutive_misses = 0   # Licznik kolejnych niedopasowań
        self.fail_safe_counter = 0    # Licznik fail-safe (ile próbek przepuścić po serii dopasowań)
        
        # Parametry fail-safe
        self.fail_safe_match_threshold = 3  # Po ilu dopasowaniach włączyć fail-safe
        self.fail_safe_duration = 5         # Ile próbek przepuścić w trybie fail-safe
        self.max_fail_safe_attempts = 3     # Maksymalna liczba aktywacji fail-safe z rzędu
        
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
            
            # Jeśli mamy serię dopasowań, aktywuj fail-safe
            if (self.consecutive_matches >= self.fail_safe_match_threshold and 
                not self.fail_safe_active and
                self.fail_safe_enabled):
                
                # Sprawdź czy nie przekraczamy limitu aktywacji
                if (time.time() - self.last_fail_safe_time > 10 or 
                    self.fail_safe_activations < self.max_fail_safe_attempts):
                    
                    self.fail_safe_counter = self.fail_safe_duration
                    self.fail_safe_active = True
                    self.fail_safe_activations += 1
                    self.last_fail_safe_time = time.time()
                    self.stats['fail_safe_activations'] += 1
                    
                    if self.debug:
                        print(f"[FAIL-SAFE] 🔄 Aktywacja! Przepuszczam następne {self.fail_safe_duration} próbek")
        else:
            self.consecutive_misses += 1
            self.consecutive_matches = 0
            self.stats['failed_frames'] += 1
        
        # Aktualizuj licznik fail-safe
        if self.fail_safe_counter > 0:
            self.fail_safe_counter -= 1
            self.stats['fail_safe_frames'] += 1
            if self.debug and self.fail_safe_counter == 0:
                print(f"[FAIL-SAFE] ✅ Tryb fail-safe zakończony")
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
                return self.last_recognized_speaker, True, self.last_similarity, self.recognizer.speaker_names.get(str(self.last_recognized_speaker), f"Speaker_{self.last_recognized_speaker}")
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
        
        # FAIL-SAFE LOGIC: Jeśli jesteśmy w trybie fail-safe, przepuszczamy audio
        if self.fail_safe_counter > 0:
            should_pass = True
            if self.debug:
                print(f"[FAIL-SAFE] 🛡️  Przepuszczam (pozostało: {self.fail_safe_counter})")
        
        # Normal logic (jeśli nie w fail-safe)
        elif self.block_other_speakers and self.selected_speaker_id != -1:
            # Mode: only selected speaker
            if is_match:
                # Add to decision buffer
                self.decision_buffer.append(True)
                should_pass = True
                if self.debug:
                    avg_similarity = np.mean(self.speaker_history.get(speaker_id, [similarity]))
                    print(f"[TRACKER] ✅ Passing audio from '{speaker_name}' (sim={similarity:.3f}, avg={avg_similarity:.3f})")
            else:
                # Add to decision buffer
                self.decision_buffer.append(False)
                if self.debug:
                    if speaker_id is not None:
                        print(f"[TRACKER] ❌ Blocking '{speaker_name}' (not selected, sim={similarity:.3f})")
                    else:
                        print(f"[TRACKER] ❌ Blocking (no speaker recognized)")
        else:
            # Mode: all speakers or auto
            should_pass = True
            if speaker_id is not None and self.debug:
                avg_similarity = np.mean(self.speaker_history.get(speaker_id, [similarity]))
                print(f"[TRACKER] 🔊 Passing '{speaker_name}' (sim={similarity:.3f}, avg={avg_similarity:.3f})")
        
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
                avg_similarity = np.mean(self.speaker_history.get(speaker_id, [0])) if self.speaker_history.get(speaker_id) else 0
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
            return "Brak danych"
        
        match_rate = self.stats['matched_frames'] / self.stats['total_frames'] * 100
        fail_safe_rate = self.stats['fail_safe_frames'] / self.stats['total_frames'] * 100
        
        return (f"Match rate: {match_rate:.1f}%, "
                f"Fail-safe: {self.stats['fail_safe_activations']} aktywacji, "
                f"Fail-safe frames: {fail_safe_rate:.1f}%")
    
    def toggle_fail_safe(self):
        """Toggle fail-safe mode"""
        self.fail_safe_enabled = not self.fail_safe_enabled
        return self.fail_safe_enabled

# ============================================
# NON-BLOCKING INPUT
# ============================================

class NonBlockingInput:
    """Non-blocking stdin input"""
    def __init__(self):
        self.old_settings = None
        self.setup_nonblocking()
        
    def setup_nonblocking(self):
        """Configure stdin for non-blocking read"""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        fd = sys.stdin.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    
    def restore(self):
        """Restore terminal settings"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Get key if pressed, otherwise None"""
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

class RealTimeDenoiserSpeakerFilter:
    def __init__(self, denoise_model_path=None, vad_model_path=None,
                 denoise_strength=0.5, input_gain=1.0, output_gain=1.0,
                 speaker_gain=1.0, debug_passthrough=False):
        """
        Real-time denoiser with speaker filtering and fail-safe
        """
        self.denoise_strength = max(0.1, min(0.9, denoise_strength))
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
        
        # Licznik dla ograniczenia wyświetlania
        self.process_counter = 0
        self.display_interval = 20  # Wyświetlaj informacje co 20 procesowań
        
        # Audio processing parameters for smooth output
        self.window_size = int(0.5 * SAMPLE_RATE)  # 0.5 second window for processing
        self.hop_size = int(0.25 * SAMPLE_RATE)    # 0.25 second hop (50% overlap)
        
        print(f"Debug passthrough mode: {debug_passthrough}")
        print(f"Using device: {DEVICE}")
        print(f"Denoise strength: {denoise_strength}")
        print(f"Speaker gain: {speaker_gain}")
        print(f"Speaker recognition: YOUR MODEL INTEGRATED")
        print(f"Selection mode: BY SPEAKER ID")
        print(f"FAIL-SAFE: ENABLED ({self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples)")
        print(f"Processing window: {self.window_size} samples ({self.window_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Hop size: {self.hop_size} samples ({self.hop_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Overlap: 50% for smooth audio output")
        
        if debug_passthrough:
            print("DEBUG: Running in passthrough mode - no processing")
            self.denoise_model = None
        else:
            # Load denoising model
            if denoise_model_path and os.path.exists(denoise_model_path):
                print(f"Loading denoising model from: {denoise_model_path}")
                self.load_denoising_model(denoise_model_path)
            else:
                print("WARNING: No denoising model, skipping denoising steps")
                self.denoise_model = None
        
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
    
    def setup_simple_filters(self):
        """Simple filters"""
        nyquist = 0.5 * SAMPLE_RATE
        self.b_hp, self.a_hp = signal.butter(2, 80/nyquist, btype='high')
        self.filter_state = None
    
    def apply_filter(self, audio):
        """Apply filter"""
        if len(audio) == 0:
            return audio
        
        if self.filter_state is None:
            filtered, self.filter_state = signal.lfilter(self.b_hp, self.a_hp, audio,
                                                       zi=np.zeros(max(len(self.a_hp), len(self.b_hp))-1))
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
    
    def select_speaker(self, speaker_id):
        """Select which speaker to allow through"""
        if speaker_id == -1:
            self.selected_speaker_id = -1
            self.speaker_tracker.set_selected_speaker(-1)
            print(f"\n🎤 Mode: ALL SPEAKERS (no filtering)")
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
            print(f"   Fail-safe: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")
    
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
                      f"Avg similarity: {spk['avg_similarity']:.3f}, "
                      f"Active: {time.time() - spk['last_active']:.1f}s ago {status}")
        else:
            print("\n[SPEAKER INFO] No active speakers")
        
        # Fail-safe stats
        print(f"\n[FAIL-SAFE] Stats: {self.speaker_tracker.get_fail_safe_stats()}")
        print(f"[FAIL-SAFE] Enabled: {self.speaker_tracker.fail_safe_enabled}")
        print(f"[FAIL-SAFE] Active: {self.speaker_tracker.fail_safe_active}")
        print(f"[FAIL-SAFE] Counter: {self.speaker_tracker.fail_safe_counter}")
    
    def toggle_speech_debug(self):
        """Toggle debug mode"""
        self.debug_speech_detection = not self.debug_speech_detection
        self.speaker_tracker.debug = self.debug_speech_detection
        print(f"[DEBUG] Debug mode: {'ENABLED' if self.debug_speech_detection else 'DISABLED'}")
    
    def toggle_fail_safe(self):
        """Toggle fail-safe mode"""
        enabled = self.speaker_tracker.toggle_fail_safe()
        print(f"[FAIL-SAFE] Fail-safe mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def adjust_fail_safe_params(self):
        """Adjust fail-safe parameters"""
        print("\n[FAIL-SAFE] Current parameters:")
        print(f"   Matches to activate: {self.speaker_tracker.fail_safe_match_threshold}")
        print(f"   Samples to pass: {self.speaker_tracker.fail_safe_duration}")
        
        try:
            new_threshold = input(f"   New match threshold [{self.speaker_tracker.fail_safe_match_threshold}]: ").strip()
            if new_threshold:
                self.speaker_tracker.fail_safe_match_threshold = int(new_threshold)
            
            new_duration = input(f"   New fail-safe duration [{self.speaker_tracker.fail_safe_duration}]: ").strip()
            if new_duration:
                self.speaker_tracker.fail_safe_duration = int(new_duration)
            
            print(f"[FAIL-SAFE] Updated: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")
        except ValueError:
            print("[FAIL-SAFE] Invalid input, keeping current values")
    
    def process_audio_with_speaker_filter(self, audio_chunk_48k):
        """Process audio: denoise + speaker recognition + filtering with fail-safe"""
        if self.debug_passthrough:
            return audio_chunk_48k
        
        # Zwiększ licznik i sprawdź czy wyświetlać informacje
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
            print(f"Error in resampling to 16k: {e}")
            return audio_chunk_48k
        
        # 2. SPEAKER RECOGNITION WITH FAIL-SAFE
        if should_display:
            print(f"[PROCESS] Step 1: Speaker recognition (fail-safe: {self.speaker_tracker.fail_safe_enabled})")
        
        # Recognize speaker in this audio batch (with fail-safe logic)
        speaker_id, should_pass, similarity, speaker_name = self.speaker_tracker.process_audio(audio_16k)
        
        # 3. DENOISING
        if should_display:
            print(f"[PROCESS] Step 2: Denoising (strength={self.denoise_strength})")
        
        denoised_16k = self.denoise_audio(audio_16k, self.denoise_strength)
        
        # 4. FILTER BASED ON SELECTED SPEAKER (with fail-safe consideration)
        if should_pass:
            # Pass the audio (it's from the selected speaker or fail-safe is active)
            selected_audio = denoised_16k
            if should_display:
                if speaker_id is not None:
                    status = "FAIL-SAFE" if self.speaker_tracker.fail_safe_counter > 0 else "MATCH"
                    print(f"[FILTER] ✅ {status}: Passing audio from '{speaker_name}' (similarity={similarity:.3f})")
                else:
                    if self.speaker_tracker.fail_safe_counter > 0:
                        print(f"[FILTER] 🛡️  FAIL-SAFE: Passing audio (counter: {self.speaker_tracker.fail_safe_counter})")
                    else:
                        print(f"[FILTER] 🔊 Passing audio (no speaker filtering)")
        else:
            # Block audio (not from selected speaker)
            selected_audio = np.zeros_like(denoised_16k)
            if should_display:
                if speaker_id is not None:
                    print(f"[FILTER] 🔇 Blocking audio from '{speaker_name}' (not selected)")
                else:
                    print(f"[FILTER] 🔇 Blocking audio (no speaker recognized)")
        
        # 5. APPLY SPEAKER GAIN
        if should_pass and speaker_id is not None:
            audio_peak = np.max(np.abs(selected_audio))
            if audio_peak > 0:
                selected_audio = selected_audio * min(self.speaker_gain, 0.5 / audio_peak)
        
        selected_audio = np.clip(selected_audio, -1.0, 1.0)
        
        # 6. Match length
        if len(selected_audio) > len(audio_16k):
            selected_audio = selected_audio[:len(audio_16k)]
        elif len(selected_audio) < len(audio_16k):
            selected_audio = np.pad(selected_audio, 
                                  (0, len(audio_16k) - len(selected_audio)), 
                                  mode='constant')
        
        # 7. Resample back to 48kHz
        try:
            output_48k = resampy.resample(
                selected_audio,
                MODEL_SAMPLE_RATE,
                SAMPLE_RATE,
                filter='kaiser_fast'
            )
        except Exception as e:
            print(f"Error in resampling to 48k: {e}")
            return audio_chunk_48k
        
        # 8. Apply identity filter (high-pass)
        if should_display:
            print(f"[PROCESS] Step 3: Identity filter")
        
        output_48k = self.apply_filter(output_48k)
        
        # 9. Match size
        if len(output_48k) > len(audio_chunk_48k):
            output_48k = output_48k[:len(audio_chunk_48k)]
        elif len(output_48k) < len(audio_chunk_48k):
            output_48k = np.pad(output_48k,
                              (0, len(audio_chunk_48k) - len(output_48k)),
                              mode='constant')
        
        if should_display:
            fail_safe_status = "ACTIVE" if self.speaker_tracker.fail_safe_counter > 0 else "INACTIVE"
            print(f"[PROCESS] Completed: Recognize → Denoise → Filter | Fail-safe: {fail_safe_status}")
        
        return output_48k
    
    def input_callback(self, indata, frames, time_info, status):
        """Input callback - collect audio chunks"""
        if status:
            print(f"Input status: {status}")
        
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
            print(f"Output status: {status}")
        
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
        print("\n[KEYBOARD] Speaker Recognition Commands:")
        print("[KEYBOARD] 'n' - select next active speaker")
        print("[KEYBOARD] 'a' - ALL speakers mode (no filtering)")
        print("[KEYBOARD] 'i' - show speaker information")
        print("[KEYBOARD] 'd' - toggle debug mode")
        print("[KEYBOARD] 'f' - toggle fail-safe mode")
        print("[KEYBOARD] 'F' (Shift+f) - adjust fail-safe parameters")
        print("[KEYBOARD] 'l' - list registered speakers")
        print("[KEYBOARD] '0'-'9' - select speaker by ID")
        print("[KEYBOARD] 'g' - increase input gain (+0.5)")
        print("[KEYBOARD] 'G' (Shift+g) - decrease input gain (-0.5)")
        print("[KEYBOARD] '+' - increase speaker gain (+1.0)")
        print("[KEYBOARD] '-' - decrease speaker gain (-1.0)")
        print("[KEYBOARD] 't' - set similarity threshold")
        print("[KEYBOARD] 's' - show fail-safe statistics")
        print("[KEYBOARD] 'q' - quit program")
        
        last_key_time = 0
        key_debounce = 0.3
        
        while self.is_running:
            try:
                key = input_handler.get_key()
                
                if key:
                    current_time = time.time()
                    
                    if current_time - last_key_time > key_debounce:
                        if key == 'n' or key == 'N':
                            self.select_next_speaker()
                            last_key_time = current_time
                        elif key == 'a' or key == 'A':
                            self.select_speaker(-1)  # All speakers mode
                            last_key_time = current_time
                        elif key == 'i' or key == 'I':
                            self.show_speaker_info()
                            last_key_time = current_time
                        elif key == 'd' or key == 'D':
                            self.toggle_speech_debug()
                            last_key_time = current_time
                        elif key == 'f' or key == 'F':
                            if key == 'f':
                                self.toggle_fail_safe()
                            else:
                                self.adjust_fail_safe_params()
                            last_key_time = current_time
                        elif key == 'l' or key == 'L':
                            print("\n[SPEAKERS] Registered speakers:")
                            speakers = self.speaker_tracker.list_registered_speakers()
                            for line in speakers:
                                print(f"  {line}")
                            last_key_time = current_time
                        elif key == 's' or key == 'S':
                            print(f"\n[STATS] Fail-safe statistics:")
                            print(f"  {self.speaker_tracker.get_fail_safe_stats()}")
                            print(f"  Consecutive matches: {self.speaker_tracker.consecutive_matches}")
                            print(f"  Fail-safe counter: {self.speaker_tracker.fail_safe_counter}")
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
                        elif key == '+':
                            self.speaker_gain = min(10.0, self.speaker_gain + 1.0)
                            print(f"[GAIN] Increased speaker gain to: {self.speaker_gain:.1f}")
                            last_key_time = current_time
                        elif key == '-':
                            self.speaker_gain = max(1.0, self.speaker_gain - 1.0)
                            print(f"[GAIN] Decreased speaker gain to: {self.speaker_gain:.1f}")
                            last_key_time = current_time
                        elif key == 't' or key == 'T':
                            try:
                                print(f"\n[THRESHOLD] Current similarity threshold: {self.speaker_tracker.recognizer.similarity_threshold:.2f}")
                                print("[THRESHOLD] Enter new threshold (0.1-0.9): ")
                                # Simple input
                                time.sleep(1)
                                # Możesz dodać prawdziwe wczytywanie tutaj
                                # Dla uproszczenia ustawmy 0.3
                                self.speaker_tracker.recognizer.similarity_threshold = 0.3
                                print(f"[THRESHOLD] Set similarity threshold to: 0.3")
                            except:
                                pass
                            last_key_time = current_time
                        elif key == 'q' or key == 'Q':
                            print("\n[KEYBOARD] Detected 'q' - quitting...")
                            self.is_running = False
                            break
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[KEYBOARD] Listener error: {e}")
                time.sleep(0.1)
    
    def processing_loop(self):
        """Main processing loop with overlap-add for smooth audio"""
        print("Processing loop started...")
        
        if self.debug_passthrough:
            print("DEBUG PASSTHROUGH MODE: Direct audio passthrough")
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
                    print(f"Processing loop error: {e}")
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
                                chunk = output_to_send[i:i+chunk_size]
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
                            chunk = output_to_send[i:i+chunk_size]
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
                    print(f"Processing loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)
    
    def run(self, input_device=None, output_device=None):
        """Run the processor"""
        print("\n" + "="*60)
        print("Real-time Audio Denoiser with Speaker Filtering & FAIL-SAFE")
        print("="*60)
        print(f"Speaker recognition: YOUR MODEL INTEGRATED")
        print(f"Processing: Recognize → Denoise → Filter")
        print(f"Fail-safe: {self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples")
        print(f"Current speaker: {'ALL (no filtering)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
        print(f"Denoise strength: {self.denoise_strength}")
        print(f"Speaker gain: {self.speaker_gain}x (auto-adjusted)")
        print(f"Input gain: {self.input_gain} (adjust with 'g'/'G')")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Processing window: {self.window_size} samples ({self.window_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Hop size: {self.hop_size} samples ({self.hop_size/SAMPLE_RATE*1000:.0f}ms)")
        print(f"Overlap: 50% for smooth audio")
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
                    print("   You should hear your microphone input directly")
                else:
                    print("🎤 AI Speaker Filtering System running!")
                    print(f"   Selected speaker: {'ALL (no filtering)' if self.selected_speaker_id == -1 else f'ID: {self.selected_speaker_id}'}")
                    print(f"   Fail-safe: ENABLED ({self.speaker_tracker.fail_safe_match_threshold} matches → {self.speaker_tracker.fail_safe_duration} samples)")
                    print(f"   Processing pipeline: Recognize → Denoise → Filter")
                    print(f"   Display interval: {self.display_interval} windows")
                    print(f"   Audio processing: 50% overlap-add for smooth output")
                    print(f"   Use 'n' to select next active speaker")
                    print(f"   Use 'a' for ALL speakers mode (no filtering)")
                    print(f"   Use '0'-'9' to select speaker by ID")
                    print(f"   Use 'i' to show speaker information")
                    print(f"   Use 'l' to list registered speakers")
                    print(f"   Use 'd' to toggle debug mode")
                    print(f"   Use 'f' to toggle fail-safe mode")
                    print(f"   Use 'F' to adjust fail-safe parameters")
                    print(f"   Use '+'/'-' to adjust speaker gain")
                    print(f"   Use 't' to adjust similarity threshold (currently: {self.speaker_tracker.recognizer.similarity_threshold:.2f})")
                    print(f"   Use 's' to show fail-safe statistics")
                
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
                        
                        print(f"Status: {audio_status}, "
                              f"active_speakers={active_count}, "
                              f"{selected_info}, "
                              f"{fail_safe_status}, "
                              f"threshold={self.speaker_tracker.recognizer.similarity_threshold:.2f}, "
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
            
            # Show final statistics
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total samples processed: {self.samples_processed}")
            print(f"   {self.speaker_tracker.get_fail_safe_stats()}")
            print(f"   Consecutive matches: {self.speaker_tracker.consecutive_matches}")
            print(f"   Fail-safe activations: {self.speaker_tracker.stats['fail_safe_activations']}")
            
            print(f"\nProcessor stopped.")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Real-time audio denoiser with speaker filtering and fail-safe")
    parser.add_argument("--denoise-model", default="denoiser_ckpt.pt", help="Path to denoising model")
    parser.add_argument("--vad-model", default=None, help="Path to VAD model (optional)")
    parser.add_argument("--input-device", type=int, default=None, help="Input device ID")
    parser.add_argument("--output-device", type=int, default=None, help="Output device ID")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size")
    parser.add_argument("--denoise-strength", type=float, default=0.7,
                       help="Denoising strength (0.3-0.9)")
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
    
    print(f"Configuration:")
    print(f"  Denoising model: {args.denoise_model}")
    print(f"  Speaker recognition: YOUR MODEL INTEGRATED")
    print(f"  Selection: BY SPEAKER ID")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  FAIL-SAFE: {args.fail_safe_matches} matches → {args.fail_safe_duration} samples")
    print(f"  Input device: {args.input_device or 'default'}")
    print(f"  Output device: {args.output_device or 'default'}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chunk size: {CHUNK_SIZE} samples")
    print(f"  Denoise strength: {args.denoise_strength}")
    print(f"  Input gain: {args.input_gain}")
    print(f"  Output gain: {args.output_gain}")
    print(f"  Speaker gain: {args.speaker_gain}")
    print(f"  Debug passthrough: {args.debug_passthrough}")
    
    try:
        processor = RealTimeDenoiserSpeakerFilter(
            denoise_model_path=args.denoise_model,
            vad_model_path=args.vad_model,
            denoise_strength=args.denoise_strength,
            input_gain=args.input_gain,
            output_gain=args.output_gain,
            speaker_gain=args.speaker_gain,
            debug_passthrough=args.debug_passthrough
        )
        
        # Set the similarity threshold
        processor.speaker_tracker.recognizer.similarity_threshold = args.similarity_threshold
        
        # Set fail-safe parameters
        processor.speaker_tracker.fail_safe_match_threshold = args.fail_safe_matches
        processor.speaker_tracker.fail_safe_duration = args.fail_safe_duration
        
        processor.run(args.input_device, args.output_device)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
