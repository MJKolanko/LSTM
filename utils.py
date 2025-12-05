# utils.py
import os
import numpy as np
import soundfile as sf
import librosa
import random

SAMPLE_RATE = 16000

def load_wav(path, sr=SAMPLE_RATE):
    wav, orig_sr = sf.read(path)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(float), orig_sr, sr)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)

def save_wav(path, wav, sr=SAMPLE_RATE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav.astype(np.float32), sr)

def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)

def fix_length(wav, length):
    if len(wav) >= length:
        return wav[:length]
    else:
        return np.pad(wav, (0, length - len(wav)))

def random_choose(wav_list):
    return wav_list[random.randrange(len(wav_list))]
