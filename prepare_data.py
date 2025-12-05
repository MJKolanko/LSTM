import os
import random
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import librosa

RAW_DIR = "dataset/raw_clips"                    # ← Twój katalog z WAV
OUT_DIR = "dataset/separation_long"

SAMPLE_RATE = 16000
N_SAMPLES = 4000

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_audio(path):
    audio, sr = sf.read(path)

    # Normalizacja do mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resampling jeśli trzeba
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    return torch.tensor(audio, dtype=torch.float32)


def pad_to_same_length(a, b):
    L = max(len(a), len(b))
    a2 = torch.zeros(L)
    b2 = torch.zeros(L)
    a2[:len(a)] = a
    b2[:len(b)] = b
    return a2, b2


def mix_snr(s1, s2, snr_db):
    rms1 = s1.pow(2).mean().sqrt() + 1e-9
    rms2 = s2.pow(2).mean().sqrt() + 1e-9

    scale = rms1 / (10 ** (snr_db / 20) * rms2)
    s2_scaled = s2 * scale

    mix = s1 + s2_scaled
    return mix, s1, s2_scaled


def main():
    wavs = list(Path(RAW_DIR).rglob("*.wav"))
    print("Znaleziono WAV:", len(wavs))
    assert len(wavs) > 2, f"Brakuje plików WAV w {RAW_DIR}"

    for idx in range(N_SAMPLES):
        p1, p2 = random.sample(wavs, 2)

        a1 = load_audio(p1)
        a2 = load_audio(p2)

        a1, a2 = pad_to_same_length(a1, a2)

        snr = random.uniform(-5, 5)
        mix, s1, s2 = mix_snr(a1, a2, snr)

        sf.write(f"{OUT_DIR}/mix_{idx}.wav", mix.numpy(), SAMPLE_RATE)
        sf.write(f"{OUT_DIR}/s1_{idx}.wav",  s1.numpy(), SAMPLE_RATE)
        sf.write(f"{OUT_DIR}/s2_{idx}.wav",  s2.numpy(), SAMPLE_RATE)

        if idx % 50 == 0:
            print("Generated", idx)

    print("DONE.")


if __name__ == "__main__":
    main()
