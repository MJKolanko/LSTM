# prepare_noisy_data.py
import os, random
import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa

RAW_ROOT = "librimix_synth"   # expects existing librimix_synth with train/test/mix,s1,s2
OUT_ROOT = "librimix_noisy"
NOISE_DIR = "background_noises"  # optional folder if you have noises; otherwise will add gaussian
SAMPLE_RATE = 16000

os.makedirs(OUT_ROOT, exist_ok=True)
for split in ["train", "test"]:
    for sub in ["mix", "clean"]:
        os.makedirs(os.path.join(OUT_ROOT, split, sub), exist_ok=True)

def list_wavs(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")])

# optional: load background noises if provided
noises = []
if os.path.exists(NOISE_DIR):
    for n in list_wavs(NOISE_DIR):
        audio, sr = librosa.load(n, sr=SAMPLE_RATE, mono=True)
        noises.append((audio, SAMPLE_RATE))

def add_noise(clean, sr=16000):
    # choose between real noise or gaussian
    if noises:
        n, nsr = random.choice(noises)
        # n is already resampled to SAMPLE_RATE above
        if len(n) < len(clean):
            n = np.tile(n, int(np.ceil(len(clean)/len(n))))
        noise_seg = n[:len(clean)]
        snr_db = random.uniform(-5, 5)
        rms = lambda x: np.sqrt(np.mean(x**2)+1e-9)
        r_clean = rms(clean); r_noise = rms(noise_seg)
        scale = (r_clean / (r_noise+1e-9)) / (10**(snr_db/20.0))
        noise_seg = noise_seg * scale
        noisy = clean + noise_seg
    else:
        # gaussian noise with random SNR
        snr_db = random.uniform(-5, 5)
        rms = lambda x: np.sqrt(np.mean(x**2)+1e-9)
        r_clean = rms(clean)
        sigma = r_clean / (10**(snr_db/20.0))
        noise = np.random.normal(0, sigma, size=clean.shape)
        noisy = clean + noise

    # normalize each signal independently to avoid changing SNR between samples
    peak_noisy = max(1.0, np.max(np.abs(noisy)))
    noisy = noisy / peak_noisy
    peak_clean = max(1.0, np.max(np.abs(clean)))
    clean = clean / peak_clean

    return noisy.astype(np.float32), clean.astype(np.float32)


for split in ["train", "test"]:
    mix_dir = os.path.join(RAW_ROOT, split, "mix")
    s1_dir  = os.path.join(RAW_ROOT, split, "s1")
    s2_dir  = os.path.join(RAW_ROOT, split, "s2")
    files = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])
    pbar = tqdm(files, desc=f"Preparing {split}")
    for idx, fn in enumerate(pbar):
        s1, _ = sf.read(os.path.join(s1_dir, fn))
        s2, _ = sf.read(os.path.join(s2_dir, fn))
        # mono
        if s1.ndim > 1: s1 = s1.mean(axis=1)
        if s2.ndim > 1: s2 = s2.mean(axis=1)

        # pad to same length
        L = max(len(s1), len(s2))
        if len(s1) < L: s1 = np.pad(s1, (0, L-len(s1)))
        if len(s2) < L: s2 = np.pad(s2, (0, L-len(s2)))
        clean = (s1 + s2).astype(np.float32)

        noisy, clean_n = add_noise(clean, sr=SAMPLE_RATE)
        out_noisy = os.path.join(OUT_ROOT, split, "mix", fn)
        out_clean = os.path.join(OUT_ROOT, split, "clean", fn)
        sf.write(out_noisy, noisy, SAMPLE_RATE)
        sf.write(out_clean, clean_n, SAMPLE_RATE)
