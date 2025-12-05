# prepare_data.py
"""
Generuje syntetyczne mixy z data/clean/spk*/ oraz dodaje szumy z data/noise/.
Zapisuje w dataset/{train,val}/mix.wav i dataset/{train,val}/s1.wav, s2.wav, (s3.wav)
"""

import os
import glob
import random
import argparse
from tqdm import tqdm
import numpy as np

from utils import load_wav, save_wav, SAMPLE_RATE, fix_length

def collect_files(clean_dir):
    spk_dirs = sorted([os.path.join(clean_dir,d) for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir,d))])
    spk_files = [sorted(glob.glob(os.path.join(d,"*.wav"))) for d in spk_dirs]
    return spk_files

def collect_noise(noise_dir):
    if not os.path.isdir(noise_dir):
        return []
    return sorted(glob.glob(os.path.join(noise_dir,"*.wav")))

def mix_and_save(out_dir, sources, mix, idx):
    os.makedirs(out_dir, exist_ok=True)
    save_wav(os.path.join(out_dir, "mix", f"{idx}.wav"), mix)
    for i, s in enumerate(sources):
        save_wav(os.path.join(out_dir, f"s{i+1}", f"{idx}.wav"), s)

def add_noise_to_mix(mix, noises, snr_db):
    if not noises:
        return mix, None
    noise_path = random.choice(noises)
    noise = load_wav(noise_path)
    # repeat/pad noise to mix length
    if len(noise) < len(mix):
        reps = int(np.ceil(len(mix) / len(noise)))
        noise = np.tile(noise, reps)[:len(mix)]
    else:
        start = random.randint(0, len(noise) - len(mix))
        noise = noise[start:start+len(mix)]
    # scale noise to achieve SNR (mix power vs noise power)
    mix_r = np.sqrt(np.mean(mix**2) + 1e-12)
    noise_r = np.sqrt(np.mean(noise**2) + 1e-12)
    target_noise_r = mix_r / (10**(snr_db/20.0))
    noise = noise * (target_noise_r / (noise_r + 1e-12))
    return mix + noise, noise

def generate(dataset_dir, spk_files, noises, n_mixes, segment_len_sec=4, snr_db=5):
    segment_len = int(segment_len_sec * SAMPLE_RATE)
    for i in tqdm(range(n_mixes)):
        # choose one file per speaker
        chosen = [random.choice(files) for files in spk_files]
        sources = []
        for path in chosen:
            wav = load_wav(path)
            if len(wav) <= segment_len:
                # pad or repeat
                wav = fix_length(wav, segment_len)
            else:
                start = random.randint(0, len(wav)-segment_len)
                wav = wav[start:start+segment_len]
            sources.append(wav)
        mix = np.sum(sources, axis=0)
        noisy_mix, noise = add_noise_to_mix(mix, noises, snr_db)
        # save noisy mix and clean sources (targets)
        out_dir = dataset_dir
        os.makedirs(os.path.join(out_dir, "mix"), exist_ok=True)
        for k in range(len(sources)):
            os.makedirs(os.path.join(out_dir, f"s{k+1}"), exist_ok=True)
        save_wav(os.path.join(out_dir, "mix", f"{i}.wav"), noisy_mix)
        for k, s in enumerate(sources):
            save_wav(os.path.join(out_dir, f"s{k+1}", f"{i}.wav"), s)
    print("Done", dataset_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", default="data/clean_spk", help="folder with spk1, spk2, ...")
    parser.add_argument("--noise_dir", default="data/noise", help="noise wavs (optional)")
    parser.add_argument("--out_dir", default="dataset", help="output dataset base")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--segment_len", type=float, default=4.0)
    parser.add_argument("--snr_db", type=float, default=5.0)
    args = parser.parse_args()

    spk_files = collect_files(args.clean_dir)
    if len(spk_files) < 2:
        raise RuntimeError("Potrzebne co najmniej 2 katalogi spk (spk1, spk2).")
    noises = collect_noise(args.noise_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    # train
    generate(os.path.join(args.out_dir, "train"), spk_files, noises, args.n_train, args.segment_len, args.snr_db)
    # val
    generate(os.path.join(args.out_dir, "val"), spk_files, noises, args.n_val, args.segment_len, args.snr_db)

if __name__ == "__main__":
    main()
