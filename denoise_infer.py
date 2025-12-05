import os
import glob
import time
import torch
from utils import load_wav, save_wav, SAMPLE_RATE
import argparse
import numpy as np
from train_denoiser import UNet1D
import shutil

def denoise_file(model, wav):
    model.eval()
    with torch.no_grad():
        wav = wav.astype(np.float32)
        x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        x = x.to(next(model.parameters()).device)

        est = model(x)
        est = est.squeeze().cpu().numpy()

    return est

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="denoiser_ckpt.pt")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="dataset_denoised")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading model:", args.model)

    model = UNet1D(in_chan=1, base=32)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)

    # plik logu
    time_log_path = os.path.join(args.out, "denoise_times.txt")
    os.makedirs(args.out, exist_ok=True)
    time_log = open(time_log_path, "w", encoding="utf-8")

    for split in ["train", "val"]:
        mix_files = sorted(glob.glob(os.path.join(args.dataset, split, "mix", "*.wav")))
        out_dir = os.path.join(args.out, split, "mix")
        os.makedirs(out_dir, exist_ok=True)

        for mf in mix_files:
            wav = load_wav(mf)

            start = time.time()
            den = denoise_file(model, wav)
            dt = time.time() - start

            # log czasu
            time_log.write(f"{split}/{os.path.basename(mf)}: {dt:.4f} s\n")

            den = den[:len(wav)]
            save_wav(os.path.join(out_dir, os.path.basename(mf)), den, SAMPLE_RATE)

        # kopiujemy źródła
        src_dirs = [d for d in os.listdir(os.path.join(args.dataset, split)) if d.startswith("s")]
        for s in src_dirs:
            src_out = os.path.join(args.out, split, s)
            os.makedirs(src_out, exist_ok=True)
            for f in sorted(glob.glob(os.path.join(args.dataset, split, s, "*.wav"))):
                shutil.copy(f, os.path.join(src_out, os.path.basename(f)))

    time_log.close()
    print("DONE! Denoised dataset saved to", args.out)
    print("Times saved to:", time_log_path)

if __name__ == "__main__":
    main()
