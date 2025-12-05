import os
import numpy as np
import torch
import itertools
import time
from asteroid.models import ConvTasNet
from dataset import MixCleanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

SAMPLE_RATE = 16000

def si_sdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-9)
    s_target = alpha * ref
    e_noise = est - s_target
    return 10 * np.log10((np.sum(s_target**2) + 1e-9) /
                         (np.sum(e_noise**2) + 1e-9))

def permute_si_sdr(est_sources, ref_sources):
    n = est_sources.shape[0]
    perms = itertools.permutations(range(n))
    best = -1e9
    for p in perms:
        score = np.mean([
            si_sdr(est_sources[i], ref_sources[p[i]])
            for i in range(n)
        ])
        best = max(best, score)
    return best

def evaluate(sep_model_path, dataset_root, n_src, device="cpu"):
    model = ConvTasNet(n_src=n_src)
    model.load_state_dict(torch.load(sep_model_path, map_location=device))
    model.to(device)
    model.eval()

    ds = MixCleanDataset(os.path.join(dataset_root, "val"))
    dl = DataLoader(ds, batch_size=1, num_workers=2)

    scores = []
    times = []      # <-- czasy inferencji

    # utwórz plik logu
    time_log_path = os.path.join(dataset_root, "eval_times.txt")
    time_log = open(time_log_path, "w", encoding="utf-8")

    with torch.no_grad():
        for idx, (mix, _, sources) in enumerate(tqdm(dl)):
            mix = mix.to(device)

            # start pomiaru czasu
            start = time.time()

            with torch.cuda.amp.autocast():
                est = model(mix)

            # koniec pomiaru
            dt = time.time() - start
            times.append(dt)

            # nazwa pliku (rekonstrukcja z datasetu)
            filename = f"{idx:05d}.wav"
            time_log.write(f"{filename}: {dt:.4f} s\n")

            est = est.cpu().numpy()[0]
            srcs = sources.numpy()[0]

            # dopasowanie długości
            L = min(est.shape[1], srcs.shape[1])
            est = est[:, :L]
            srcs = srcs[:, :L]

            score = permute_si_sdr(est, srcs)
            scores.append(score)

    # zapis średniego czasu
    mean_time = float(np.mean(times))
    time_log.write(f"\nMean inference time: {mean_time:.4f} s\n")
    time_log.close()

    print("Times saved to:", time_log_path)

    return float(np.mean(scores)), scores


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sep_model", required=True)
    parser.add_argument("--dataset", default="dataset_denoised")
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mean_score, scores = evaluate(args.sep_model, args.dataset,
                                  args.n_src, args.device)
    print("Mean SI-SDR (perm):", mean_score)
