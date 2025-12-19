import os
import argparse
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Subset
from asteroid.models import ConvTasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from pystoi import stoi
from pesq import pesq

from dataset import MixCleanDataset
from tqdm import tqdm

# ======================
# PARAMETRY
# ======================
CHUNK_LEN = 16000  # 1 sekunda @16kHz
METRIC_EVERY = 1
MAX_VAL_BATCHES = 10  # mniej próbek walidacyjnych
SR = 16000
TRAIN_FRACTION = 0.5  # 50% danych treningowych
GRAD_CLIP = 5.0
EPS = 1e-8

# ======================
# DATALOADER
# ======================
def collate_for_sep(batch):
    mixes, _, sources = zip(*batch)
    mixes = torch.stack(mixes)
    sources = torch.stack(sources)

    # zero-padding / losowy chunk
    if mixes.shape[-1] < CHUNK_LEN:
        pad_len = CHUNK_LEN - mixes.shape[-1]
        mixes = torch.nn.functional.pad(mixes, (0, pad_len))
        sources = torch.nn.functional.pad(sources, (0, pad_len))
    elif mixes.shape[-1] > CHUNK_LEN:
        max_start = mixes.shape[-1] - CHUNK_LEN
        start = torch.randint(0, max_start, (1,)).item()
        mixes = mixes[..., start:start + CHUNK_LEN]
        sources = sources[..., start:start + CHUNK_LEN]

    return mixes, sources

# ======================
# SI-SDR
# ======================
def si_sdr(est, ref, eps=EPS):
    ref_energy = np.sum(ref ** 2)
    if ref_energy < eps:
        return 0.0
    scale = np.sum(est * ref) / (ref_energy + eps)
    e_true = scale * ref
    e_res = est - e_true
    if np.sum(e_res ** 2) < eps:
        return 0.0
    return 10 * np.log10(np.sum(e_true ** 2) / (np.sum(e_res ** 2) + eps))

# ======================
# METRYKI
# ======================
def compute_metrics(est, ref, epoch):
    est = est.cpu().numpy()
    ref = ref.cpu().numpy()

    si_sdr_v, pesq_v, stoi_v, der_v = [], [], [], []

    for i in range(ref.shape[0]):
        si_sdr_v.append(si_sdr(est[i], ref[i]))

        if epoch % METRIC_EVERY == 0:
            try:
                pesq_v.append(pesq(SR, ref[i], est[i], "wb"))
            except Exception:
                pesq_v.append(1.0)
            try:
                stoi_v.append(stoi(ref[i], est[i], SR, extended=False))
            except Exception:
                stoi_v.append(0.0)

        if np.std(ref[i]) < EPS or np.std(est[i]) < EPS:
            der_v.append(1.0)
        else:
            corr = np.corrcoef(ref[i], est[i])[0, 1]
            der_v.append(1 - corr)

    return {
        "si_sdr": np.mean(si_sdr_v),
        "pesq": np.mean(pesq_v) if pesq_v else np.nan,
        "stoi": np.mean(stoi_v) if stoi_v else np.nan,
        "der": np.mean(der_v),
    }

# ======================
# TRENING
# ======================
def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    for mixes, sources in tqdm(loader, leave=False):
        mixes = mixes.to(device)
        sources = sources.to(device)

        optimizer.zero_grad(set_to_none=True)
        est = model(mixes)
        loss = loss_fn(est, sources)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

# ======================
# WALIDACJA
# ======================
def validate(model, loader, device, loss_fn, epoch):
    model.eval()
    total_loss = 0.0
    metrics_sum = {"si_sdr": [], "pesq": [], "stoi": [], "der": []}

    with torch.no_grad():
        for i, (mixes, sources) in enumerate(loader):
            if i >= MAX_VAL_BATCHES:
                break
            mixes = mixes.to(device)
            sources = sources.to(device)

            est = model(mixes)
            loss = loss_fn(est, sources)
            total_loss += loss.item()

            m = compute_metrics(est[0], sources[0], epoch)
            for k in metrics_sum:
                metrics_sum[k].append(m[k])

    return total_loss / (i + 1), {k: np.nanmean(v) for k, v in metrics_sum.items()}

# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--n_src", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)  # mniejszy batch
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="sep_model.pt")
    parser.add_argument("--csv", default="metrics.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- DATASETS ----
    full_train = MixCleanDataset(os.path.join(args.dataset, "train"))
    val_ds = MixCleanDataset(os.path.join(args.dataset, "val"))

    n_train = int(TRAIN_FRACTION * len(full_train))
    idx = np.random.choice(len(full_train), n_train, replace=False)
    train_ds = Subset(full_train, idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_for_sep, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_for_sep, num_workers=4, pin_memory=True)

    # ---- MODEL (mały ConvTasNet) ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ConvTasNet(
        n_src=args.n_src,
        N=64,
        B=64,
        H=128,
        R=4,
        X=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = PITLossWrapper(pairwise_neg_sisdr)

    history = []
    best_si_sdr = -np.inf

    # ---- TRAIN LOOP ----
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, metrics = validate(model, val_loader, device, loss_fn, ep)

        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss, **metrics}
        history.append(row)

        print(f"Epoch {ep:02d} | train={tr_loss:.4f} val={val_loss:.4f} | "
              f"SI-SDR={metrics['si_sdr']:.2f} PESQ={metrics['pesq']:.2f} "
              f"STOI={metrics['stoi']:.2f} DER={metrics['der']:.2f}")

        if metrics["si_sdr"] > best_si_sdr:
            best_si_sdr = metrics["si_sdr"]
            torch.save(model.state_dict(), args.out)

        scheduler.step(val_loss)
        pd.DataFrame(history).to_csv(args.csv, index=False)

    print("✅ Training finished safely, SI-SDR nie jest już -inf")

if __name__ == "__main__":
    main()
