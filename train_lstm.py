# train_lstm.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
from models import STFTEncoder, LSTFDenoiser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Simple metrics (lightweight implementations)
def si_sdr_batch(est, ref, eps=1e-8):
    # est/ref shape: (B, T)
    B = est.shape[0]
    out = []
    for i in range(B):
        e = est[i]
        r = ref[i]
        # zero-mean
        e = e - e.mean()
        r = r - r.mean()
        ref_energy = np.sum(r**2) + eps
        scale = np.sum(e * r) / ref_energy
        proj = scale * r
        noise = e - proj
        sdr = 10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps))
        out.append(sdr)
    return float(np.mean(out))

# Dataset: noisy -> clean
class STFTMixDataset(Dataset):
    def __init__(self, root_dir, sr=16000):
        """
        root_dir structure:
            root_dir/mix/*.wav
            root_dir/clean/*.wav
        """
        self.mix_dir   = os.path.join(root_dir, "mix")
        self.clean_dir = os.path.join(root_dir, "clean")

        self.files = sorted([f for f in os.listdir(self.mix_dir) if f.endswith(".wav")])
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]

        mix, _ = sf.read(os.path.join(self.mix_dir, fn))
        clean, _ = sf.read(os.path.join(self.clean_dir, fn))

        if mix.ndim > 1: mix = mix.mean(axis=1)
        if clean.ndim > 1: clean = clean.mean(axis=1)

        # pad to same length
        L = max(len(mix), len(clean))
        if len(mix) < L:
            mix = np.pad(mix, (0, L - len(mix)))
        if len(clean) < L:
            clean = np.pad(clean, (0, L - len(clean)))

        return mix.astype(np.float32), clean.astype(np.float32)

# Collate with random chunking to fixed segment length to stabilize training
MAX_SEG_SECONDS = 4  # use 4s chunks
MAX_SEG_SAMPLES = MAX_SEG_SECONDS * 16000

def collate_pad(batch):
    mixes = [x[0] for x in batch]
    refs = [x[1] for x in batch]

    B = len(mixes)

    # randomly crop or pad to MAX_SEG_SAMPLES
    segs_mix = torch.zeros(B, MAX_SEG_SAMPLES, dtype=torch.float32)
    segs_ref = torch.zeros(B, MAX_SEG_SAMPLES, dtype=torch.float32)

    for i, (m, r) in enumerate(zip(mixes, refs)):
        L = m.shape[0]
        if L > MAX_SEG_SAMPLES:
            start = np.random.randint(0, L - MAX_SEG_SAMPLES + 1)
            seg_m = m[start:start + MAX_SEG_SAMPLES]
            seg_r = r[start:start + MAX_SEG_SAMPLES]
        else:
            seg_m = np.pad(m, (0, MAX_SEG_SAMPLES - L))
            seg_r = np.pad(r, (0, MAX_SEG_SAMPLES - L))
        segs_mix[i] = torch.from_numpy(seg_m)
        segs_ref[i] = torch.from_numpy(seg_r)

    return segs_mix, segs_ref

# SI-SDR loss (torch)
def si_sdr_loss(est, ref, eps=1e-8):
    # est/ref: (B, T)
    B = est.shape[0]
    ref_energy = torch.sum(ref ** 2, dim=1, keepdim=True) + eps
    scale = torch.sum(est * ref, dim=1, keepdim=True) / ref_energy
    proj = scale * ref
    noise = est - proj
    sdr = 10 * torch.log10((torch.sum(proj ** 2, dim=1) + eps) / (torch.sum(noise ** 2, dim=1) + eps))
    return -sdr.mean()

def train(root_dir, epochs=30, batch_size=8, device="cpu", logdir="runs/denoiser"):
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "test")

    train_ds = STFTMixDataset(train_dir)
    val_ds   = STFTMixDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_pad, num_workers=1, pin_memory=True)

    encoder = STFTEncoder(n_fft=512, hop_length=128, device=device)
    model   = LSTFDenoiser(n_fft=512, hidden=256, n_layers=2).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    writer = SummaryWriter(logdir)
    os.makedirs("checkpoints", exist_ok=True)

    best_val = float("inf")
    print("Device:", device)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep} [train]")
        train_loss = 0.0

        for mixes, refs in pbar:
            mixes = mixes.to(device)
            refs  = refs.to(device)

            spec = encoder.stft(mixes)          # (B, F, T) complex
            mag  = torch.abs(spec)              # (B, F, T)
            mag_for_model = mag.permute(0, 2, 1)  # (B, T, F)

            mask = model(mag_for_model)         # (B, T, F)
            mask = mask.permute(0, 2, 1)        # (B, F, T)

            # IMPORTANT: apply mask to magnitude (fixed)
            rec_mag = mask * mag

            phase = torch.angle(spec)
            real = rec_mag * torch.cos(phase)
            imag = rec_mag * torch.sin(phase)

            masked_complex = torch.complex(real, imag)  # (B, F, T)
            rec = encoder.istft(masked_complex, length=mixes.shape[1])

            loss = si_sdr_loss(rec, refs)
            opt.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=train_loss / (pbar.n + 1e-9))

        avg_train = train_loss / len(train_loader)
        print(f"Epoch {ep} avg train loss = {avg_train:.4f}")
        writer.add_scalar("Loss/train", avg_train, ep)

        # validation
        model.eval()
        val_loss = 0.0
        si_sdr_vals = []

        with torch.no_grad():
            for mixes, refs in val_loader:
                mixes = mixes.to(device)
                refs  = refs.to(device)

                spec = encoder.stft(mixes)
                mag  = torch.abs(spec)
                mag_for_model = mag.permute(0, 2, 1)

                mask = model(mag_for_model)
                mask = mask.permute(0, 2, 1)

                rec_mag = mask * mag
                phase = torch.angle(spec)
                real = rec_mag * torch.cos(phase)
                imag = rec_mag * torch.sin(phase)
                masked_complex = torch.complex(real, imag)
                rec = encoder.istft(masked_complex, length=mixes.shape[1])

                loss = si_sdr_loss(rec, refs)
                val_loss += loss.item()

                rec_np = rec.cpu().numpy()
                refs_np = refs.cpu().numpy()
                s = si_sdr_batch(rec_np, refs_np)
                si_sdr_vals.append(s)

        avg_val = val_loss / len(val_loader)
        avg_si_sdr = float(np.mean(si_sdr_vals))
        print(f"Epoch {ep} VAL → loss={avg_val:.4f}, SI-SDR={avg_si_sdr:.3f}")

        writer.add_scalar("Loss/val", avg_val, ep)
        writer.add_scalar("Metric/SI-SDR", avg_si_sdr, ep)

        torch.save(model.state_dict(), f"checkpoints/lstm_denoiser_ep{ep}.pth")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "checkpoints/lstm_denoiser_best.pth")
            print("✓ Saved NEW BEST MODEL!")

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(root_dir="./librimix_noisy", epochs=30, batch_size=8, device=device, logdir="runs/denoiser")
