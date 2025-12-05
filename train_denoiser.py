# train_denoiser.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MixCleanDataset
from tqdm import tqdm
import yaml

# ---- prosty UNet1D ----
class UNet1D(nn.Module):
    def __init__(self, in_chan=1, base=32):
        super().__init__()
        # enkoder
        self.enc1 = nn.Sequential(nn.Conv1d(in_chan, base, 15, padding=7), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(base, base*2, 15, stride=2, padding=7), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(base*2, base*4, 15, stride=2, padding=7), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(base*4, base*8, 15, stride=2, padding=7), nn.ReLU())
        # dekoder
        self.dec4 = nn.Sequential(nn.ConvTranspose1d(base*8, base*4, 15, stride=2, padding=7, output_padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose1d(base*8, base*2, 15, stride=2, padding=7, output_padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(base*4, base, 15, stride=2, padding=7, output_padding=1), nn.ReLU())
        self.out = nn.Conv1d(base*2, 1, 1)

    def forward(self, x):
        # x: (B, 1, T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.out(torch.cat([d2, e1], dim=1))
        return out  # (B,1,T)

def collate_fn(batch):
    mixes, clean_mixes, sources = zip(*batch)
    # stack and ensure dims
    mixes = torch.stack(mixes)  # (B, T)
    clean_mixes = torch.stack(clean_mixes)
    sources = torch.stack(sources)  # (B, n_src, T)
    # add channel dim
    mixes = mixes.unsqueeze(1)
    clean_mixes = clean_mixes.unsqueeze(1)
    return mixes, clean_mixes, sources

def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    tot = 0.0
    for mixes, clean_mixes, _ in tqdm(loader):
        mixes = mixes.to(device)
        clean_mixes = clean_mixes.to(device)
        est = model(mixes)
        clean_mixes = clean_mixes.unsqueeze(1)
        clean_mixes = clean_mixes.squeeze(-2)  # usuń niepotrzebny wymiar jeśli się pojawi
        loss = loss_fn(est, clean_mixes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot / len(loader)

def validate(model, loader, device, loss_fn):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for mixes, clean_mixes, _ in loader:
            mixes = mixes.to(device)
            clean_mixes = clean_mixes.to(device)
            est = model(mixes)
            clean_mixes = clean_mixes.unsqueeze(1)
            clean_mixes = clean_mixes.squeeze(-2)  # usuń niepotrzebny wymiar jeśli się pojawi
            loss = loss_fn(est, clean_mixes)
            tot += loss.item()
    return tot / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset", help="dataset base with train/ and val/")
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="denoiser_ckpt.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_ds = MixCleanDataset(os.path.join(args.data_dir, "train"))
    val_ds = MixCleanDataset(os.path.join(args.data_dir, "val"))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate_fn)

    model = UNet1D(in_chan=1, base=32).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    best_val = 1e9
    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, train_loader, opt, args.device, loss_fn)
        val = validate(model, val_loader, args.device, loss_fn)
        print(f"Epoch {ep}: train_loss={tr:.6f} val_loss={val:.6f}")
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), args.out)
            print("Saved best denoiser ->", args.out)

if __name__ == "__main__":
    main()
