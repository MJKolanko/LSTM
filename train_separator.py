import os
import argparse
import torch
from torch.utils.data import DataLoader
from asteroid.models import ConvTasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from dataset import MixCleanDataset
from tqdm import tqdm


def collate_for_sep(batch):
    mixes, _, sources = zip(*batch)
    mixes = torch.stack(mixes)          # (B, T)
    sources = torch.stack(sources)      # (B, n_src, T)
    return mixes, sources


def train_epoch(model, loader, optimizer, device, loss_fn, scaler):
    model.train()
    total = 0.0

    for mixes, sources in tqdm(loader):
        mixes = mixes.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # AMP – mixed precision
        with torch.cuda.amp.autocast():
            est = model(mixes)
            loss = loss_fn(est, sources)

        # backward (AMP)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()

    return total / len(loader)


def validate(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for mixes, sources in loader:
            mixes = mixes.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                est = model(mixes)
                loss = loss_fn(est, sources)

            total += loss.item()

    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--n_src", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="sep_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # datasets
    train_ds = MixCleanDataset(os.path.join(args.dataset, "train"))
    val_ds = MixCleanDataset(os.path.join(args.dataset, "val"))

    # SZYBSZY DATALOADER
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_for_sep,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, collate_fn=collate_for_sep,
        num_workers=2, pin_memory=True
    )

    device = args.device
    model = ConvTasNet(n_src=args.n_src).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = PITLossWrapper(pairwise_neg_sisdr)

    scaler = torch.cuda.amp.GradScaler()   # AMP ←←

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
        val = validate(model, val_loader, device, loss_fn)

        print(f"Epoch {ep}: train={tr:.6f} | val={val:.6f}")

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), args.out)
            print(">>> Saved separator ->", args.out)


if __name__ == "__main__":
    main()
