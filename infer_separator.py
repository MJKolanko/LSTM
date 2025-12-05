# infer_separator.py
import os
import argparse
import torch
import soundfile as sf
import numpy as np
from asteroid.models import ConvTasNet
from utils import load_wav, save_wav, SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, default="sep_model.pt", help="Ścieżka do sep_model.pt")
    parser.add_argument("--input", required=True, help="Plik .wav z mixem")
    parser.add_argument("--outdir", default="sep_output", help="Gdzie zapisać wyniki")
    parser.add_argument("--n_src", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Ładuję model:", args.model)
    model = ConvTasNet(n_src=args.n_src)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    model.eval()

    # ---------------------------
    # Wczytaj próbkę
    # ---------------------------
    wav = load_wav(args.input).astype(np.float32)
    x = torch.from_numpy(wav).unsqueeze(0).to(args.device)  # (1, T)

    # ---------------------------
    # Inference z AMP
    # ---------------------------
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            est = model(x)   # (1, n_src, T)

    est = est.squeeze(0).cpu().numpy()   # (n_src, T)

    # ---------------------------
    # Zapisz źródła
    # ---------------------------
    for i in range(args.n_src):
        out_path = os.path.join(args.outdir, f"source_{i+1}.wav")
        save_wav(out_path, est[i], SAMPLE_RATE)
        print("Zapisano:", out_path)

    print("\nDONE — możesz odsłuchać wyniki w folderze:", args.outdir)


if __name__ == "__main__":
    main()
