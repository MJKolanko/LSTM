import os, torch, soundfile as sf
import numpy as np
from models import STFTEncoder, LSTFDenoiser
from tqdm import tqdm

SRC_NOISY = "librimix_noisy"
SRC_CLEAN = "librimix_synth"
OUT       = "librimix_cleaned"
CKPT      = "checkpoints/lstm_denoiser_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

encoder = STFTEncoder(n_fft=512, hop_length=128, device=device)
model = LSTFDenoiser(n_fft=512, hidden=256, n_layers=2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------
# BEZPIECZNE ODSZUMIANIE
# ------------------------------------------------------------
def denoise(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    length = len(audio)

    x = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        spec = encoder.stft(x)                  # (B, F, T)
        mag  = torch.abs(spec).permute(0, 2, 1) # (B, T, F)

        mask = model(mag)                       # (B, T, F)
        mask = mask.permute(0, 2, 1)            # (B, F, T)

        phase = torch.angle(spec)
        real  = mask * torch.cos(phase)
        imag  = mask * torch.sin(phase)
        est   = torch.complex(real, imag)

        out = encoder.istft(est, length=length)

    out = out.squeeze().cpu().numpy()

    # -------------------------------------------------
    # BEZPIECZNA NORMALIZACJA (match RMS, bez clippingu)
    # -------------------------------------------------
    eps = 1e-8
    rms_in  = np.sqrt(np.mean(audio**2) + eps)
    rms_out = np.sqrt(np.mean(out**2) + eps)

    if rms_out > 0:
        out *= (rms_in / rms_out)

    # ograniczenie amplitudy
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out / peak * 0.99

    return out.astype(np.float32), sr

# ------------------------------------------------------------
# PRZETWARZANIE CAŁEGO LIBRIMIXA
# ------------------------------------------------------------
for split in ["train", "test"]:
    print(f"Processing {split}...")

    noisy_mix_dir = os.path.join(SRC_NOISY, split, "mix")
    clean_s1_dir  = os.path.join(SRC_CLEAN, split, "s1")
    clean_s2_dir  = os.path.join(SRC_CLEAN, split, "s2")

    out_mix = os.path.join(OUT, split, "mix")
    out_s1  = os.path.join(OUT, split, "s1")
    out_s2  = os.path.join(OUT, split, "s2")

    os.makedirs(out_mix, exist_ok=True)
    os.makedirs(out_s1, exist_ok=True)
    os.makedirs(out_s2, exist_ok=True)

    files = sorted(f for f in os.listdir(noisy_mix_dir) if f.endswith(".wav"))

    for fn in tqdm(files, desc=split):

        # 1) DENISE MIX
        out_audio, sr = denoise(os.path.join(noisy_mix_dir, fn))
        sf.write(os.path.join(out_mix, fn), out_audio, sr)

        # 2) KOPIA ORYGINALNYCH CZYSTYCH GŁOSÓW
        s1, _ = sf.read(os.path.join(clean_s1_dir, fn))
        s2, _ = sf.read(os.path.join(clean_s2_dir, fn))

        sf.write(os.path.join(out_s1, fn), s1, 16000)
        sf.write(os.path.join(out_s2, fn), s2, 16000)

print("DONE.")
