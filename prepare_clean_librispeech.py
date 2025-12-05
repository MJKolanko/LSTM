import os
import glob
import argparse
import random
import soundfile as sf
import librosa

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def convert_to_wav(src, dst, sr=16000):
    wav, orig_sr = sf.read(src)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr, sr)
    sf.write(dst, wav, sr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_root", default="data/clean",
                        help="folder gdzie masz LibriSpeech typu 19/198/*.flac")
    parser.add_argument("--out_dir", default="data/clean_spk",
                        help="gdzie zapisać spk1, spk2 itp")
    parser.add_argument("--n_speakers", type=int, default=3,
                        help="ile speakerów wybrać")
    args = parser.parse_args()

    # znajdź wszystkie pliki flac
    flacs = glob.glob(os.path.join(args.clean_root, "**", "*.flac"), recursive=True)
    if not flacs:
        raise RuntimeError("Nie znaleziono żadnych .flac w {}".format(args.clean_root))

    # speaker = pierwszy numer w nazwie np. 19-198-0000.flac → 19
    speakers = {}
    for f in flacs:
        base = os.path.basename(f)
        spk = base.split("-")[0]   # pierwszy numer
        speakers.setdefault(spk, []).append(f)

    print(f"Znaleziono {len(speakers)} speakerów.")
    spk_list = sorted(speakers.keys())

    # wybieramy speakerów
    chosen = random.sample(spk_list, args.n_speakers)
    print("Wybrani speakerzy:", chosen)

    # konwersja do spkX
    for i, spk in enumerate(chosen, 1):
        out_spk = os.path.join(args.out_dir, f"spk{i}")
        ensure_dir(out_spk)

        files = speakers[spk]
        print(f"Konwertuję {len(files)} plików z speakera {spk} → {out_spk}")

        for f in files:
            base = os.path.splitext(os.path.basename(f))[0] + ".wav"
            out_file = os.path.join(out_spk, base)
            convert_to_wav(f, out_file)

    print("Gotowe. Wygenerowano strukturę:")
    print(f"{args.out_dir}/spk1, spk2, ...")

if __name__ == "__main__":
    main()
