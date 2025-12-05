import os
import tarfile
import urllib.request
import zipfile

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"

OUT_DIR = "data"

def download(url, dest):
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print("Done.")

def extract_tar(path, out_dir):
    print(f"Extracting {path}...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(out_dir)
    print("Done.")

def extract_zip(path, out_dir):
    print(f"Extracting {path}...")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(out_dir)
    print("Done.")

def reorganize_librispeech():
    print("Reorganizing LibriSpeech → data/clean...")
    clean_out = os.path.join(OUT_DIR, "clean")
    os.makedirs(clean_out, exist_ok=True)

    root = os.path.join(OUT_DIR, "LibriSpeech", "train-clean-100")
    for speaker in os.listdir(root):
        spk_dir = os.path.join(root, speaker)
        if not os.path.isdir(spk_dir):
            continue

        dest = os.path.join(clean_out, speaker)
        os.makedirs(dest, exist_ok=True)

        for chapter in os.listdir(spk_dir):
            chapter_dir = os.path.join(spk_dir, chapter)
            for f in os.listdir(chapter_dir):
                if f.endswith(".wav"):
                    src = os.path.join(chapter_dir, f)
                    dst = os.path.join(dest, f)
                    os.rename(src, dst)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    librispeech_tar = os.path.join(OUT_DIR, "train-clean-100.tar.gz")

    download(LIBRISPEECH_URL, librispeech_tar)

    extract_tar(librispeech_tar, OUT_DIR)

    reorganize_librispeech()

    print("All datasets downloaded and prepared.")

if __name__ == "__main__":
    main()
