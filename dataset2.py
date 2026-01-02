# dataset.py
import os
import glob
import torch
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000

class MixCleanDataset(torch.utils.data.Dataset):
    """
    Dataset expects folder structure:
    root/mix/*.wav
    root/s1/*.wav
    root/s2/*.wav
    (root/s3/*.wav optional)
    Returns:
      noisy_mix: (T,)
      clean_mix: (T,)  -> sum of clean sources
      sources: (n_src, T)
    """
    def __init__(self, root):
        self.root = root
        self.mix_files = sorted(glob.glob(os.path.join(root, "mix", "*.wav")))
        # find number of sources by inspecting folder names
        src_dirs = [d for d in os.listdir(root) if d.startswith("s")]
        self.n_src = len(src_dirs)
        self.src_files = []
        for i in range(self.n_src):
            sfns = sorted(glob.glob(os.path.join(root, f"s{i+1}", "*.wav")))
            self.src_files.append(sfns)
        assert len(self.mix_files) > 0

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_path = self.mix_files[idx]
        mix, sr = sf.read(mix_path)
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        mix = mix.astype(np.float32)

        sources = []
        for i in range(self.n_src):
            path = self.src_files[i][idx]
            s, sr = sf.read(path)
            if s.ndim > 1:
                s = s.mean(axis=1)
            sources.append(s.astype(np.float32))

        sources = np.stack(sources, axis=0)  # (n_src, T)
        clean_mix = np.sum(sources, axis=0).astype(np.float32)

        return torch.from_numpy(mix), torch.from_numpy(clean_mix), torch.from_numpy(sources)
