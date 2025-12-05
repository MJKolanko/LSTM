# models.py
import torch
import torch.nn as nn

class STFTEncoder:
    def __init__(self, n_fft=512, hop_length=128, device="cpu"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        # create window lazily per-call so it lives on correct device
        self.window = None

    def _get_window(self):
        if self.window is None or self.window.device.type != torch.device(self.device).type:
            self.window = torch.hann_window(self.n_fft).to(self.device)
        return self.window

    def stft(self, x):
        # x: (B, T)
        window = self._get_window()
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )

    def istft(self, spec, length):
        window = self._get_window()
        # spec: complex tensor (B, F, T) or (F, T)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=length,
            return_complex=False
        )

class LSTFDenoiser(nn.Module):
    def __init__(self, n_fft=512, hidden=256, n_layers=2):
        super().__init__()
        self.n_freq = n_fft // 2 + 1   # 257

        self.lstm = nn.LSTM(
            input_size=self.n_freq,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Linear(hidden * 2, self.n_freq)
        self.act = nn.Sigmoid()

    def forward(self, mag):
        # mag: (B, T, F)
        B, T, F = mag.shape
        assert F == self.n_freq, f"Expected F={self.n_freq}, got {F}"

        x, _ = self.lstm(mag)
        x = self.out(x)
        x = self.act(x)
        return x  # (B, T, F)
