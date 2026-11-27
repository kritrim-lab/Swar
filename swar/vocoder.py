"""Vocoder: mel spectrogram -> waveform, written from scratch.

Generator (shipped): ConvNeXt blocks predict an STFT, an inverse STFT turns it into audio (Vocos design).
Discriminators (training only): they learn to tell real from generated audio.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .data import HOP, N_FFT, N_MELS


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, hidden, scale):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x):  # [B, dim, T]
        h = self.norm(self.depthwise(x).transpose(1, 2))
        h = self.scale * self.down(F.gelu(self.up(h)))
        return x + h.transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, dim=512, hidden=1536, layers=8):
        super().__init__()
        self.embed = nn.Conv1d(N_MELS, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.Sequential(*(ConvNeXtBlock(dim, hidden, 1 / layers) for _ in range(layers)))
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_FFT + 2)  # log magnitude and phase for each frequency
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, mel):
        """mel [B, N_MELS, T] -> wav [B, (T - 1) * HOP], the inverse of wav_to_mel's framing."""
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2)
        x = self.final_norm(self.blocks(x).transpose(1, 2))
        log_magnitude, phase = self.head(x).transpose(1, 2).float().chunk(2, dim=1)
        spectrum = torch.polar(log_magnitude.exp().clamp(max=100.0), phase)
        return torch.istft(spectrum, N_FFT, HOP, window=self.window)


def run_layers(x, layers, post):
    features = []
    for layer in layers:
        x = F.leaky_relu(layer(x), 0.1)
        features.append(x)
    x = post(x)
    features.append(x)
    return x, features


class PeriodDiscriminator(nn.Module):
    """Folds the waveform every `period` samples, to catch periodic structure such as pitch."""

    def __init__(self, period):
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024]
        layers = [nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0)) for a, b in zip(channels, channels[1:])]
        layers.append(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0)))
        self.layers = nn.ModuleList(weight_norm(layer) for layer in layers)
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, wav):  # [B, samples]
        x = F.pad(wav[:, None], (0, -wav.shape[1] % self.period), mode="reflect")
        return run_layers(x.view(len(wav), 1, -1, self.period), self.layers, self.post)


class ResolutionDiscriminator(nn.Module):
    """Looks at the magnitude spectrogram at one resolution, to catch smeared or buzzy sound."""

    def __init__(self, n_fft):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        layers = [nn.Conv2d(1, 32, (3, 9), padding=(1, 4))]
        layers += [nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4)) for _ in range(3)]
        layers.append(nn.Conv2d(32, 32, 3, padding=1))
        self.layers = nn.ModuleList(weight_norm(layer) for layer in layers)
        self.post = weight_norm(nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, wav):
        spectrum = torch.stft(wav, self.n_fft, self.n_fft // 4, window=self.window, return_complex=True)
        x = spectrum.abs().transpose(1, 2)[:, None]  # [B, 1, frames, frequencies]
        return run_layers(x, self.layers, self.post)


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.all = nn.ModuleList(
            [PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11)]
            + [ResolutionDiscriminator(n) for n in (1024, 2048, 512)]
        )

    def forward(self, wav):
        return [d(wav) for d in self.all]  # list of (score, features)


def discriminator_loss(real, fake):
    return sum(((1 - r) ** 2).mean() + (f ** 2).mean() for (r, _), (f, _) in zip(real, fake))


def generator_losses(real, fake):
    """Returns (adversarial, feature matching)."""
    adversarial = sum(((1 - score) ** 2).mean() for score, _ in fake)
    matching = sum(
        (r.detach() - f).abs().mean()
        for (_, real_features), (_, fake_features) in zip(real, fake)
        for r, f in zip(real_features, fake_features)
    )
    return adversarial, matching
