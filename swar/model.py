"""Acoustic model: Nepali characters -> mel spectrogram, written from scratch.

text -> encoder -> one predicted mel frame per character (mu)
training:  monotonic alignment search matches characters to real frames
inference: duration predictor says how many frames each character gets
then:      stretch to frame length -> flow-matching decoder turns noise into a mel

Flow matching (as in Matcha-TTS) learns the direction from random noise to a real mel. Sampling
follows it in a few steps, which keeps detail that a plain regression decoder averages away.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .data import N_MELS, SYMBOLS

SIGMA_MIN = 1e-4  # noise left at the end of the flow


class Block(nn.Module):
    """Self-attention then a convolutional feed-forward. The convolutions carry word order."""

    def __init__(self, dim, heads, kernel=5, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim * 4, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(dim * 4, dim, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x [B, T, D], mask [B, T] True on real positions
        keep = mask.unsqueeze(-1)
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)[0])
        h = (self.norm2(x) * keep).transpose(1, 2)
        h = self.drop(F.gelu(self.conv1(h))) * keep.transpose(1, 2)
        x = x + self.drop(self.conv2(h).transpose(1, 2))
        return x * keep


class DurationPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, 3, padding=1) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(2))
        self.out = nn.Linear(dim, 1)

    def forward(self, x, mask):
        keep = mask.unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv((x * keep).transpose(1, 2)).transpose(1, 2)))
        return self.out(x).squeeze(-1) * mask  # log frames per character


class FlowDecoder(nn.Module):
    """Predicts the velocity that moves a noisy mel toward a real one at time t (0 noise, 1 mel)."""

    def __init__(self, dim, heads, layers):
        super().__init__()
        self.input = nn.Linear(N_MELS + dim, dim)
        self.time = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.output = nn.Linear(dim, N_MELS)

    def forward(self, noisy, t, condition, mask):  # noisy [B, T, N_MELS], t [B], condition [B, T, D]
        half = condition.shape[-1] // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        angles = 1000 * t[:, None] * frequencies
        time = self.time(torch.cat([angles.sin(), angles.cos()], dim=-1))[:, None]
        x = self.input(torch.cat([noisy, condition], dim=-1))
        for block in self.blocks:
            x = block(x + time, mask)
        return self.output(x) * mask.unsqueeze(-1)


def lengths_to_mask(lengths, size):
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


@torch.no_grad()
def monotonic_alignment_search(score, text_lens, mel_lens):
    """Best left-to-right path where every character gets at least one frame.

    score [B, text, mel] (higher is better). Returns a 0/1 tensor of the same shape.
    Needs mel_lens >= text_lens. Runs on CPU: a short Python loop is quicker there.
    """
    device = score.device
    score, text_lens, mel_lens = score.float().cpu(), text_lens.cpu(), mel_lens.cpu()
    batch, n_text, n_mel = score.shape
    best = torch.full((batch, n_text), float("-inf"))
    best[:, 0] = score[:, 0, 0]
    advanced = torch.zeros(batch, n_text, n_mel, dtype=torch.bool)
    for t in range(1, n_mel):
        previous = F.pad(best[:, :-1], (1, 0), value=float("-inf"))  # came from the character before
        advanced[:, :, t] = previous > best
        best = torch.maximum(best, previous) + score[:, :, t]

    path = torch.zeros_like(score)
    rows = torch.arange(batch)
    char = text_lens - 1
    for t in range(n_mel - 1, -1, -1):
        active = t < mel_lens
        path[rows[active], char[active], t] = 1.0
        char = char - (advanced[rows, char, t] & active).long()
    return path.to(device)


class AcousticModel(nn.Module):
    def __init__(self, dim=192, heads=2, encoder_layers=4, decoder_layers=6):
        super().__init__()
        self.embed = nn.Embedding(len(SYMBOLS), dim, padding_idx=0)
        self.encoder = nn.ModuleList(Block(dim, heads) for _ in range(encoder_layers))
        self.to_mu = nn.Linear(dim, N_MELS)
        self.duration = DurationPredictor(dim)
        # shortcut: attends over whole clips (up to 15 s); on out-of-memory lower --batch-size,
        # and if that is not enough train the decoder on random 2-4 s crops
        self.decoder = FlowDecoder(dim, heads, decoder_layers)

    def encode(self, ids, mask):
        x = self.embed(ids)
        for block in self.encoder:
            x = block(x, mask)
        # mu reads the whole sentence: in real speech a letter's sound depends on its neighbours.
        # On tiny data this can memorise a wrong alignment, which the diagonal warm-up counters.
        return x, self.to_mu(x) * mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(self, condition, mask, steps=32, temperature=1.0):
        """Euler integration from noise (t=0) to mel (t=1). condition [B, T, D] -> [B, T, N_MELS]."""
        x = torch.randn(*condition.shape[:2], N_MELS, device=condition.device) * temperature
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device)
            x = x + self.decoder(x, t, condition, mask) / steps
        return x * mask.unsqueeze(-1)

    def forward(self, ids, id_lens, mels, mel_lens, diagonal=0.0):
        """mels [B, N_MELS, T]. Returns (losses dict, alignment [B, text, T]).

        diagonal > 0 nudges the alignment toward an even left-to-right reading; use it early on.
        """
        text_mask = lengths_to_mask(id_lens, ids.shape[1])
        mel_mask = lengths_to_mask(mel_lens, mels.shape[2])
        target = mels.transpose(1, 2)
        x, mu = self.encode(ids, text_mask)

        # log-likelihood of each real frame under a unit Gaussian centred on each character's mu
        # autocast off: mixed precision would run this matmul in fp16 and blur the alignment
        with torch.no_grad(), torch.autocast(mu.device.type, enabled=False):
            m, y = mu.float(), target.float()
            score = (m @ y.transpose(1, 2)) - 0.5 * (m.pow(2).sum(-1)[:, :, None] + y.pow(2).sum(-1)[:, None, :])
        if diagonal > 0:
            char_at = (torch.arange(ids.shape[1], device=ids.device)[None, :, None] + 0.5) / id_lens[:, None, None]
            time_at = (torch.arange(mels.shape[2], device=ids.device)[None, None, :] + 0.5) / mel_lens[:, None, None]
            score = score - diagonal * (char_at - time_at) ** 2
        align = monotonic_alignment_search(score, id_lens, mel_lens)

        def mel_average(error):
            return (error * mel_mask.unsqueeze(-1)).sum() / (mel_mask.sum() * N_MELS)

        stretch = align.transpose(1, 2)  # [B, T, text]
        prior_loss = mel_average((stretch @ mu - target) ** 2)

        # flow matching: pick a point on the straight line from noise to the real mel, predict its direction
        noise = torch.randn_like(target)
        t = torch.rand(len(target), device=target.device)
        at = t[:, None, None]
        noisy = (1 - (1 - SIGMA_MIN) * at) * noise + at * target
        velocity = self.decoder(noisy, t, stretch @ x, mel_mask)
        flow_loss = mel_average((velocity.float() - (target - (1 - SIGMA_MIN) * noise)) ** 2)

        log_frames = torch.log(align.sum(-1).clamp(min=1))
        predicted = self.duration(x.detach(), text_mask)  # detach: must not bend the alignment
        duration_loss = (((predicted - log_frames) ** 2) * text_mask).sum() / text_mask.sum()
        return {"flow": flow_loss, "prior": prior_loss, "duration": duration_loss}, align

    @torch.no_grad()
    def infer(self, ids, speed=1.0, steps=32, temperature=1.0):
        """ids [text] -> mel [N_MELS, T] for one sentence. Call .eval() first."""
        ids = ids[None]
        mask = torch.ones_like(ids, dtype=torch.bool)
        x, _ = self.encode(ids, mask)
        # cap at 100 frames (1.2 s) per character so a bad checkpoint cannot exhaust memory
        frames = (self.duration(x, mask).exp() / speed).round().long().clamp(1, 100)[0]
        x = x[0].repeat_interleave(frames, dim=0)[None]
        frame_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return self.sample(x, frame_mask, steps, temperature)[0].T
