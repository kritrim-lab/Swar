"""Train the vocoder on audio alone. Re-running the same command resumes from --ckpt.

    python train_vocoder.py --data /folder/with/wavs --ckpt /path/to/vocoder.pt
"""
import argparse
import math
import os
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from swar.checkpoint import connect, load, pull, save
from swar.data import HOP, MEL_STD, SR, wav_to_mel
from swar.vocoder import Discriminators, Generator, discriminator_loss, generator_losses

SEGMENT = 64 * HOP  # about 0.74 s


class Segments(torch.utils.data.Dataset):
    """One random slice from each wav file under a folder."""

    def __init__(self, folder):
        # "._name.wav" files are macOS metadata, not audio
        self.paths = sorted(p for p in Path(folder).rglob("*.wav") if not p.name.startswith("._"))
        if not self.paths:
            raise ValueError(f"no wav files under {folder}")
        print(f"{len(self.paths)} audio files")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = str(self.paths[i])
        info = sf.info(path)
        wanted = math.ceil(SEGMENT * info.samplerate / SR)
        start = random.randint(0, max(0, info.frames - wanted))
        wav, sr = sf.read(path, start=start, frames=wanted, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return torch.nn.functional.pad(wav, (0, max(0, SEGMENT - len(wav))))[:SEGMENT]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="folder searched recursively for .wav files")
    parser.add_argument("--ckpt", required=True, help="checkpoint file to resume from and save to")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fp32", action="store_true", help="disable mixed precision if training is unstable")
    args = parser.parse_args(argv)

    torch.manual_seed(1234)
    random.seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bucket = connect()
    if bucket is not None:
        pull(bucket, args.ckpt)
    dataset = Segments(args.data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device == "cuda", drop_last=len(dataset) >= args.batch_size,
        persistent_workers=args.workers > 0,
    )
    # shortcut: one GPU and a constant learning rate, add multi-GPU / decay if training stalls
    amp = device == "cuda" and not args.fp32
    generator, discriminators = Generator().to(device), Discriminators().to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discriminators.parameters(), lr=args.lr, betas=(0.8, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)  # one scaler serves both optimizers
    parts = dict(generator=generator, discriminators=discriminators, opt_g=opt_g, opt_d=opt_d, scaler=scaler)

    step = 0
    if os.path.exists(args.ckpt):
        step = load(args.ckpt, device, **parts)
        torch.manual_seed(1234 + step)
        random.seed(1234 + step)
        print(f"resumed from {args.ckpt} at step {step}")
    print(f"device={device} generator={sum(p.numel() for p in generator.parameters()) / 1e6:.1f}M")

    while step < args.steps:
        for real in loader:
            real = real.to(device, non_blocking=True)
            mel = wav_to_mel(real)
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                fake = generator(mel)
                d_loss = discriminator_loss(discriminators(real), discriminators(fake.detach()))
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            discriminators.requires_grad_(False)
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    real_out = discriminators(real)
                adversarial, matching = generator_losses(real_out, discriminators(fake))
            mel_loss = (wav_to_mel(fake.float()) - mel).abs().mean() * MEL_STD  # fp32, log-mel units
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(45 * mel_loss + adversarial + 2 * matching).backward()
            scaler.step(opt_g)
            scaler.update()
            discriminators.requires_grad_(True)

            step += 1
            if step % args.log_every == 0:
                print(f"step {step} mel={mel_loss.item():.3f} adv={adversarial.item():.2f} "
                      f"match={matching.item():.2f} disc={d_loss.item():.2f}", flush=True)
            if step % args.save_every == 0 and step < args.steps:
                save(args.ckpt, step, bucket, **parts)
            if step >= args.steps:
                break
    save(args.ckpt, step, bucket, final=True, **parts)
    print(f"saved {args.ckpt} at step {step}")


if __name__ == "__main__":
    main()
