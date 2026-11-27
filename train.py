"""Train the acoustic model. Re-running the same command resumes from --ckpt.

    python train.py --data /path/to/manifest.txt --ckpt /path/to/acoustic.pt
"""
import argparse
import os

import torch

from swar.checkpoint import connect, load, pull, save
from swar.data import TTSDataset, collate
from swar.model import AcousticModel

DIAGONAL, DIAGONAL_STEPS = 2000.0, 1000  # alignment warm-up, fades to zero


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="manifest of audio_path|text lines")
    parser.add_argument("--ckpt", required=True, help="checkpoint file to resume from and save to")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    bucket = connect()
    if bucket is not None:
        pull(bucket, args.ckpt)
    dataset = TTSDataset(args.data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, pin_memory=amp, drop_last=len(dataset) >= args.batch_size,
        persistent_workers=args.workers > 0,
    )
    # shortcut: one GPU and no validation split, add both once the first model talks
    model = AcousticModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)  # fp16: T4 has no fast bf16
    parts = dict(model=model, optimizer=optimizer, scaler=scaler)

    step = 0
    if os.path.exists(args.ckpt):
        step = load(args.ckpt, device, **parts)
        torch.manual_seed(1234 + step)  # do not replay the same shuffle order after a resume
        print(f"resumed from {args.ckpt} at step {step}")
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    model.train()
    while step < args.steps:
        for batch in loader:
            batch = [t.to(device, non_blocking=True) for t in batch]
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                losses, _ = model(*batch, diagonal=DIAGONAL * max(0.0, 1 - step / DIAGONAL_STEPS))
                loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} " + " ".join(f"{k}={v.item():.3f}" for k, v in losses.items()), flush=True)
            if step % args.save_every == 0 and step < args.steps:
                save(args.ckpt, step, bucket, **parts)
            if step >= args.steps:
                break
    save(args.ckpt, step, bucket, final=True, **parts)
    print(f"saved {args.ckpt} at step {step}")


if __name__ == "__main__":
    main()
