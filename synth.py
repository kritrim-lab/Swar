"""Speak a sentence with a trained checkpoint.

    python synth.py --ckpt acoustic.pt --vocoder vocoder.pt --text "नमस्ते" --out hello.wav
"""
import argparse

import soundfile as sf
import torch

from swar.checkpoint import load
from swar.data import SR, mel_to_wav, text_to_ids
from swar.model import AcousticModel
from swar.vocoder import Generator

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--vocoder", help="vocoder checkpoint; without it the audio sounds robotic")
parser.add_argument("--text", required=True)
parser.add_argument("--out", default="out.wav")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=32, help="decoder refinement steps; more is slower")
parser.add_argument("--temperature", type=float, default=1.0, help="variation; lower is flatter")
args = parser.parse_args()

if args.steps < 1 or args.speed <= 0:
    parser.error("--steps must be at least 1 and --speed above 0")
ids = text_to_ids(args.text)
if not ids:
    parser.error("--text is empty")
model = AcousticModel()
step = load(args.ckpt, model=model)
load(args.ckpt, ema=model)  # the averaged weights, when the checkpoint has them
mel = model.eval().infer(torch.tensor(ids), args.speed, args.steps, args.temperature)
if args.vocoder:
    generator = Generator()
    load(args.vocoder, generator=generator)
    with torch.no_grad():
        wav = generator.eval()(mel[None])[0]
else:
    print("no --vocoder given: using Griffin-Lim, which sounds robotic")
    wav = mel_to_wav(mel)
sf.write(args.out, (wav / wav.abs().max().clamp(min=1.0)).numpy(), SR)
print(f"wrote {args.out} (checkpoint step {step})")
