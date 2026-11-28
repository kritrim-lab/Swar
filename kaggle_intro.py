# Paste this whole file into one Kaggle notebook cell. GPU T4, Internet On, no secrets. Safe to re-run.
# Attach the output of the finished training notebook (Add Input) so both checkpoints are available.
import glob, json, os, subprocess, sys
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

SENTENCES = [   # edit freely: Devanagari only
    "नमस्ते!",
    "मेरो नाम स्वर हो।",
    "म नेपाली बोल्ने एउटा कृत्रिम आवाज हुँ।",
    "मलाई कृत्रिम ल्याबले सुरुदेखि नै बनाएको हो।",
    "तपाईंले लेखेका कुरा म बोलेर सुनाउन सक्छु।",
]
PAUSE = 0.35   # seconds of silence between sentences
code = "/tmp/Swar"

def run(*command, stdin=None):
    done = subprocess.run(command, input=stdin, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr[-3000:])
    return done.stdout.strip()

def furthest(name):   # torch stays out of the notebook kernel, so each step is read in a subprocess
    found = [p for root in ("/kaggle/input", "/kaggle/working")
             for p in glob.glob(f"{root}/**/{name}", recursive=True)]
    assert found, f"{name} not found: attach the training notebook's output"
    read = "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])"
    steps = {p: int(run(sys.executable, "-c", read, p)) for p in found}
    best = max(steps, key=steps.get)
    print(f"{name}: step {steps[best]} from {best}")
    return best

if os.path.isdir(code):
    run("git", "-C", code, "pull")
else:
    run("git", "clone", "https://github.com/kritrim-lab/Swar.git", code)
assert '"ema"' in open(f"{code}/swar/checkpoint.py", encoding="utf-8").read(), "old code: push first"
acoustic, vocoder = furthest("acoustic_v2.pt"), furthest("vocoder.pt")

# one process loads both models once, on the GPU when there is one, and speaks every sentence
SPEAK = """
import json, sys
sys.path.insert(0, sys.argv[1])
import soundfile as sf
import torch
from swar.checkpoint import load
from swar.data import SR, text_to_ids
from swar.model import AcousticModel
from swar.vocoder import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, generator = AcousticModel(), Generator()
load(sys.argv[2], model=model)
load(sys.argv[2], ema=model)
load(sys.argv[3], generator=generator)
model.eval().to(device)
generator.eval().to(device)
print("spoke on", device)
for number, text in enumerate(json.load(sys.stdin), 1):
    mel = model.infer(torch.tensor(text_to_ids(text), device=device))
    with torch.no_grad():
        wav = generator(mel[None])[0]
    sf.write(f"{sys.argv[4]}/{number:02d}.wav", (wav / wav.abs().max().clamp(min=1.0)).cpu().numpy(), SR)
"""
lines = "/kaggle/working/lines"
os.makedirs(lines, exist_ok=True)
print(run(sys.executable, "-c", SPEAK, code, acoustic, vocoder, lines, stdin=json.dumps(SENTENCES)))

parts = []
for number, sentence in enumerate(SENTENCES, 1):
    out = f"{lines}/{number:02d}.wav"
    wav, rate = sf.read(out, dtype="float32")
    parts += [wav, np.zeros(int(rate * PAUSE), dtype="float32")]
    print(f"{number}. {sentence}")
    display(Audio(out))

print("All together")
sf.write("/kaggle/working/intro.wav", np.concatenate(parts[:-1]), rate)
display(Audio("/kaggle/working/intro.wav"))
