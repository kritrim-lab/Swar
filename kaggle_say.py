# Second notebook cell: run the kaggle_intro.py cell once first (it sets code, acoustic, vocoder).
# The first run of this cell loads the models; later runs reuse them. Edit SENTENCES and run again.
import json, subprocess, sys
from IPython.display import Audio, display

SENTENCES = [   # Devanagari and digits only
    "के तपाईं भोलि काठमाडौं आउनुहुन्छ?",
]
SPEED, TEMPERATURE = 1.0, 1.0   # speed 0.9 is slower; temperature 0.667 is flatter

# holds the models and answers one request per line; torch stays out of the notebook kernel
WORKER = """
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
step = load(sys.argv[2], model=model)
load(sys.argv[2], ema=model)
voice_step = load(sys.argv[3], generator=generator)
model.eval().to(device)
generator.eval().to(device)
print(json.dumps(f"loaded speech {step} and vocoder {voice_step} on {device}"), flush=True)
for number, line in enumerate(sys.stdin):
    text, speed, temperature = json.loads(line)
    try:
        ids = text_to_ids(text)
        if not ids:
            raise ValueError("empty sentence")
        mel = model.infer(torch.tensor(ids, device=device), speed, temperature=temperature)
        with torch.no_grad():
            wav = generator(mel[None])[0]
        path = f"/kaggle/working/say_{number:03d}.wav"
        sf.write(path, (wav / wav.abs().max().clamp(min=1.0)).cpu().numpy(), SR)
        print(json.dumps({"wav": path}), flush=True)
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
"""

def reply():
    line = speaker.stdout.readline()
    if not line:
        raise RuntimeError("the speaking process stopped; run this cell again and read the error above")
    return json.loads(line)

if "speaker" not in globals() or speaker.poll() is not None:
    speaker = subprocess.Popen([sys.executable, "-c", WORKER, code, acoustic, vocoder],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    print(reply())

for text in SENTENCES:
    speaker.stdin.write(json.dumps([text, SPEED, TEMPERATURE]) + "\n")
    speaker.stdin.flush()
    answer = reply()
    print(text)
    if "error" in answer:
        print("  could not speak it:", answer["error"])
    else:
        display(Audio(answer["wav"]))
