# Paste this whole file into one Kaggle notebook cell (GPU T4, Internet On). No training. Safe to re-run.
# Secret needed: HF_TOKEN. Add Input: BOTH training outputs, so the old and the new checkpoints are there.
import glob, html, json, os, random, subprocess, sys
from IPython.display import HTML, Audio, display
from kaggle_secrets import UserSecretsClient

OLD, NEW = (50000, 25000), (75000, 50000)   # (speech step, vocoder step)
CLIPS = 100                                 # unseen recordings to score
STRESS = [   # edit freely: Devanagari and digits
    "नमस्ते!",
    "आज २०२६ साल हो र तापक्रम २५ डिग्री छ।",
    "के तपाईं भोलि काठमाडौं आउनुहुन्छ? कृपया मलाई अहिले नै भन्नुहोस्!",
]
code, data, out = "/tmp/Swar", "/tmp/rasa_test", "/kaggle/working/compare"
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()

def run(*command):   # torch stays out of the notebook kernel: everything heavy is a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if process.wait():
        raise RuntimeError("failed, see the lines above")
    return "".join(lines)

if not os.path.isdir(code):
    run("git", "clone", "https://github.com/kritrim-lab/Swar.git", code)
run("git", "-C", code, "pull")
assert '"ema"' in open(f"{code}/swar/checkpoint.py", encoding="utf-8").read(), "old code: push first"
run(sys.executable, "-m", "pip", "install", "-q", "-r", f"{code}/requirements.txt")

# only the four test shards: neither model has trained on them
PREPARE = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import prepare_rasa
data = Path(sys.argv[2])
if not (data / "test.txt").exists():
    (data / "test").mkdir(parents=True, exist_ok=True)
    lines = []
    for shard in prepare_rasa.list_shards():
        if Path(shard).stem[:10] in ("test-00000", "test-00001", "test-00002", "test-00003"):
            parquet = prepare_rasa.fetch(shard, data / "download")
            lines += prepare_rasa.convert_shard(parquet, data / "test", "Female")[0]
            os.remove(parquet)
            print(f"{Path(shard).stem}: {len(lines)} clips so far", flush=True)
    (data / "test.txt").write_text("".join(lines), encoding="utf-8")
"""
run(sys.executable, "-c", PREPARE, code, data)

SCORE = r"""
import glob, json, math, random, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import soundfile as sf
import torch
from swar.checkpoint import load
from swar.data import HOP, MEL_MEAN, MEL_STD, SR, collate, text_to_ids, wav_to_mel
from swar.model import AcousticModel
from swar.vocoder import Generator

data, out = sys.argv[2], sys.argv[3]
config = json.loads(sys.argv[4])
device = "cuda" if torch.cuda.is_available() else "cpu"
DB = 20 / math.log(10)

def find(name, step):
    found = {}
    for root in ("/kaggle/input", "/kaggle/working"):
        for path in glob.glob(f"{root}/**/{name}", recursive=True):
            found.setdefault(torch.load(path, map_location="cpu", weights_only=True)["step"], path)
    if step not in found:
        raise SystemExit(f"{name} at step {step} not found; steps attached: {sorted(found)}")
    print(f"{name} step {step}: {found[step]}", flush=True)
    return found[step]

pairs = {}
for label, (speech, vocoder) in config["pairs"].items():
    model, generator = AcousticModel(), Generator()
    path = find("acoustic_v2.pt", speech)
    load(path, model=model)
    load(path, ema=model)  # what synth.py speaks with
    load(find("vocoder.pt", vocoder), generator=generator)
    pairs[label] = model.eval().to(device), generator.eval().to(device)

clips, stress = [], config["stress"]
lines = open(f"{data}/test.txt", encoding="utf-8").read().splitlines()
random.Random(0).shuffle(lines)
for line in lines:
    name, text = line.split("|", 1)
    try:
        ids = text_to_ids(text)
    except ValueError:
        continue
    wav, _ = sf.read(f"{data}/{name}", dtype="float32")
    mel = wav_to_mel(torch.from_numpy(wav))
    if ids and len(wav) <= 15 * SR and len(ids) <= mel.shape[1]:
        clips.append((f"{data}/{name}", text, torch.tensor(ids), mel))
    if len(clips) == config["clips"]:
        break
print(f"{len(clips)} unseen clips, on {device}", flush=True)

def level(mel):   # loudness of each frame in dB
    return DB * torch.logsumexp(mel * MEL_STD + MEL_MEAN, dim=0) - DB * math.log(len(mel))

def too_loud(made, asked):   # speech frames the vocoder made more than 1.5 dB louder than asked
    a, b = level(asked), level(made[:, :asked.shape[1]])
    speech = a > a.max() - 35
    return ((b - a > 1.5) & speech).sum().item(), speech.sum().item()

def warped_distance(a, b):   # mean mel difference in dB after matching the two timings
    cost = (torch.cdist(a.T.double(), b.T.double(), p=1) / len(a) * MEL_STD * DB).cpu().numpy()
    n, m = cost.shape
    total, steps = np.full((n + 1, m + 1), np.inf), np.zeros((n + 1, m + 1))
    total[0, 0] = 0
    for k in range(2, n + m + 1):   # one anti-diagonal at a time
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        before = np.stack([total[i - 1, j - 1], total[i - 1, j], total[i, j - 1]])
        count = np.stack([steps[i - 1, j - 1], steps[i - 1, j], steps[i, j - 1]])
        pick, column = before.argmin(0), np.arange(len(i))
        total[i, j] = before[pick, column] + cost[i - 1, j - 1]
        steps[i, j] = count[pick, column] + 1
    return total[n, m] / steps[n, m]

results = {}
for label, (model, generator) in pairs.items():
    scores = {k: [] for k in ("prior", "duration", "flow", "length", "distance", "vocoder")}
    loud, loud_real = [0, 0], [0, 0]
    for number, (_, _, ids, mel) in enumerate(clips):
        batch = [t.to(device) for t in collate([(ids, mel)])]
        real = mel.to(device)
        torch.manual_seed(number)   # the same random noise for both model pairs
        with torch.no_grad():
            losses, _ = model(*batch)
            for key, value in losses.items():
                scores[key].append(value.item())
            made = model.infer(ids.to(device))
            spoken = wav_to_mel(generator(made[None])[0])
            copied = wav_to_mel(generator(real[None])[0])
        scores["length"].append(abs(made.shape[1] / real.shape[1] - 1) * 100)
        scores["distance"].append(warped_distance(made, real))
        frames = min(real.shape[1], copied.shape[1])
        scores["vocoder"].append((copied[:, :frames] - real[:, :frames]).abs().mean().item() * MEL_STD)
        for totals, pair in ((loud, too_loud(spoken, made)), (loud_real, too_loud(copied, real))):
            totals[0] += pair[0]
            totals[1] += pair[1]
        if (number + 1) % 25 == 0:
            print(f"{label}: {number + 1}/{len(clips)}", flush=True)
    results[label] = dict(scores=scores, loud=100 * loud[0] / loud[1], loud_real=100 * loud_real[0] / loud_real[1])

samples = [(text, path) for path, text, _, _ in clips if 40 <= len(text) <= 100][:4] + [(t, None) for t in stress]
for number, (text, _) in enumerate(samples):
    for label, (model, generator) in pairs.items():
        torch.manual_seed(number)
        with torch.no_grad():
            wav = generator(model.infer(torch.tensor(text_to_ids(text), device=device))[None])[0]
        sf.write(f"{out}/{number}_{label}.wav", (wav / wav.abs().max().clamp(min=1.0)).cpu().numpy(), SR)
with open(f"{out}/results.json", "w", encoding="utf-8") as handle:
    json.dump(dict(results=results, samples=samples), handle)
"""
os.makedirs(out, exist_ok=True)
settings = json.dumps(dict(pairs={"old": OLD, "new": NEW}, clips=CLIPS, stress=STRESS))
run(sys.executable, "-c", SCORE, code, data, out, settings)

report = json.load(open(f"{out}/results.json", encoding="utf-8"))
old, new = report["results"]["old"], report["results"]["new"]
print(f"\nOld = speech {OLD[0]} + vocoder {OLD[1]}.  New = speech {NEW[0]} + vocoder {NEW[1]}.  Lower is better.")
print(f"{'on unseen recordings':38}{'old':>9}{'new':>9}   new is better on")
for key, name in (("prior", "speech: sound per letter (prior)"), ("duration", "speech: timing (duration)"),
                  ("flow", "speech: detail (flow)"), ("length", "sentence length error, %"),
                  ("distance", "distance from real voice, dB"), ("vocoder", "vocoder: copying real audio")):
    a, b = old["scores"][key], new["scores"][key]
    wins = sum(y < x for x, y in zip(a, b))
    print(f"{name:38}{sum(a) / len(a):9.3f}{sum(b) / len(b):9.3f}   {wins}/{len(a)} clips")
print(f"{'too-loud frames, speaking from text %':38}{old['loud']:9.2f}{new['loud']:9.2f}")
print(f"{'too-loud frames, copying real audio %':38}{old['loud_real']:9.2f}{new['loud_real']:9.2f}")

print("\nBLIND TEST: for each sentence decide which of A or B sounds more like a real person.")
key = []
for number, (text, path) in enumerate(report["samples"]):
    order = random.sample(["old", "new"], 2)
    key.append(f"{number + 1}: A = {order[0]}, B = {order[1]}")
    print(f"\n{number + 1}. {text}")
    if path:
        print("real recording"); display(Audio(path))
    for letter, label in zip("AB", order):
        print(letter); display(Audio(f"{out}/{number}_{label}.wav"))
display(HTML("<details><summary>Show which was which (open after choosing)</summary><pre>"
             + html.escape("\n".join(key)) + "</pre></details>"))
