# Paste this whole file into one Kaggle notebook cell (GPU T4 x2, Internet On). Safe to re-run.
# Secret needed: HF_TOKEN. Attach the dataset or notebook output that holds both checkpoints.
# Change VOCODER_STEPS / SPEECH_STEPS to train further. Both models resume from their last save.
# VOCODER_FROM / SPEECH_FROM: the run stops early if the attached checkpoints are older than these.
# Checkpoints exist only on Kaggle: run this with Save Version -> Save & Run All, so the output is kept
# even if the tab is closed. Keep it the only cell in the notebook. Limit: 12 h per run.
import os, re, subprocess, sys, threading, time
from kaggle_secrets import UserSecretsClient
from IPython.display import Audio, display

VOCODER_STEPS, SPEECH_STEPS = 100000, 135000
VOCODER_FROM, SPEECH_FROM = 50000, 75000
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
started = time.time()
code, data = "/tmp/Swar", "/tmp/rasa"
voc = "/kaggle/working/ckpt/vocoder.pt"
acoustic = "/kaggle/working/ckpt/acoustic_v2.pt"

def start(tag, command, gpu=0, cwd=None):
    print(f"\n$ {tag}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def show():
        for line in process.stdout:
            print(f"[{(time.time() - started) / 60:6.1f}m] {tag:7}| {line}", end="", flush=True)
    reader = threading.Thread(target=show)
    reader.start()
    return tag, process, reader

def wait(*jobs):
    try:
        for _, process, reader in jobs:
            process.wait(); reader.join()
    finally:   # stopping the cell stops the training too
        for _, process, _ in jobs:
            if process.poll() is None:
                process.terminate()
    failed = [tag for tag, process, _ in jobs if process.returncode != 0]
    if failed:
        raise RuntimeError(f"failed: {failed}")

def run(*command, cwd=None):
    wait(start("setup", command, cwd=cwd))

assert subprocess.check_output(["nvidia-smi", "-L"], text=True).count("GPU ") >= 2, "pick GPU T4 x2"
if not os.path.isdir(code):
    run("git", "clone", "https://github.com/kritrim-lab/Swar.git", code)
run("git", "-C", code, "pull")
assert "EMA_DECAY" in open(f"{code}/train.py", encoding="utf-8").read(), "old code: push first"
run(sys.executable, "-m", "pip", "install", "-q", "-r", f"{code}/requirements.txt")
# take the furthest copy of each checkpoint, load both models and report progress before any training
open("/tmp/progress.py", "w").write("""
import glob, os, shutil, sys
sys.path.insert(0, ".")
import torch
from swar.checkpoint import load
from swar.model import AcousticModel
from swar.vocoder import Generator

def saved_step(path):
    return torch.load(path, map_location="cpu", weights_only=True)["step"]

for path, target, least, key, model in ((sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), "model", AcousticModel()),
                                        (sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), "generator", Generator())):
    found = [path] if os.path.exists(path) else []
    found += glob.glob("/kaggle/input/**/" + os.path.basename(path), recursive=True)
    if not found:   # never start from step 0 by accident
        raise SystemExit(f"{path}: NOT FOUND. Attach the dataset or notebook output that holds it.")
    best = max(found, key=saved_step)
    if saved_step(best) < least:   # an older copy would train for hours from the wrong place
        raise SystemExit(f"{best} is only step {saved_step(best)}, need {least}: attach the newer checkpoint.")
    if best != path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(best, path + ".tmp")
        os.replace(path + ".tmp", path)
    step = load(path, **{key: model})
    print(f"{path}: loaded from {best}, step {step} of {target} ({step / target:.0%}), {max(target - step, 0)} left")
""")
run(sys.executable, "/tmp/progress.py", acoustic, str(SPEECH_STEPS), str(SPEECH_FROM),
    voc, str(VOCODER_STEPS), str(VOCODER_FROM), cwd=code)
run(sys.executable, "-u", "prepare_rasa.py", "--out", data, cwd=code)

wait(start("vocoder", [sys.executable, "-u", "train_vocoder.py", "--data", f"{data}/train", "--ckpt", voc,
                       "--steps", str(VOCODER_STEPS), "--save-every", "1000"], gpu=0, cwd=code),
     start("speech", [sys.executable, "-u", "train.py", "--data", f"{data}/train.txt", "--ckpt", acoustic,
                      "--steps", str(SPEECH_STEPS)], gpu=1, cwd=code))

allowed = r"""[\u0900-\u097F\u200C\u200D .,?!;:'"()-]+"""
for line in open(f"{data}/test.txt", encoding="utf-8"):
    wav, text = line.strip().split("|", 1)
    if 40 <= len(text) <= 100 and re.fullmatch(allowed, text):
        break
run(sys.executable, "synth.py", "--ckpt", acoustic, "--vocoder", voc, "--text", text,
    "--out", "/kaggle/working/vocoder.wav", cwd=code)
print(text)
print("Real recording"); display(Audio(f"{data}/{wav}"))
print("Our model + our vocoder"); display(Audio("/kaggle/working/vocoder.wav"))
