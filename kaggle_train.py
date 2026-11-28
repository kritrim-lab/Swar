# Paste this whole file into one Kaggle notebook cell (GPU T4 x2, Internet On). Safe to re-run.
# Secrets needed: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET, HF_TOKEN.
# Change VOCODER_STEPS / SPEECH_STEPS to train further. Both models resume from their last save.
import glob, os, re, shutil, subprocess, sys, threading, time
from kaggle_secrets import UserSecretsClient
from IPython.display import Audio, display

VOCODER_STEPS, SPEECH_STEPS = 25000, 50000
for name in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET", "HF_TOKEN"):
    os.environ[name] = UserSecretsClient().get_secret(name).strip()
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
assert "daily download cap" in open(f"{code}/swar/checkpoint.py", encoding="utf-8").read(), "old code: push first"
run(sys.executable, "-m", "pip", "install", "-q", "-r", f"{code}/requirements.txt")
for path in (voc, acoustic):   # from an attached dataset or earlier notebook output: no B2 download
    found = glob.glob(f"/kaggle/input/**/{os.path.basename(path)}", recursive=True)
    print(path, "<-", found[:1] if not os.path.exists(path) else "already here")
    if found and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(found[0], path)

# load both models and report progress before any training
open("/tmp/progress.py", "w").write("""
import os, sys
sys.path.insert(0, ".")
from swar.checkpoint import connect, load, pull
from swar.model import AcousticModel
from swar.vocoder import Generator
bucket = connect()
for path, target, key, model in ((sys.argv[1], int(sys.argv[2]), "model", AcousticModel()),
                                 (sys.argv[3], int(sys.argv[4]), "generator", Generator())):
    if bucket:
        pull(bucket, path)   # newest first, so this is the step training resumes from
    if not os.path.exists(path):
        print(f"{path}: NOT FOUND, training would start from step 0")
        continue
    step = load(path, **{key: model})
    print(f"{path}: loaded, step {step} of {target} ({step / target:.0%}), {max(target - step, 0)} left")
""")
run(sys.executable, "/tmp/progress.py", acoustic, str(SPEECH_STEPS), voc, str(VOCODER_STEPS), cwd=code)
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
