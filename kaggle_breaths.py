# One Kaggle notebook cell: do the female speaker's recordings contain breaths and laughs?
# No GPU needed (use a CPU session). Internet On, HF_TOKEN secret ticked. It only finds LIKELY spots:
# your ears decide what they are.
import collections, io, os, re
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from IPython.display import Audio, display
from kaggle_secrets import UserSecretsClient

SHARDS = ["00000", "00010", "00020"]   # a sample of the training files; add more to search further
PLAY = 12                              # likely breaths to play
LAUGH_CLIPS = 6                        # happy / surprised recordings to play

def breaths(wav, rate):
    """Stretches of quiet, hiss-like sound lasting 0.2-1 s: (start s, end s, score)."""
    size, hop = int(0.025 * rate), int(0.010 * rate)
    if len(wav) < size * 10:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(wav, size)[::hop] * np.hanning(size)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    level = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    band = power[:, int(100 * size / rate):int(6000 * size / rate)]
    flatness = np.exp(np.log(band).mean(1)) / band.mean(1)   # near 0 for voice, higher for hiss
    peak, floor = np.percentile(level, 95), np.percentile(level, 5)
    soft = (level > max(floor + 6, peak - 50)) & (level < peak - 18) & (flatness > 0.08)
    edges = np.flatnonzero(np.diff(np.r_[False, soft, False]))
    found = []
    for a, b in zip(edges[::2], edges[1::2]):
        seconds = (b - a) * hop / rate
        if 0.2 <= seconds <= 1.0:
            found.append((a * hop / rate, b * hop / rate, seconds * flatness[a:b].mean()))
    return found

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
candidates, laughs, styles, laugh_text = [], [], collections.Counter(), []
with_breath = clips = 0
for shard in SHARDS:
    path = hf_hub_download("ai4bharat/Rasa", f"Nepali/train-{shard}-of-00057.parquet", repo_type="dataset",
                           local_dir="/tmp/rasa_breaths")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["text", "gender", "style", "audio"]):
        for row in batch.to_pylist():
            if row["gender"] != "Female":
                continue
            wav, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            wav, style = wav.mean(1), row["style"].upper()
            clips += 1
            styles[style] += 1
            found = breaths(wav, rate)
            with_breath += bool(found)
            candidates += [(score, a, b, style, row["text"], wav, rate) for a, b, score in found]
            candidates = sorted(candidates, key=lambda c: -c[0])[:PLAY]
            if style in ("HAPPY", "SURPRISE") and len(laughs) < LAUGH_CLIPS:
                laughs.append((style, row["text"], wav, rate))
            if re.search("हाहा|हा हा|हेहे|हे हे|हिहि|ही ही", row["text"]):
                laugh_text.append(row["text"])
    os.remove(path)
    print(f"{shard}: {clips} clips so far, {with_breath} with a likely breath", flush=True)

print(f"\n{with_breath} of {clips} clips have at least one likely breath ({with_breath / max(clips, 1):.0%})")
print("styles searched:", dict(styles))
print(f"transcripts that spell out laughter: {len(laugh_text)}")
for text in laugh_text[:5]:
    print("   ", text)

print("\n--- LIKELY BREATHS: for each, the spot in context, then the spot alone (made louder) ---")
for number, (_, a, b, style, text, wav, rate) in enumerate(candidates, 1):
    print(f"\n{number}. {style}, at {a:.1f} s: {text}")
    display(Audio(wav[int(max(a - 0.8, 0) * rate):int((b + 0.8) * rate)], rate=rate))
    display(Audio(wav[int(a * rate):int(b * rate)], rate=rate))

print("\n--- HAPPY / SURPRISE recordings: listen for laughs, sighs or gasps ---")
for style, text, wav, rate in laughs:
    print(f"\n{style}: {text}")
    display(Audio(wav, rate=rate))
