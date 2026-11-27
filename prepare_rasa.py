"""Download AI4Bharat Rasa (Nepali) and keep one speaker as 22.05 kHz wav files. Run on Colab or Kaggle.

    HF_TOKEN=... python prepare_rasa.py --out /content/rasa

Writes <out>/train.txt (train on this) and <out>/test.txt (unseen sentences, never train on these).
Each 250-300 MB shard is deleted once converted. Re-running skips finished shards.
"""
import argparse
import collections
import io
import os
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from swar.data import SR

REPO = "ai4bharat/Rasa"
# the model has no style input, so emotional clips would only blur the neutral voice
EXCLUDED_STYLES = {"ANGER", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISE"}


def list_shards():
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                  if f.startswith("Nepali/") and f.endswith(".parquet"))


def fetch(shard, folder):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, shard, repo_type="dataset", local_dir=folder)


def convert_shard(parquet, wavs, gender):
    """Writes this speaker's clips into wavs/. Returns manifest lines and {(style, kept): seconds}."""
    import pyarrow.parquet as pq

    lines, seconds = [], collections.Counter()
    columns = ["filename", "text", "gender", "style", "audio"]
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            if row["gender"] != gender:
                continue
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            kept = row["style"].upper() not in EXCLUDED_STYLES
            seconds[row["style"], kept] += len(audio) / rate
            if not kept:
                continue
            name = row["filename"].removesuffix(".wav")
            if not re.fullmatch(r"\w[\w.-]*", name):  # it becomes a path, so trust nothing odd
                raise ValueError(f"unsafe file name in {parquet}: {name!r}")
            wav = torch.from_numpy(audio.mean(axis=1))
            if rate != SR:
                wav = torchaudio.functional.resample(wav, rate, SR)
            sf.write(wavs / f"{name}.wav", wav.clamp(-1, 1).numpy(), SR, subtype="PCM_16")
            lines.append(f"{wavs.name}/{name}.wav|{' '.join(row['text'].split())}\n")
    return lines, seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gender", default="Female", choices=["Female", "Male"])
    args = parser.parse_args(argv)

    shards = list_shards()
    if not shards:
        raise SystemExit("no Nepali shards found: is HF_TOKEN set and the dataset access accepted?")
    totals = collections.Counter()
    for number, shard in enumerate(shards, 1):
        stem = Path(shard).stem  # e.g. train-00000-of-00057
        split = stem.split("-")[0]
        part = args.out / "parts" / f"{stem}.txt"
        if part.exists():
            continue
        wavs = args.out / split  # separate folders, so the vocoder never trains on test audio
        wavs.mkdir(parents=True, exist_ok=True)
        part.parent.mkdir(parents=True, exist_ok=True)
        parquet = fetch(shard, args.out / "download")
        lines, seconds = convert_shard(parquet, wavs, args.gender)
        os.remove(parquet)
        totals.update(seconds)
        # written last and atomically: a part file means the whole shard is done
        part.with_suffix(".tmp").write_text("".join(lines), encoding="utf-8")
        os.replace(part.with_suffix(".tmp"), part)
        print(f"[{number}/{len(shards)}] {stem}: kept {len(lines)} clips", flush=True)

    for split in ("train", "test"):
        parts = sorted((args.out / "parts").glob(f"{split}-*.txt"))
        text = "".join(p.read_text(encoding="utf-8") for p in parts)
        (args.out / f"{split}.txt").write_text(text, encoding="utf-8")
        print(f"{split}.txt: {text.count(chr(10))} clips")
    for (style, kept), total in sorted(totals.items()):
        print(f"  {'kept   ' if kept else 'dropped'} {style:16} {total / 3600:6.2f} h  (this run)")


if __name__ == "__main__":
    main()
