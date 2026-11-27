"""Turn the extracted OpenSLR 143 folder into a training manifest.

    python prepare_openslr143.py male-female-data --voice female
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("folder", type=Path)
parser.add_argument("--voice", choices=["female", "male"], default="female")
args = parser.parse_args()

source = args.folder / f"{args.voice.title()}Voice.tsv"  # columns: audio_id, sentence
lines = []
for row in source.read_text(encoding="utf-8-sig").splitlines()[1:]:
    if not row.strip():
        continue
    audio_id, sentence = row.split("\t")
    if not (args.folder / f"{audio_id}.wav").is_file():
        raise SystemExit(f"missing audio for {audio_id}")
    lines.append(f"{audio_id}.wav|{sentence.strip()}\n")
manifest = args.folder / f"{args.voice}.txt"
manifest.write_text("".join(lines), encoding="utf-8")
print(f"wrote {len(lines)} lines to {manifest}")
