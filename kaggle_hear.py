# One Kaggle notebook cell: hear how the real speaker says a word. Needs Internet and the HF_TOKEN secret.
# If the kaggle_say.py cell has run, your model also speaks each of the same sentences, for comparison.
import json, re, unicodedata
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from IPython.display import Audio, display
from kaggle_secrets import UserSecretsClient

WORDS = {"वन": False, "अदालत": True}   # True: also accept longer words that start with it
CLIPS = 3                              # recordings per word
EXCLUDED = {"ANGER", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISE"}   # as in prepare_rasa.py

def has(text, word, prefix):
    words = re.findall("[ऀ-ॣ०-ॿ‌‍]+", unicodedata.normalize("NFC", text))
    return any(w.startswith(word) if prefix else w == word for w in words)

fs = HfFileSystem(token=UserSecretsClient().get_secret("HF_TOKEN").strip())
shards = sorted(fs.glob("datasets/ai4bharat/Rasa/Nepali/train-*.parquet"))
assert shards, "no shards found: is the HF_TOKEN secret ticked?"
wanted = {word: CLIPS for word in WORDS}
found = []   # (word, text, wav bytes)
for shard in shards:
    if not any(wanted.values()):
        break
    with fs.open(shard, "rb") as handle:
        file = pq.ParquetFile(handle)
        for group in range(file.num_row_groups):
            rows = file.read_row_group(group, columns=["text", "gender", "style"]).to_pylist()
            picks = []
            for i, row in enumerate(rows):
                if row["gender"] != "Female" or row["style"].upper() in EXCLUDED:
                    continue
                for word, prefix in WORDS.items():
                    if wanted[word] and has(row["text"], word, prefix):
                        wanted[word] -= 1
                        picks.append((i, word, row["text"]))
                        break
            if picks:   # audio is large: read it only where a wanted sentence is
                audio = file.read_row_group(group, columns=["audio"]).column("audio")
                found += [(word, text, audio[i].as_py()["bytes"]) for i, word, text in picks]
    print(f"searched {shard.rsplit('/', 1)[-1]}: {len(found)} recordings so far", flush=True)

for word, text, wav in sorted(found, key=lambda item: item[0]):
    print(f"\n{word}:  {text}")
    print("real speaker")
    display(Audio(data=wav))
    if "speaker" in globals() and speaker.poll() is None:   # the model loaded by kaggle_say.py
        speaker.stdin.write(json.dumps([text, 1.0, 1.0]) + "\n")
        speaker.stdin.flush()
        answer = reply()
        print("your model" if "wav" in answer else f"your model could not speak it: {answer['error']}")
        if "wav" in answer:
            display(Audio(answer["wav"]))
