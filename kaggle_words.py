# One Kaggle notebook cell: how often did the model see these words? Needs Internet and the HF_TOKEN secret.
# Reads only the text columns of the Nepali training shards: no audio is downloaded, no GPU needed.
import collections, re, unicodedata
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from kaggle_secrets import UserSecretsClient

WORDS = ["वन", "बन", "अदालत", "आज", "बन्द", "वकिल", "वकील"]   # also counts words starting with these
EXCLUDED = {"ANGER", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISE"}   # as in prepare_rasa.py

fs = HfFileSystem(token=UserSecretsClient().get_secret("HF_TOKEN").strip())
shards = sorted(fs.glob("datasets/ai4bharat/Rasa/Nepali/train-*.parquet"))
assert shards, "no shards found: is the HF_TOKEN secret ticked?"

texts = []
for number, shard in enumerate(shards, 1):
    with fs.open(shard, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["text", "gender", "style"])
    for row in table.to_pylist():
        if row["gender"] == "Female" and row["style"].upper() not in EXCLUDED:
            texts.append(" ".join(unicodedata.normalize("NFC", row["text"]).split()))
    if number % 10 == 0 or number == len(shards):
        print(f"{number}/{len(shards)} shards, {len(texts)} sentences", flush=True)

# a word is a run of Devanagari letters; the danda full stops (0964, 0965) are not part of it
split = [re.findall("[ऀ-ॣ०-ॿ‌‍]+", text) for text in texts]
for word in WORDS:
    exact = sum(w == word for words in split for w in words)
    forms = collections.Counter(w for words in split for w in words if w.startswith(word) and w != word)
    print(f"\n{word}: alone {exact} times, longer words starting with it {sum(forms.values())}")
    if forms:
        print("   ", ", ".join(f"{w} {n}" for w, n in forms.most_common(8)))
    for text in [t for t, words in zip(texts, split) if word in words][:2]:
        print("    e.g.", text)
pairs = sum(a == "आज" and b == "बन्द" for words in split for a, b in zip(words, words[1:]))
print(f"\n'आज बन्द' next to each other: {pairs} times")
