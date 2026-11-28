"""Text to ids, audio to mel, and the training dataset."""
import collections
import re
import unicodedata
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

SR, N_FFT, HOP, N_MELS = 22050, 1024, 256, 80
# shortcut: fixed log-mel statistics, compute them from the corpus if training looks unstable
MEL_MEAN, MEL_STD = -5.5, 2.0

# Order is baked into checkpoints: only ever append.
SYMBOLS = (
    ["<pad>"]
    + list(" .,?!;:-'\"()")
    + [chr(c) for c in range(0x0900, 0x0980)]  # Devanagari: includes halanta, nukta, danda, digits
    + ["‌", "‍"]  # ZWNJ / ZWJ change how Nepali conjuncts are read
)
IDS = {s: i for i, s in enumerate(SYMBOLS)}
# curly quotes and long dashes sound the same as their plain forms
PLAIN = str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014", "''\"\"--")

_MEL_ARGS = dict(norm="slaney", mel_scale="slaney")
_mel = torchaudio.transforms.MelSpectrogram(
    SR, N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0, **_MEL_ARGS
)


# Nepali number words 0-99 are irregular, so they are all listed
NUMBERS = """
शून्य एक दुई तीन चार पाँच छ सात आठ नौ
दश एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार उन्नाइस
बीस एक्काइस बाइस तेइस चौबीस पच्चीस छब्बीस सत्ताइस अट्ठाइस उनन्तीस
तीस एकतीस बत्तीस तेत्तीस चौँतीस पैँतीस छत्तीस सैँतीस अठतीस उनन्चालीस
चालीस एकचालीस बयालीस त्रिचालीस चवालीस पैँतालीस छयालीस सतचालीस अठचालीस उनन्पचास
पचास एकाउन्न बाउन्न त्रिपन्न चउन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनन्साठी
साठी एकसट्ठी बयसट्ठी त्रिसट्ठी चौंसट्ठी पैंसट्ठी छयसट्ठी सतसट्ठी अठसट्ठी उनन्सत्तरी
सत्तरी एकहत्तर बहत्तर त्रिहत्तर चौहत्तर पचहत्तर छयहत्तर सतहत्तर अठहत्तर उनासी
असी एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी अठासी उनान्नब्बे
नब्बे एकान्नब्बे बयान्नब्बे त्रियान्नब्बे चौरान्नब्बे पन्चानब्बे छयान्नब्बे सन्तान्नब्बे अन्ठान्नब्बे उनान्सय
""".split()
SCALES = ((1_000_000_000, "अर्ब"), (10_000_000, "करोड"), (100_000, "लाख"), (1000, "हजार"), (100, "सय"))


def number_words(n):
    for size, name in SCALES:
        if n >= size:
            rest = n % size
            return f"{number_words(n // size)} {name}" + (f" {number_words(rest)}" if rest else "")
    return NUMBERS[n]


def _spoken(match):
    raw = match.group()
    # shortcut: phone numbers are guessed by length or a leading zero; decimals and clock times
    # get no special reading, add them when they matter
    if "," not in raw and (len(raw) >= 10 or (len(raw) > 1 and int(raw[0]) == 0)):
        return " ".join(NUMBERS[int(digit)] for digit in raw)
    n = int(raw.replace(",", ""))
    if 1100 <= n < 2000:  # years and amounts are said in hundreds: "उन्नाइस सय त्रिपन्न"
        return f"{NUMBERS[n // 100]} सय" + (f" {NUMBERS[n % 100]}" if n % 100 else "")
    return number_words(n)


def normalise(text):
    # shortcut: no abbreviation or symbol expansion, add it before reading arbitrary text
    text = unicodedata.normalize("NFC", text).translate(PLAIN)
    text = re.sub(r"[0-9०-९]+(?:,[0-9०-९]{2,3})*", _spoken, text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def text_to_ids(text):
    text = normalise(text)
    unknown = sorted(set(text) - IDS.keys())
    if unknown:
        raise ValueError(f"unsupported characters {unknown} in: {text!r}")
    return [IDS[c] for c in text]


def wav_to_mel(wav):
    """wav [..., samples] -> normalised log-mel [..., N_MELS, frames]."""
    return (_mel.to(wav.device)(wav).clamp(min=1e-5).log() - MEL_MEAN) / MEL_STD


def mel_to_wav(mel):
    """Griffin-Lim: robotic, only for checking the acoustic model before the vocoder is trained."""
    magnitude = (mel * MEL_STD + MEL_MEAN).exp()
    linear = torchaudio.transforms.InverseMelScale(N_FFT // 2 + 1, N_MELS, SR, **_MEL_ARGS)(magnitude)
    return torchaudio.transforms.GriffinLim(N_FFT, hop_length=HOP, power=1.0, n_iter=60)(linear)


class TTSDataset(torch.utils.data.Dataset):
    """Manifest lines are `audio_path|text`, paths relative to the manifest."""

    def __init__(self, manifest, max_seconds=15.0):
        manifest = Path(manifest)
        self.items, skipped, unknown = [], 0, collections.Counter()
        for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            name, sep, text = line.partition("|")
            if not sep:
                raise ValueError(f"{manifest}:{number}: expected audio_path|text")
            path = manifest.parent / name
            bad = [c for c in normalise(text) if c not in IDS]
            if bad:  # e.g. Latin words or %, which are spoken as words we cannot guess
                unknown.update(bad)
                skipped += 1
                continue
            ids = text_to_ids(text)
            info = sf.info(str(path))
            seconds = info.frames / info.samplerate
            # alignment needs at least one mel frame per character
            if not ids or seconds > max_seconds or len(ids) > seconds * SR / HOP:
                skipped += 1
                continue
            self.items.append((path, ids))
        print(f"{len(self.items)} clips loaded, {skipped} skipped")
        if unknown:
            print(f"unsupported characters (their lines were skipped): {dict(unknown.most_common())}")
        if not self.items:
            raise ValueError("no usable clips")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, ids = self.items[i]
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        # shortcut: resample + mel on every read, precompute to disk if the GPU sits waiting
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return torch.tensor(ids), wav_to_mel(wav)


def collate(batch):
    ids, mels = zip(*batch)
    id_lens = torch.tensor([len(x) for x in ids])
    mel_lens = torch.tensor([m.shape[1] for m in mels])
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True)
    mels = torch.nn.utils.rnn.pad_sequence([m.T for m in mels], batch_first=True).transpose(1, 2)
    return ids, id_lens, mels, mel_lens
