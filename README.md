<img width="1671" height="941" alt="Swar" src="https://github.com/user-attachments/assets/412dddff-026a-4e49-abe9-fef92af833e2" />


Experimental Nepali text-to-speech, model code written from scratch in PyTorch.

```
Nepali text -> encoder -> alignment / durations -> flow decoder -> mel spectrogram -> vocoder -> audio
              |---------------------- acoustic model --------------------|
```

| Part | Parameters | Used when speaking |
|---|---|---|
| Acoustic model | 16.9M | yes |
| Vocoder generator | 13.5M | yes |
| Vocoder discriminators | 41.4M | no, training only |

## Train (Google Colab or Kaggle, GPU runtime)

PyTorch is preinstalled on both. Make a manifest with one `audio_path|text` line per clip
(paths relative to the manifest, clips up to 15 s, one voice per manifest because the model has
no speaker input). Lines with characters outside Devanagari and basic punctuation are skipped.

To start with real data, [OpenSLR 143](https://openslr.org/143/) needs no sign-up (about 1 hour of
female speech: enough to prove training works, not enough for a good voice; its page labels it
non-commercial):

```
wget https://openslr.org/resources/143/male-female-data.tgz && tar -xzf male-female-data.tgz
python prepare_openslr143.py male-female-data --voice female   # writes male-female-data/female.txt
```

Then:

```
pip install -r requirements.txt
python test_swar.py
python train.py --data /path/to/manifest.txt --ckpt /path/to/acoustic.pt
python train_vocoder.py --data /folder/with/wavs --ckpt /path/to/vocoder.pt
python synth.py --ckpt /path/to/acoustic.pt --vocoder /path/to/vocoder.pt --text "नमस्ते" --out hello.wav
```

The two models train separately, in any order. Re-running a training command resumes from `--ckpt`.
Free sessions get cut and their disks can be wiped, so keep `--ckpt` somewhere that survives: on
Kaggle under `/kaggle/working` followed by Save Version, on Colab on a mounted Google Drive.

First run on real data: train on a manifest of about 8 clips with `--steps 2000` and check that
`flow` and `prior` loss fall (`flow` falls slowly and never reaches zero) and one of those sentences
is recognisable, before starting a long run.


## Full dataset: Rasa

[AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (CC BY 4.0, by AI4Bharat) has about
29 hours of one female Nepali speaker. Accept its terms on that page, create a Hugging Face read
token, store it as the `HF_TOKEN` secret, load it into the environment, then:

```
python prepare_rasa.py --out /content/rasa
python train.py --data /content/rasa/train.txt --ckpt /content/ckpt/acoustic.pt
python train_vocoder.py --data /content/rasa/train --ckpt /content/ckpt/vocoder.pt
```

It downloads about 18 GB (both speakers are mixed in the same files) and keeps the female,
non-emotional clips as 22.05 kHz audio. Sentences in `test.txt` are never trained on: use them to
judge how the model reads new text.
