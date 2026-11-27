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
Free sessions get cut and their disks can be wiped, so set up Backblaze below.

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

## Backblaze B2 checkpoints

When configured, both training commands download the checkpoint at the start and upload it at
every save (`--save-every`, default 500 steps).

1. In Backblaze, create a **private** bucket, then an application key limited to that bucket with
   **Read and Write** access (it must be able to delete files). Never put the key in the repo or
   in a notebook cell.
2. Add three secrets named `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY` and `B2_BUCKET`
   (Colab: the key icon in the sidebar, with notebook access on. Kaggle: Add-ons, Secrets).
3. Before training, run one of these in a notebook cell. The `!python ...` commands inherit them.

```python
import os
from google.colab import userdata                  # Colab
for name in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET"):
    os.environ[name] = userdata.get(name)
```

```python
import os
from kaggle_secrets import UserSecretsClient       # Kaggle
for name in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET"):
    os.environ[name] = UserSecretsClient().get_secret(name)
```

Check the training output: `B2 is OFF` means the variables did not arrive.

How it behaves:

- The file in B2 is `checkpoints/<name of --ckpt>`. **The same name resumes the same run**, so use a
  new name for a new experiment (`trial.pt` for the 8-clip test, `acoustic.pt` for the real one).
  `B2_PREFIX` changes the folder.
- Only the **2 newest versions** of each file are kept; older ones are permanently deleted.
- Whichever of B2 and the local file has more training steps wins.
- If B2 cannot be checked at startup, training stops. A failed upload during training only warns
  and is retried at the next save; a failed final upload ends with an error.
- Run one session per checkpoint name at a time, or they overwrite each other.
- Each upload pauses training; the output prints how long. Raise `--save-every` if it is slow.
\n