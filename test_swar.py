"""Quick correctness checks on fake data, no GPU or dataset needed: python test_swar.py"""
import importlib.util
import io
import itertools
import math
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch

import prepare_rasa
import train
import train_vocoder
from swar import checkpoint
from swar.data import HOP, SR, normalise, text_to_ids, wav_to_mel
from swar.model import monotonic_alignment_search


def brute_force(score):
    n_text, n_mel = score.shape
    best = -math.inf
    for cuts in itertools.combinations(range(1, n_mel), n_text - 1):
        edges = (0, *cuts, n_mel)
        best = max(best, sum(score[i, edges[i]:edges[i + 1]].sum().item() for i in range(n_text)))
    return best


def test_alignment():
    torch.manual_seed(0)
    for _ in range(50):
        text_lens, mel_lens = torch.tensor([4, 2, 1]), torch.tensor([7, 9, 3])
        score = torch.randn(3, 4, 9)
        path = monotonic_alignment_search(score, text_lens, mel_lens)
        for b, (n, t) in enumerate(zip(text_lens.tolist(), mel_lens.tolist())):
            p = path[b, :n, :t]
            assert path[b].sum() == t, "path leaked into padding"
            assert (p.sum(0) == 1).all() and (p.sum(1) >= 1).all()
            assert (p.argmax(0).diff() >= 0).all(), "alignment went backwards"
            assert math.isclose((p * score[b, :n, :t]).sum().item(), brute_force(score[b, :n, :t]), abs_tol=1e-4)


def test_text():
    assert len(text_to_ids("नमस्ते")) == 6
    assert text_to_ids("  क   ख ") == text_to_ids("क ख")
    assert text_to_ids("\u201cक\u201d \u2014") == text_to_ids('"क" -')
    try:
        text_to_ids("hello")
    except ValueError:
        return
    raise AssertionError("unsupported characters were accepted")


def test_numbers():
    assert normalise("५ र 21") == "पाँच र एक्काइस"
    assert normalise("१००") == "एक सय"
    assert normalise("१,२५,०००") == "एक लाख पच्चीस हजार"
    assert normalise("२०८३ साल") == "दुई हजार त्रियासी साल"
    assert normalise("१९५३") == "उन्नाइस सय त्रिपन्न"
    assert normalise("१,०१,५००") == "एक लाख एक हजार पाँच सय"
    assert normalise("०१") == "शून्य एक"
    assert normalise("९८४१२३४५६७") == "नौ आठ चार एक दुई तीन चार पाँच छ सात"
    assert normalise("१, २") == "एक, दुई", "a list was read as one number"
    for n in (0, 99, 101, 1001, 12_34_56_789, 5_000_000_000):
        text_to_ids(str(n))  # every word uses known characters


def test_train_resume_and_synth():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines = []
        for i, (text, seconds) in enumerate([("नमस्ते।", 1.0), ("मेरो घर", 1.4), ("धन्यवाद", 0.8)]):
            sf.write(tmp / f"{i}.wav", 0.1 * torch.randn(int(SR * seconds)).numpy(), SR)
            lines.append(f"{i}.wav|{text}")
        (tmp / "manifest.txt").write_text("\n".join(lines), encoding="utf-8")
        ckpt = str(tmp / "ckpt" / "a.pt")
        args = ["--data", str(tmp / "manifest.txt"), "--ckpt", ckpt, "--batch-size", "3", "--workers", "0"]
        train.main(args + ["--steps", "2"])
        train.main(args + ["--steps", "3"])
        state = torch.load(ckpt, weights_only=True)
        assert state["step"] == 3, "did not resume"
        key = "to_mu.weight"
        assert not torch.equal(state["ema"][key], state["model"][key]), "averaged weights are not separate"
        del state["ema"]  # as saved before weight averaging was added
        torch.save(state, ckpt)
        train.main(args + ["--steps", "4"])
        assert "ema" in torch.load(ckpt, weights_only=True), "old speech checkpoint did not resume"

        voc = str(tmp / "ckpt" / "v.pt")
        args = ["--data", str(tmp), "--ckpt", voc, "--batch-size", "2", "--workers", "0"]
        train_vocoder.main(args + ["--steps", "2"])
        train_vocoder.main(args + ["--steps", "3"])
        state = torch.load(voc, weights_only=True)
        assert state["step"] == 3, "vocoder did not resume"
        del state["scaler"]  # as saved before mixed precision was added
        torch.save(state, voc)
        train_vocoder.main(args + ["--steps", "4"])
        assert torch.load(voc, weights_only=True)["step"] == 4, "old vocoder checkpoint did not resume"

        out = tmp / "out.wav"
        sys.argv = ["synth.py", "--ckpt", ckpt, "--vocoder", voc, "--text", "नमस्ते", "--out", str(out)]
        runpy.run_path(str(Path(__file__).parent / "synth.py"), run_name="__main__")
        assert sf.info(str(out)).frames > 0


def test_learns_alignment():
    from swar.model import AcousticModel

    torch.manual_seed(0)
    batch, n = 16, 8
    ids = torch.randint(1, 20, (batch, n))
    truth = torch.randint(2, 9, (batch, n))  # how many frames each character really lasts
    # as in real speech, a character's sound depends on its neighbour (here: the next one)
    book, after = torch.randn(20, 80), torch.randn(20, 80)
    following = torch.cat([ids[:, 1:], torch.zeros(batch, 1, dtype=torch.long)], dim=1)
    sounds = (book[ids] + after[following]) / 2 ** 0.5
    clips = [s.repeat_interleave(d, dim=0) for s, d in zip(sounds, truth)]
    mel_lens = torch.tensor([len(c) for c in clips])
    mels = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True).transpose(1, 2)

    model = AcousticModel(dim=64, encoder_layers=2, decoder_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    history = []
    for step in range(100):
        losses, align = model(ids, torch.full((batch,), n), mels, mel_lens,
                              diagonal=train.DIAGONAL * max(0.0, 1 - step / 60))
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # the flow loss is noisy and never reaches zero, so judge progress on the other two
        history.append((losses["prior"] + losses["duration"]).item())
    close = ((align.sum(-1) - truth).abs() <= 1).float().mean().item()
    print(f"loss {history[0]:.2f} -> {history[-1]:.2f}, durations within 1 frame: {close:.0%}")
    assert history[-1] < 0.5 * history[0], "loss did not fall"
    assert close >= 0.85, "alignment did not find the true durations"
    prior = losses["prior"].item()
    print(f"prior loss {prior:.3f}")
    assert prior < 0.15, "mu cannot represent sounds that depend on context"

    # the sampler must turn noise into the right sounds, end to end
    model.eval()
    text_mask = torch.ones(ids.shape, dtype=torch.bool)
    frame_mask = torch.arange(mels.shape[2])[None] < mel_lens[:, None]
    with torch.no_grad():
        condition = align.transpose(1, 2) @ model.encode(ids, text_mask)[0]
    # lower temperature: this barely trained toy is judged on distance to one exact target
    sampled = model.sample(condition, frame_mask, temperature=0.667)
    target = mels.transpose(1, 2)
    error = (sampled - target)[frame_mask].abs().mean().item()
    from_noise = (torch.randn_like(target) - target)[frame_mask].abs().mean().item()
    print(f"sampled mel error {error:.2f}, pure noise would be {from_noise:.2f}")
    assert error < 0.5 * from_noise, "sampler does not reconstruct the sounds"


def test_vocoder_learns():
    from swar.vocoder import Generator

    torch.manual_seed(0)
    time = torch.arange(32 * HOP) / SR
    wav = sum(0.2 * torch.sin(2 * math.pi * f * time) for f in (220, 440, 880))[None]
    # speech fills the whole spectrum; a bare tone is mostly silence, where log-mel error is erratic
    wav = wav + 0.05 * torch.randn_like(wav)
    mel = wav_to_mel(wav)
    generator = Generator(dim=64, hidden=128, layers=2)
    assert generator(mel).shape == wav.shape, "output is not aligned with the input audio"
    optimizer = torch.optim.AdamW(generator.parameters(), lr=3e-3)
    history = []
    for _ in range(60):
        loss = (wav_to_mel(generator(mel)) - mel).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    print(f"vocoder mel loss {history[0]:.2f} -> {history[-1]:.2f}")
    assert history[-1] < 0.3 * history[0], "vocoder loss did not fall"


def test_prepare_rasa():
    """A tiny fake Rasa shard: no download and no token."""
    if importlib.util.find_spec("pyarrow") is None:
        print("pyarrow not installed, skipping")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    def wav_bytes():
        buffer = io.BytesIO()
        sf.write(buffer, 0.1 * torch.randn(48000).numpy(), 48000, format="WAV")
        return buffer.getvalue()

    rows = [{"filename": name, "text": "नमस्ते  घर", "gender": gender, "style": style,
             "audio": {"bytes": wav_bytes(), "path": name}}
            for name, gender, style in [("F_NEWS_1", "Female", "NEWS"), ("F_FEAR_1", "Female", "FEAR"),
                                        ("M_NEWS_1", "Male", "NEWS")]]
    real = prepare_rasa.list_shards, prepare_rasa.fetch
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pq.write_table(pa.Table.from_pylist(rows), tmp / "fake.parquet")
        fetched = []

        def fake_fetch(shard, folder):
            fetched.append(shard)
            Path(folder).mkdir(parents=True, exist_ok=True)
            return shutil.copy(tmp / "fake.parquet", Path(folder) / "shard.parquet")

        prepare_rasa.list_shards = lambda: ["Nepali/test-00000-of-00001.parquet", "Nepali/train-00000-of-00001.parquet"]
        prepare_rasa.fetch = fake_fetch
        try:
            out = tmp / "rasa"
            prepare_rasa.main(["--out", str(out)])
            prepare_rasa.main(["--out", str(out)])
        finally:
            prepare_rasa.list_shards, prepare_rasa.fetch = real
        assert len(fetched) == 2, "a rerun downloaded finished shards again"
        for split in ("train", "test"):
            assert (out / f"{split}.txt").read_text(encoding="utf-8") == f"{split}/F_NEWS_1.wav|नमस्ते घर\n"
            info = sf.info(str(out / split / "F_NEWS_1.wav"))
            assert (info.samplerate, info.subtype, info.frames) == (SR, "PCM_16", SR)
        assert sorted(p.name for p in out.rglob("*.wav")) == ["F_NEWS_1.wav"] * 2, "male or emotional clip kept"
        assert not list(out.rglob("*.parquet")), "downloaded shard was not deleted"


def test_nan_is_not_saved():
    broken = torch.nn.Linear(2, 2)
    broken.weight.data.fill_(float("nan"))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            checkpoint.save(str(Path(tmp) / "a.pt"), 1, model=broken)
        except SystemExit:
            assert not list(Path(tmp).iterdir()), "NaN weights reached the disk"
            return
    raise AssertionError("NaN weights were saved")


if __name__ == "__main__":
    for test in (test_alignment, test_text, test_numbers, test_train_resume_and_synth, test_learns_alignment, test_vocoder_learns,
                 test_nan_is_not_saved, test_prepare_rasa):
        test()
        print("ok", test.__name__)
