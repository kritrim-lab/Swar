"""Quick correctness checks on fake data, no GPU or dataset needed: python test_swar.py"""
import contextlib
import importlib.util
import io
import itertools
import math
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch

import train
import train_vocoder
from swar import checkpoint
from swar.data import HOP, SR, text_to_ids, wav_to_mel
from swar.model import monotonic_alignment_search


# the tests must never touch a real bucket, even in a notebook where the keys are set
for variable in (*checkpoint.B2_VARIABLES, "B2_PREFIX"):
    os.environ.pop(variable, None)


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
        assert torch.load(ckpt, weights_only=True)["step"] == 3, "did not resume"

        voc = str(tmp / "ckpt" / "v.pt")
        args = ["--data", str(tmp), "--ckpt", voc, "--batch-size", "2", "--workers", "0"]
        train_vocoder.main(args + ["--steps", "2"])
        train_vocoder.main(args + ["--steps", "3"])
        assert torch.load(voc, weights_only=True)["step"] == 3, "vocoder did not resume"

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
    sounds = torch.randn(20, 80)  # one fixed sound per character
    clips = [sounds[i].repeat_interleave(d, dim=0) for i, d in zip(ids, truth)]
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

    # the sampler must turn noise into the right sounds, end to end
    model.eval()
    text_mask = torch.ones(ids.shape, dtype=torch.bool)
    frame_mask = torch.arange(mels.shape[2])[None] < mel_lens[:, None]
    with torch.no_grad():
        condition = align.transpose(1, 2) @ model.encode(ids, text_mask)[0]
    sampled = model.sample(condition, frame_mask)
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


def expect_exit(function):
    try:
        function()
    except (SystemExit, ConnectionError) as error:
        return str(error)
    raise AssertionError("should have stopped")


def test_backblaze():
    """Against b2sdk's in-memory fake of B2: no account, keys or network."""
    if importlib.util.find_spec("b2sdk") is None:
        print("b2sdk not installed, skipping")
        return
    from b2sdk.v2 import B2Api, B2HttpApiConfig, Bucket, InMemoryAccountInfo, RawSimulator

    from swar.model import AcousticModel

    api = B2Api(InMemoryAccountInfo(), api_config=B2HttpApiConfig(_raw_api_class=RawSimulator))
    api.authorize_account("production", *api.session.raw_api.create_account())
    api.create_bucket("swar-test", "allPrivate")
    real = checkpoint.b2_api, train.AcousticModel, Bucket.get_file_info_by_name, Bucket.upload_local_file
    checkpoint.b2_api = lambda: api
    train.AcousticModel = lambda: AcousticModel(dim=16, encoder_layers=1, decoder_layers=1)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sf.write(tmp / "0.wav", 0.1 * torch.randn(SR).numpy(), SR)
            (tmp / "manifest.txt").write_text("0.wav|नमस्ते", encoding="utf-8")
            ckpt = tmp / "run" / "a.pt"

            def run(steps):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    train.main(["--data", str(tmp / "manifest.txt"), "--ckpt", str(ckpt), "--steps", str(steps),
                                "--batch-size", "1", "--workers", "0", "--save-every", "1"])
                return output.getvalue()

            def local_step():
                return torch.load(ckpt, weights_only=True)["step"]

            def remote_steps():
                bucket = api.get_bucket_by_name("swar-test")
                return [int(v.file_info["step"]) for v in bucket.list_file_versions("checkpoints/a.pt")]

            os.environ["B2_BUCKET"] = "swar-test"
            expect_exit(lambda: run(1))  # half configured
            os.environ.update(B2_APPLICATION_KEY_ID="fake", B2_APPLICATION_KEY="fake")

            run(2)
            assert remote_steps() == [2, 1]
            shutil.copy(ckpt, tmp / "step2.pt")

            ckpt.unlink()  # a fresh machine
            assert "at step 2" in run(4), "did not resume from the B2 copy"
            assert remote_steps() == [4, 3], "should keep only the two newest"

            shutil.copy(tmp / "step2.pt", ckpt)  # a stale copy, e.g. left on Google Drive
            assert "at step 4" in run(5), "stale local file won over B2"

            Bucket.upload_local_file = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline"))
            assert "UPLOAD FAILED" in expect_exit(lambda: run(7))  # mid-run warns, the end must fail
            assert local_step() == 7 and remote_steps() == [5, 4]
            Bucket.upload_local_file = real[3]
            assert "keeping local step 7" in run(8), "local progress must survive a failed upload"
            assert remote_steps()[0] == 8

            ckpt.unlink()
            Bucket.get_file_info_by_name = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline"))
            expect_exit(lambda: run(9))  # cannot check B2: stop, never start from zero
            assert not ckpt.exists() and remote_steps()[0] == 8

            broken = torch.nn.Linear(2, 2)
            broken.weight.data.fill_(float("nan"))
            assert "NaN" in expect_exit(lambda: checkpoint.save(str(ckpt), 9, model=broken))
    finally:
        checkpoint.b2_api, train.AcousticModel, Bucket.get_file_info_by_name, Bucket.upload_local_file = real
        for variable in checkpoint.B2_VARIABLES:
            os.environ.pop(variable, None)


if __name__ == "__main__":
    for test in (test_alignment, test_text, test_train_resume_and_synth, test_learns_alignment, test_vocoder_learns, test_backblaze):
        test()
        print("ok", test.__name__)
