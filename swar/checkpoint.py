"""One checkpoint file per training run."""
import os

import torch


def save(path, step, **parts):
    """parts are anything with state_dict(): models, optimizers, scaler."""
    for name, part in parts.items():
        if isinstance(part, torch.nn.Module) and not all(p.isfinite().all() for p in part.parameters()):
            raise SystemExit(f"{name} has NaN or infinite weights at step {step}; kept the last good checkpoint")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {name: part.state_dict() for name, part in parts.items()}
    torch.save({**state, "step": step}, path + ".tmp")
    os.replace(path + ".tmp", path)  # a session killed mid-save keeps the old checkpoint


def load(path, device="cpu", **parts):
    """Fills the given parts in place and returns the step."""
    state = torch.load(path, map_location=device, weights_only=True)  # never unpickle arbitrary code
    for name, part in parts.items():
        # a scaler saved on CPU is empty; older checkpoints have no "ema" or no scaler at all
        saved = state.get(name) if name in ("scaler", "ema") else state[name]
        if saved:
            part.load_state_dict(saved)
    return state["step"]
