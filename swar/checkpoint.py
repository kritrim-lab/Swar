"""One checkpoint file per training run, mirrored to Backblaze B2 when its variables are set.

B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET turn it on. B2_PREFIX (default
"checkpoints") is the folder in the bucket. Keys are read from the environment only.
"""
import os
import time

import torch

B2_VARIABLES = ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET")
KEEP_VERSIONS = 2  # per checkpoint in B2: the newest, and one to fall back on


def b2_api():
    from b2sdk.v2 import B2Api, InMemoryAccountInfo

    api = B2Api(InMemoryAccountInfo())
    api.authorize_account("production", os.environ["B2_APPLICATION_KEY_ID"], os.environ["B2_APPLICATION_KEY"])
    return api


def connect():
    """The B2 bucket, or None when B2 is not configured. Bad keys fail here, before training."""
    missing = [name for name in B2_VARIABLES if not os.environ.get(name)]
    if len(missing) == len(B2_VARIABLES):
        print("B2 is OFF: checkpoints are saved on this machine only")
        return None
    if missing:
        raise SystemExit(f"B2 is half configured, missing: {', '.join(missing)}")
    return b2_api().get_bucket_by_name(os.environ["B2_BUCKET"])


def remote_name(path):
    prefix = os.environ.get("B2_PREFIX", "checkpoints").strip("/")
    return "/".join(part for part in (prefix, os.path.basename(path)) if part)


def pull(bucket, path):
    """Download the B2 checkpoint when it is further along than the local one, or there is none."""
    name = remote_name(path)
    # Listing, not get_file_info_by_name: B2 refuses that one once the daily download cap is used up.
    # Any error here stops the run: starting from zero would later overwrite the good copy.
    versions = (v for v in bucket.list_file_versions(name) if v.action != "start")  # "start": unfinished upload
    info = max(versions, key=lambda v: v.upload_timestamp, default=None)
    if info is None or info.action == "hide":
        print(f"B2: no {name} yet")
        return
    remote_step = int(info.file_info.get("step", -1))
    if os.path.exists(path):
        local_step = torch.load(path, map_location="cpu", weights_only=True)["step"]
        if local_step >= remote_step:
            print(f"B2: keeping local step {local_step} (B2 has step {remote_step})")
            return
    print(f"B2: downloading {name} step {remote_step} ({info.size / 1e6:.0f} MB)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bucket.download_file_by_id(info.id_).save_to(path + ".tmp")
    os.replace(path + ".tmp", path)


def push(bucket, path, step, final=False):
    name = remote_name(path)
    started = time.time()
    try:
        uploaded = bucket.upload_local_file(path, name, file_info={"step": str(step)})
    except Exception as error:
        if final:
            raise SystemExit(f"B2 UPLOAD FAILED: step {step} exists only at {path}\n{error}")
        print(f"WARNING: B2 upload of step {step} failed, retrying at the next save: {error}")
        return
    print(f"B2: uploaded {name} step {step} ({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - started:.0f} s)")
    # B2 keeps every version of a file; without pruning a long run fills the account
    try:
        versions = [v for v in bucket.list_file_versions(name) if v.file_name == name]
        older = sorted((v for v in versions if v.id_ != uploaded.id_), key=lambda v: v.upload_timestamp, reverse=True)
        for version in older[KEEP_VERSIONS - 1:]:
            bucket.api.delete_file_version(version.id_, version.file_name)
    except Exception as error:
        print(f"WARNING: could not delete old B2 versions of {name}: {error}")


def save(path, step, bucket=None, final=False, **parts):
    """parts are anything with state_dict(): models, optimizers, scaler."""
    for name, part in parts.items():
        if isinstance(part, torch.nn.Module) and not all(p.isfinite().all() for p in part.parameters()):
            raise SystemExit(f"{name} has NaN or infinite weights at step {step}; kept the last good checkpoint")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {name: part.state_dict() for name, part in parts.items()}
    torch.save({**state, "step": step}, path + ".tmp")
    os.replace(path + ".tmp", path)  # a session killed mid-save keeps the old checkpoint
    # shortcut: training waits for the upload; move it to a thread if the printed times hurt
    if bucket is not None:
        push(bucket, path, step, final)


def load(path, device="cpu", **parts):
    """Fills the given parts in place and returns the step."""
    state = torch.load(path, map_location=device, weights_only=True)  # never unpickle arbitrary code
    for name, part in parts.items():
        # a scaler saved on CPU is empty, and early vocoder checkpoints have none at all
        saved = state.get(name) if name == "scaler" else state[name]
        if saved:
            part.load_state_dict(saved)
    return state["step"]
