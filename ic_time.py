"""Fixed HyperFlow timestep data shared by explicit setup and IC inference."""

import hashlib

import torch
import torch.nn.functional as F


IC_REPO = "Efficient-Large-Model/Open-H3-Regenerate-2K-IC"
IC_REVISION = "54a7104b204fb46486cf85c2f600942118186d57"
TIME_FORMAT = "olm_h3_ic_time_v1"
HYPERFLOW_FILE = "minimax_h3_hyperflow_8step_v1.0.safetensors"
HYPERFLOW_SHA256 = "9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447"
SILENCE_SHA256 = "dc90d8b8505f7160110b4122fb78348b232877e2928f7049c6833dcebde3f18c"
RAW_SIGMAS = (1., .931506, .839236, .703462, .5, .296538, .160764, .068494, 0.)


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_grid():
    raw = torch.tensor(RAW_SIGMAS, dtype=torch.float32)
    return 12 * raw / (1 + 11 * raw)


def time_points():
    grid = fixed_grid()
    return (torch.cat((1 - grid[:-1], torch.tensor([.999, 1.]))),
            torch.cat((1 - grid[1:], torch.tensor([.999, 1.]))))


def selector_table(times):
    table = torch.zeros(1025, len(times))
    for index, t in enumerate(times.tolist()):
        if t == 1.:
            table[1024, index] = 1.
        else:
            lo = min(1023, int(t * 1024))
            table[lo:lo + 2, index] = 1.
    pos = times * 1024
    lo = pos.floor().long().clamp(max=1023)
    values = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))
    if not torch.equal(values, torch.eye(len(times))):
        raise ValueError("HyperFlow timestep selectors overlap")
    return table


def hyperflow_features(base, time_lora, metadata):
    current = {key: value.float().clone() for key, value in base.items()}
    endpoint = {key: value.clone() for key, value in current.items()}
    scale = float(metadata["lora_alpha"]) / float(metadata["lora_rank"])
    for prefix, weights in (("time_embedder", current), ("endpoint_time_embedder", endpoint)):
        for layer in ("linear_1", "linear_2"):
            root = f"transformer.{prefix}.{layer}"
            weights[f"time_embedder.{layer}.weight"].add_(
                time_lora[root + ".lora_B.weight"].float() @ time_lora[root + ".lora_A.weight"].float(), alpha=scale)
    times, ends = time_points()

    def embed(t, weights):
        freqs = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(128, dtype=torch.float32) / 128)
        angles = t[:, None] * freqs[None]
        x = torch.cat((angles.cos(), angles.sin()), dim=-1)
        x = F.silu(F.linear(x, weights["time_embedder.linear_1.weight"], weights["time_embedder.linear_1.bias"]))
        return F.linear(x, weights["time_embedder.linear_2.weight"], weights["time_embedder.linear_2.bias"])

    gate = float(metadata["hyperflow_gate"])
    return F.silu((1 - gate) * embed(times, current) + gate * embed(ends, endpoint))
