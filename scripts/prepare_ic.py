"""Prepare reusable IC timestep data for the native pruned Ref2VA model.

Explicit setup only: reads ~26 GB of selected original checkpoint tensors once,
then keeps ~185 MiB of timestep data. Nodes never download or convert weights.
Only torch and safetensors from the ComfyUI environment are needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.request

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ic_time import (HYPERFLOW_FILE, HYPERFLOW_SHA256, IC_REPO, IC_REVISION,
                     TIME_FORMAT, fixed_grid, hyperflow_features, selector_table,
                     tensor_digest, time_points)


BASE_PATH = "base_model/transformer_ref"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
ITEM_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def remote_range(url, start, length):
    end = start + length - 1
    request = urllib.request.Request(url + f"?range={start}-{end}", headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("Server ignored the byte range; refusing a full checkpoint download")
        result = bytearray()
        while len(result) < length:
            chunk = response.read(min(16 * 1024**2, length - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) != length or response.read(1):
            raise RuntimeError("Incomplete or oversized checkpoint byte range")
    return result


class Source:
    def __init__(self, directory=None):
        self.directory = directory.resolve() if directory else None
        self.root = f"https://huggingface.co/{IC_REPO}/resolve/{IC_REVISION}/{BASE_PATH}"
        if directory:
            index = (self.directory / "diffusion_pytorch_model.safetensors.index.json").read_bytes()
        else:
            with urllib.request.urlopen(self.root + "/diffusion_pytorch_model.safetensors.index.json", timeout=60) as response:
                index = response.read(2 * 1024**2 + 1)
            if len(index) > 2 * 1024**2:
                raise ValueError("Unexpected checkpoint index size")
        self.index = json.loads(index)["weight_map"]
        self.headers = {}

    def read(self, shard, start, length):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard name")
        if self.directory:
            path = (self.directory / shard).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("Checkpoint shard escapes its source directory")
            with path.open("rb") as source:
                source.seek(start)
                result = source.read(length)
            if len(result) != length:
                raise ValueError("Truncated local checkpoint")
            return bytearray(result)
        return remote_range(self.root + "/" + shard, start, length)

    def tensor(self, key):
        shard = self.index[key]
        if shard not in self.headers:
            size, = struct.unpack("<Q", self.read(shard, 0, 8))
            if not 2 <= size <= 8 * 1024**2:
                raise ValueError("Invalid safetensors header size")
            self.headers[shard] = size, json.loads(self.read(shard, 8, size))
        size, header = self.headers[shard]
        spec = header[key]
        start, end = spec["data_offsets"]
        elements = 1
        for dim in spec["shape"]:
            if not isinstance(dim, int) or dim < 1:
                raise ValueError("Invalid source tensor shape")
            elements *= dim
        if (spec["dtype"] not in DTYPES or start < 0 or not 0 < end - start < 2**31
                or end - start != elements * ITEM_BYTES[spec["dtype"]]):
            raise ValueError(f"Invalid source tensor: {key}")
        raw = self.read(shard, 8 + size + start, end - start)
        return torch.frombuffer(raw, dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"])


def prepare(bundle, source_directory=None):
    bundle = bundle.resolve()
    output = bundle / "ic_time.safetensors"
    if output.exists():
        raise FileExistsError(f"Already exists; retain or move it explicitly before rebuilding: {output}")
    hyperflow = bundle / "lora" / HYPERFLOW_FILE
    if file_sha256(hyperflow) != HYPERFLOW_SHA256:
        raise ValueError("Select the published HyperFlow 8-step v1.0 adapter; checksum mismatch")
    with safe_open(str(hyperflow), framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
        time_lora = {key: weights.get_tensor(key) for key in weights.keys() if "time_embedder." in key}
    source = Source(source_directory)
    base = {f"time_embedder.linear_{i}.{kind}": source.tensor(f"time_embedder.linear_{i}.{kind}").float()
            for i in (1, 2) for kind in ("weight", "bias")}
    features = hyperflow_features(base, time_lora, metadata)
    signature = tensor_digest({"features": features})
    cache = bundle / ".ic_time_cache"
    cache.mkdir(exist_ok=True)
    projection_metadata = {"format": TIME_FORMAT, "source_revision": IC_REVISION, "features_sha256": signature}
    outputs = {}
    for i in range(51):
        name = f"transformer_blocks.{i}.adaln_proj.linear" if i < 50 else "norm_out.linear"
        target = f"blocks.{i}.adaln_proj.linear.weight" if i < 50 else "final_layer.adaln_proj.linear.weight"
        path = cache / f"{i:02}.safetensors"
        if path.exists():
            with safe_open(str(path), framework="pt", device="cpu") as cached:
                if cached.metadata() != projection_metadata:
                    raise ValueError(f"Incompatible conversion cache: {path}")
                value = cached.get_tensor("weight")
        else:
            weight = source.tensor(name + ".weight").float()
            bias = source.tensor(name + ".bias").float()
            value = F.linear(features, weight, bias).T.contiguous()
            del weight, bias
            temporary = path.with_suffix(".partial")
            save_file({"weight": value}, str(temporary), metadata=projection_metadata)
            temporary.replace(path)
        expected = (96768 if i < 50 else 10752, 10)
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError(f"Invalid compiled timestep projection: {target}")
        outputs[target] = value
        print(f"Projection {i + 1}/51 ready", flush=True)
    times, _ = time_points()
    outputs.update(table=selector_table(times), sigmas=fixed_grid())
    temporary = output.with_suffix(".partial")
    save_file(outputs, str(temporary), metadata={
        **projection_metadata, "source_repo": IC_REPO, "hyperflow_sha256": HYPERFLOW_SHA256,
        "time_lora_sha256": tensor_digest(time_lora)})
    temporary.replace(output)
    print(f"Prepared {output.stat().st_size / 1024**2:.1f} MiB: {output}")
    print("The small per-layer cache supports interrupted setup; it may now be removed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="IC bundle containing lora/ and assets/")
    parser.add_argument("--source-directory", type=Path, help="Existing original transformer_ref shards and index; otherwise use bounded HTTPS reads")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    prepare(args.bundle, args.source_directory)


if __name__ == "__main__":
    main()
