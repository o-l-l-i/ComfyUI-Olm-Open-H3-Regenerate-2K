"""Map the released Open-H3 adapters onto native ComfyUI H3 weights."""

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import folder_paths
import torch

from comfy import lora, utils
from comfy.ldm.minimax.model import MiniMaxH3Model


CURVE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
ROOT = Path(__file__).resolve().parent
MODEL_CATEGORIES = {"base_model": "diffusion_models", "text_encoder": "text_encoders",
                    "video_vae": "vae", "audio_vae": "vae", "upscaler": "open_h3_upscalers"}


def register_model_paths():
    folder_paths.add_model_folder_path("open_h3_refiner", str(Path(folder_paths.models_dir) / "open_h3_refiner"))
    folder_paths.add_model_folder_path("open_h3_upscalers", str(Path(folder_paths.models_dir) / "latent_upscale_models"))
    for category in ("latent_upscale_models", "diffusion_models"):
        if category in folder_paths.folder_names_and_paths:
            for path in folder_paths.get_folder_paths(category):
                folder_paths.add_model_folder_path("open_h3_upscalers", path)
    paths, extensions = folder_paths.folder_names_and_paths["open_h3_upscalers"]
    folder_paths.folder_names_and_paths["open_h3_upscalers"] = (paths, extensions | {".safetensors"})


def model_names(kind):
    tokens = {"base_model": ("minimax_h3", "fl2va", "pruned"),
              "text_encoder": ("qwen3vl", "minimax_h3"), "video_vae": ("minimax_h3", "video_vae"),
              "audio_vae": ("minimax_h3", "audio_vae"), "upscaler": ("minimax_h3_latent_upscaler_3d",)}
    return [name for name in folder_paths.get_filename_list(MODEL_CATEGORIES[kind])
            if name.endswith(".safetensors") and all(t in Path(name).name.lower() for t in tokens[kind])]


def resolve_model(kind, name):
    category = MODEL_CATEGORIES[kind]
    path = Path(folder_paths.get_full_path_or_raise(category, name)).resolve()
    if not path.is_file() or not any(path.is_relative_to(Path(root).resolve())
                                    for root in folder_paths.get_folder_paths(category)):
        raise ValueError(f"Select a {kind} file inside its registered model paths")
    return path


def bundle_roots():
    roots = folder_paths.get_folder_paths("open_h3_refiner") + folder_paths.get_folder_paths("diffusers")
    return list(dict.fromkeys(Path(p).resolve() for p in [*roots, ROOT / "weights"]))


def bundle_names():
    return sorted({p.name for root in bundle_roots() if root.is_dir() for p in root.iterdir()
                   if p.is_dir() and (p / "lora/open_h3_regenerate_2k_lora.safetensors").is_file()
                   and (p / "lora/sharp_curve_branches.safetensors").is_file()})


def resolve_bundle(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Select an Open-H3 adapter folder from the loader list")
    for root in bundle_roots():
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            continue
        files = [directory / "lora" / f for f in
                 ("open_h3_regenerate_2k_lora.safetensors", "sharp_curve_branches.safetensors", "adaln_t_table.bin")]
        if all(p.is_file() and p.resolve().is_relative_to(directory) for p in files):
            return directory / "lora"
    raise FileNotFoundError(
        "Open-H3 adapter bundle needs lora/open_h3_regenerate_2k_lora.safetensors, "
        "lora/sharp_curve_branches.safetensors and lora/adaln_t_table.bin. "
        "Adapters: https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K . "
        "For the table, run scripts/fetch_sharp_table.py --help."
    )


@dataclass(frozen=True)
class ModelSelection:
    base_model: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    upscaler: str
    adapters: str
    text_encoder_device: str = "auto"

    def resolve(self):
        paths = {kind: resolve_model(kind, getattr(self, kind)) for kind in MODEL_CATEGORIES}
        paths["adapters"] = resolve_bundle(self.adapters)
        return paths


def adapter_target(name, rows):
    if name.startswith("transformer_blocks."):
        name = name.replace("transformer_blocks.", "blocks.", 1)
    elif name.startswith("token_refiner.refiner_blocks."):
        name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
    else:
        raise ValueError(f"Unexpected Open-H3 adapter module: {name}")
    for index, suffix in enumerate((".attn.to_q", ".attn.to_k", ".attn.to_v")):
        if name.endswith(suffix):
            return ("diffusion_model." + name.removesuffix(suffix) + ".attn.qkv_proj.weight",
                    (0, index * rows, rows))
    for source, target in ((".attn.to_out.0", ".attn.out_proj"),
                           (".ff.net.0.proj", ".mlp.fc1"), (".ff.net.2", ".mlp.fc2")):
        if name.endswith(source):
            return "diffusion_model." + name.removesuffix(source) + target + ".weight"
    raise ValueError(f"Unexpected Open-H3 adapter module: {name}")


def lora_patches(state, native_state):
    key_map = {}
    consumed = set()
    for key, down in state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        base = key.removesuffix(".lora_A.weight")
        up_key = base + ".lora_B.weight"
        up = state[up_key]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            raise ValueError(f"Invalid Open-H3 LoRA pair: {base}")
        destination = adapter_target(base.removeprefix("transformer."), up.shape[0])
        native_key = destination[0] if isinstance(destination, tuple) else destination
        shape = native_state[native_key].shape
        expected = (up.shape[0] * (3 if isinstance(destination, tuple) else 1), down.shape[1])
        if tuple(shape) != expected:
            raise ValueError(f"Open-H3 adapter does not match {native_key}: {tuple(shape)} != {expected}")
        key_map[base] = destination
        consumed.update((key, up_key))
    if consumed != state.keys():
        raise ValueError("Expected only paired Open-H3 lora_A / lora_B weights")
    return lora.load_lora(state, key_map, log_missing=False)


def curve_patches(state, native_state):
    patches = {}
    consumed = set()
    for key, down in state.items():
        if not key.endswith(".curve_A.weight"):
            continue
        base = key.removeprefix("transformer.").removesuffix(".curve_A.weight")
        up_key = key.replace(".curve_A.weight", ".curve_B.weight")
        up = state[up_key]
        destination = "diffusion_model." + base.replace("transformer_blocks.", "blocks.", 1) + ".weight"
        if down.ndim != 2 or up.ndim != 2 or down.shape[1] != 8 or down.shape[0] != up.shape[1]:
            raise ValueError(f"Invalid Sharp curve pair: {base}")
        if tuple(native_state[destination].shape) != (up.shape[0], 8):
            raise ValueError(f"Sharp curve branch does not match {destination}")
        delta = up.float() @ down.float()
        # The native FL2VA basis and Sharp's Ref2VA basis are different. Keep both.
        patches[destination] = ("diff", (torch.cat((torch.zeros_like(delta), delta), dim=1),
                                         {"pad_weight": True}))
        consumed.update((key, up_key))
    if consumed != state.keys():
        raise ValueError("Expected only paired Sharp curve_A / curve_B weights")
    return patches


def apply_refiner(model, directory):
    native = model.get_model_object("diffusion_model")
    if not isinstance(native, MiniMaxH3Model) or not native.use_adaln_curves:
        raise ValueError("Open-H3 requires a native pruned MiniMax-H3 FL2VA model")
    directory = Path(directory)
    table_bytes = (directory / "adaln_t_table.bin").read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != CURVE_SHA256:
        raise ValueError("Sharp's adaln_t_table.bin does not match the publisher's hash")
    sharp_table = torch.frombuffer(bytearray(table_bytes), dtype=torch.float32).reshape(1025, 8)
    native_table = native.adaln_t_table.detach().to(device="cpu", dtype=torch.float32)
    if native_table.shape != sharp_table.shape:
        raise ValueError("The native H3 timestep table must have shape [1025, 8]")
    native_state = model.model.state_dict()
    state = utils.load_torch_file(str(directory / "open_h3_regenerate_2k_lora.safetensors"), safe_load=True)
    patches = lora_patches(state, native_state)
    if len(patches) != 312:
        raise ValueError("The released Open-H3 adapter must contain all 312 LoRA pairs")
    del state
    state = utils.load_torch_file(str(directory / "sharp_curve_branches.safetensors"), safe_load=True)
    branches = curve_patches(state, native_state)
    if len(branches) != 50:
        raise ValueError("The released Open-H3 adapter must contain all 50 Sharp curve branches")
    patches.update(branches)
    final_key = "diffusion_model.final_layer.adaln_proj.linear.weight"
    final_weight = native_state[final_key]
    patches[final_key] = ("diff", (torch.zeros(final_weight.shape[0], 16), {"pad_weight": True}))
    result = model.clone()
    if len(result.add_patches(patches)) != 363:
        raise ValueError("ComfyUI did not accept every Open-H3 adapter patch")
    result.add_object_patch("diffusion_model.adaln_t_table", torch.cat((native_table, sharp_table), dim=1))

    # Equal stream scales keep the masked audio latent fixed in its native units.
    sampling = copy.deepcopy(model.get_model_object("model_sampling"))
    sampling.set_parameters(shift=6.0, audio_shift=6.0)
    result.add_object_patch("model_sampling", sampling)
    options = result.model_options["transformer_options"]
    options["minimax_h3_sigma_shift_video"] = 6.0
    options["minimax_h3_sigma_shift_audio"] = 6.0
    return result
