"""IC reference conditioning and fixed HyperFlow refinement on native ComfyUI H3."""

import copy
import hashlib
import logging
import math

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from comfy import lora, model_management, sd, utils
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

from .ic_time import (HYPERFLOW_FILE, HYPERFLOW_SHA256, IC_REPO, IC_REVISION,
                      SILENCE_SHA256, TIME_FORMAT, fixed_grid, selector_table,
                      tensor_digest, time_points)
from .loading import CURVE_SHA256, lora_patches
from .runtime import (Refiner, aligned_frames, output_memory_gib,
                      resampled_frame_indices, sample_latents, validate_canvas, window_plan)


logger = logging.getLogger(__name__)
TASK_PROMPT = ("Enhance <Video 1> into a high-quality version of the same video. "
               "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
               "Restore fine details and remove degradation. Do not continue or change the scene.")


def load_time_data(path):
    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        expected = {"format": TIME_FORMAT, "source_repo": IC_REPO, "source_revision": IC_REVISION,
                    "hyperflow_sha256": HYPERFLOW_SHA256}
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("IC timestep data is incompatible; rebuild with scripts/prepare_ic.py")
        names = {f"blocks.{i}.adaln_proj.linear.weight" for i in range(50)}
        names.add("final_layer.adaln_proj.linear.weight")
        if set(source.keys()) != names | {"table", "sigmas"}:
            raise ValueError("IC timestep data must contain all 51 projections and the fixed grid")
        projections = {}
        for name in names:
            value = source.get_tensor(name)
            rows = 10752 if name.startswith("final_layer.") else 96768
            if value.shape != (rows, 10) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid IC timestep projection: {name}")
            projections[name] = value
        table, grid = source.get_tensor("table"), source.get_tensor("sigmas")
    times, _ = time_points()
    if not torch.equal(grid, fixed_grid()) or not torch.equal(table, selector_table(times)):
        raise ValueError("IC timestep grid differs from the published HyperFlow schedule")
    return projections, table, grid, metadata


def ic_sigmas(grid, sampler):
    if sampler not in ("up3", "up4"):
        raise ValueError("Select the published IC up3 or up4 sampler")
    return grid[-(4 if sampler == "up3" else 5):].clone()


def time_projection_patches(projections, state, sharp, strength):
    patches = {}
    for name, compiled in projections.items():
        key = "diffusion_model." + name
        original = state[key].float().cpu()
        if tuple(original.shape) != (compiled.shape[0], 8):
            raise ValueError(f"IC requires native 8-column timestep projections: {key}")
        full = torch.zeros(compiled.shape[0], 18)
        full[:, :10] = compiled
        root = key.removesuffix(".weight")
        if root + ".lora_A.weight" in sharp:
            full[:, 10:] = (sharp[root + ".lora_B.weight"].float()
                            @ sharp[root + ".lora_A.weight"].float()) * strength
        full[:, :8] -= original
        patches[key] = ("diff", (full, {"pad_weight": True}))
        bias = key.removesuffix("weight") + "bias"
        patches[bias] = ("diff", (-state[bias].float().cpu(),))
    return patches


def apply_ic(model, directory, sharp_strength, ic_strength=1.):
    if not math.isfinite(sharp_strength) or sharp_strength < 0:
        raise ValueError("Sharp strength must be finite and nonnegative")
    if not math.isfinite(ic_strength) or ic_strength < 0:
        raise ValueError("IC LoRA strength must be finite and nonnegative")
    native = model.get_model_object("diffusion_model")
    if not isinstance(native, MiniMaxH3Model) or not native.use_adaln_curves:
        raise ValueError("IC requires a native pruned MiniMax-H3 Ref2VA model")
    original_table = native.adaln_t_table.detach().float().cpu()
    if (original_table.shape != (1025, 8)
            or hashlib.sha256(original_table.contiguous().numpy().tobytes()).hexdigest() != CURVE_SHA256):
        raise ValueError("IC requires the published native Ref2VA timestep basis; check the selected base model")
    projections, selectors, sigmas, metadata = load_time_data(directory / "ic_time.safetensors")
    state = model.model.state_dict()
    result = model.clone()
    for filename in (HYPERFLOW_FILE, "open_h3_regenerate_2k_ic_lora.safetensors"):
        strength = 1. if filename == HYPERFLOW_FILE else ic_strength
        if strength == 0:
            continue
        model_management.throw_exception_if_processing_interrupted()
        raw = utils.load_torch_file(str(directory / "lora" / filename), safe_load=True)
        if filename == HYPERFLOW_FILE:
            time_lora = {k: v for k, v in raw.items() if "time_embedder." in k}
            if len(time_lora) != 8 or tensor_digest(time_lora) != metadata.get("time_lora_sha256"):
                raise ValueError("HyperFlow's timestep adapters do not match the prepared IC timestep data")
            raw = {k: v for k, v in raw.items() if k not in time_lora}
            del time_lora
        patches = lora_patches(raw, state)
        if len(patches) != 312 or len(result.add_patches(patches, strength_patch=strength)) != 312:
            raise ValueError(f"IC could not apply all 312 ordinary adapters from {filename}")
        del raw, patches
    sharp = load_file(str(directory / "lora/minimax_h3_lms_v1.0_r64.safetensors")) if sharp_strength else {}
    if sharp:
        linears = {k: v for k, v in sharp.items() if ".adaln_proj.linear." not in k}
        keymap = {k.removesuffix(".lora_A.weight"): k.removesuffix(".lora_A.weight") + ".weight"
                  for k in linears if k.endswith(".lora_A.weight")}
        patches = lora.load_lora(linears, keymap, log_missing=False)
        if len(patches) != 208 or len(result.add_patches(patches, strength_patch=sharp_strength)) != 208:
            raise ValueError("IC could not apply all 208 ordinary Sharp adapters")
        del linears, patches
    patches = time_projection_patches(projections, state, sharp, sharp_strength)
    if len(result.add_patches(patches)) != 102:
        raise ValueError("IC could not apply all timestep projection and bias patches")
    result.add_object_patch("diffusion_model.adaln_t_table", torch.cat((selectors, original_table), dim=1))
    sampling = copy.deepcopy(model.get_model_object("model_sampling"))
    # Fully masked audio stays in its clean native units with equal stream shifts.
    sampling.set_parameters(shift=12., audio_shift=12.)
    result.add_object_patch("model_sampling", sampling)
    options = result.model_options["transformer_options"]
    options["minimax_h3_sigma_shift_video"] = 12.
    options["minimax_h3_sigma_shift_audio"] = 12.
    return result, sigmas


def silent_audio(directory, context):
    path = directory / "assets/silence_audio_124f.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SILENCE_SHA256:
        raise ValueError("IC requires the publisher's silence_audio_124f.safetensors")
    audio = load_file(str(path))["audio_latents"]
    if audio.shape != (2, 32, 207) or not 5 <= context <= 124:
        raise ValueError("IC silence conditioning supports the 5-124-frame context")
    return audio.permute(1, 0, 2).unsqueeze(0)[..., :round(context / 24 * 40)].float().clone()


class ICRefiner(Refiner):
    def reference(self, images, prompt, width, height, input_size, context):
        pixels = self.prepare_images(images, width, height, input_size, context)
        options = {}
        if self.text_encoder_device == "cpu":
            options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
        clip = sd.load_clip([str(self.paths["text_encoder"])], clip_type=sd.CLIPType.MINIMAX, model_options=options)
        self.patchers["text_encoder"] = clip.patcher
        try:
            description = TASK_PROMPT + ("\n\nScene description: " + prompt.strip() if prompt.strip() else "")
            result = MiniMaxH3ReferenceToVideo.execute(
                clip, description, width, height, context, vae=self.video_vae(), ref_videos={"ref_video_1": pixels})
            cond = result.result[0]
            return cond, cond[0][1]["minimax_refs"][0]["latent"].float().cpu()
        finally:
            self.release("text_encoder")
            self.offload("video_vae")

    def sample_ic(self, video, conditioning, context, seed, sharp_strength, sparse,
                  ic_strength=1., sampler="up3"):
        audio = silent_audio(self.paths["adapters"], context)
        model_management.throw_exception_if_processing_interrupted()
        model = sd.load_diffusion_model(str(self.paths["base_model"]), model_options={"dtype": torch.bfloat16})
        self.patchers["model"] = model
        try:
            model, grid = apply_ic(model, self.paths["adapters"], sharp_strength, ic_strength)
            self.patchers["model"] = model
            return sample_latents(model, conditioning, video, audio, seed, ic_sigmas(grid, sampler), sparse)
        finally:
            self.release("model")

    def frames(self, images, prompt, fps, width, height, seed, sharp_strength,
               input_size="768p", window_frames=121, sparse=True, ic_strength=1., sampler="up3"):
        validate_canvas(width, height)
        if images.ndim != 4 or images.shape[-1] not in (3, 4) or not images.is_floating_point():
            raise ValueError("Provide an RGB/RGBA IMAGE batch of floating-point video frames")
        if not math.isfinite(sharp_strength) or sharp_strength < 0:
            raise ValueError("Sharp strength must be finite and nonnegative")
        if not math.isfinite(ic_strength) or ic_strength < 0:
            raise ValueError("IC LoRA strength must be finite and nonnegative")
        ic_sigmas(fixed_grid(), sampler)
        indices = resampled_frame_indices(len(images), fps)
        count = len(indices)
        plan = list(window_plan(count, window_frames))
        context = aligned_frames(window_frames)
        info = (f"IC {sampler}, strength {ic_strength:g}, Sharp {sharp_strength:g}: "
                f"{count} frames at 24 fps (input: {len(images)} at {fps:g} fps); {len(plan)} window(s); "
                f"output IMAGE alone {output_memory_gib(count, width, height):.2f} GiB. Model/working memory is additional.")
        logger.info("Open-H3 %s", info)
        output = torch.empty((count, height, width, 3), dtype=torch.float32,
                             device=model_management.intermediate_device())
        progress = utils.ProgressBar(len(plan))
        for i, (start, stop, context_stop) in enumerate(plan):
            model_management.throw_exception_if_processing_interrupted()
            logger.info("Open-H3 IC window %d/%d: frames %d-%d", i + 1, len(plan), start, stop - 1)
            pixels = images[start:context_stop] if fps == 24.0 else images[indices[start:context_stop]]
            cond, reference = self.reference(pixels, prompt, width, height, input_size, context)
            del pixels
            video = self.upscale(reference, width, height)
            refined = self.sample_ic(video, cond, context, (seed + i) % 2**64, sharp_strength, sparse,
                                     ic_strength, sampler)
            decoded = self.decode(refined, width, height)
            output[start:stop].copy_(decoded[:stop - start])
            del cond, reference, video, refined, decoded
            progress.update_absolute(i + 1)
        return output, info
