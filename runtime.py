"""Run the released short-step refinement with native ComfyUI sampling."""

import math
import logging

import torch
import torch.nn.functional as F

from comfy import audio as comfy_audio, model_management, model_patcher, sample, samplers, sd, utils
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention

from .loading import apply_refiner
from .upscaler import LatentUpscaler


SIGMAS = {
    3: (0.909375, 6 / 7, 2 / 3, 0.0),
    2: (0.7, 42 / 85, 0.0),
}
logger = logging.getLogger(__name__)


def aligned_frames(count):
    return 5 + max(0, math.ceil((count - 5) / 17)) * 17


def window_plan(count, window_frames):
    if count < 1 or window_frames not in (22, 56, 90, 121):
        raise ValueError("Provide frames and select a supported window length")
    context = aligned_frames(window_frames)
    start = 0
    while start < count:
        stop = min(count, start + (context if count - start <= context else window_frames))
        yield start, stop, min(count, start + context)
        start = stop


def input_canvas(width, height, input_size):
    short, pixels = {"576p": (576, 1024 * 576), "768p": (768, 1344 * 768)}[input_size]
    scale = min(short / min(width, height), math.sqrt(pixels / (width * height)))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def output_memory_gib(frames, width, height):
    return frames * width * height * 3 * 4 / 1024**3


def resampled_frame_indices(count, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Input fps must be finite and greater than zero; received {fps!r}")
    if count < 1:
        raise ValueError("Provide at least one video frame")
    if fps == 24.0:
        return range(count)
    output_count = max(1, math.floor(count * 24 / fps + 0.5))
    # Match frame centres without blending or motion interpolation.
    return [min(count - 1, math.floor((i + 0.5) * fps / 24)) for i in range(output_count)]


def validate_canvas(width, height):
    if min(width, height) < 256 or width % 32 or height % 32:
        raise ValueError("Output dimensions must be at least 256 and multiples of 32")


class Refiner:
    """Components belong to one execution; CPU/GPU weights are released afterward."""

    def __init__(self, paths, text_encoder_device="auto", vae_tile_size=256):
        if vae_tile_size not in (128, 192, 256, 384, 512):
            raise ValueError("Select a supported H3 VAE spatial tile size")
        self.paths = paths
        self.text_encoder_device = text_encoder_device
        self.vae_tile_size = vae_tile_size
        self.patchers = {}
        self.video = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def offload(self, name):
        if name in self.patchers:
            model_management.unload_model_and_clones(self.patchers[name])

    def release(self, name):
        self.offload(name)
        self.patchers.pop(name, None)

    def close(self):
        for name in list(self.patchers):
            self.release(name)
        self.video = None

    def load_vae(self, name):
        state, metadata = utils.load_torch_file(str(self.paths[name]), safe_load=True, return_metadata=True)
        vae = sd.VAE(sd=state, metadata=metadata)
        vae.throw_exception_if_invalid()
        self.patchers[name] = vae.patcher
        return vae

    def video_vae(self):
        if self.video is None:
            self.video = self.load_vae("video_vae")
            native = self.video.first_stage_model
            if not hasattr(native, "tile_overlap_min") or self.video.latent_channels != 24:
                raise ValueError("Select a native MiniMax-H3 video VAE")
            native.tile_size = self.vae_tile_size
            native.tile_overlap_min = min(64, self.vae_tile_size // 4 // 16 * 16)
        return self.video

    def conditioning(self, prompt, supplied=None):
        if supplied is not None:
            return supplied
        if not prompt.strip():
            raise ValueError("Describe the visible scene/action, or connect conditioning")
        model_management.throw_exception_if_processing_interrupted()
        options = {}
        if self.text_encoder_device == "cpu":
            options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
        clip = sd.load_clip([str(self.paths["text_encoder"])], clip_type=sd.CLIPType.MINIMAX, model_options=options)
        self.patchers["text_encoder"] = clip.patcher
        try:
            return clip.encode_from_tokens_scheduled(clip.tokenize(prompt.strip()))
        finally:
            self.release("text_encoder")

    def upscale(self, video, width, height):
        if video.shape[-2:] == (height // 16, width // 16):
            return video.clone()
        vae = self.video_vae()
        with torch.device("meta"):
            upscaler = LatentUpscaler(vae.first_stage_model.latents_mean, vae.first_stage_model.latents_std)
        upscaler.load_state_dict(utils.load_torch_file(str(self.paths["upscaler"]), safe_load=True), strict=True, assign=True)
        patcher = model_patcher.ModelPatcher(upscaler, model_management.get_torch_device(), torch.device("cpu"))
        self.patchers["upscaler"] = patcher
        working = max(1024**3, video.shape[2] * (height // 16) * (width // 16) * 512 * 2 * 12)
        try:
            model_management.load_models_gpu([patcher], memory_required=working, force_full_load=True)
            return upscaler(video.to(device=model_management.get_torch_device(), dtype=torch.bfloat16),
                            height // 16, width // 16).float().cpu()
        finally:
            self.release("upscaler")

    def sample(self, video, audio, conditioning, seed, steps, start_noise, sparse):
        if steps == 0:
            return video
        model_management.throw_exception_if_processing_interrupted()
        model = sd.load_diffusion_model(str(self.paths["base_model"]), model_options={"dtype": torch.bfloat16})
        self.patchers["model"] = model
        try:
            model = apply_refiner(model, self.paths["adapters"])
            self.patchers["model"] = model
            return refine_latents(model, conditioning, video, audio, seed, steps=steps,
                                  sparse=sparse, start_sigma=start_noise)
        finally:
            self.release("model")

    def decode(self, video, width, height):
        try:
            result = self.video_vae().decode(video).reshape(-1, height, width, 3)
            if not torch.isfinite(result).all():
                raise RuntimeError("Non-finite decoded H3 frames")
            return result.to(device=model_management.intermediate_device(), dtype=torch.float32)
        finally:
            self.offload("video_vae")

    def encode_images(self, images, width, height, input_size, context_frames):
        iw, ih = input_canvas(width, height, input_size)
        pixels = utils.common_upscale(images[..., :3].movedim(-1, 1).float().cpu(), iw, ih,
                                      "bilinear", "center").movedim(1, -1).clone()
        if not torch.isfinite(pixels).all():
            raise ValueError("Video contains non-finite values")
        if len(pixels) < context_frames:
            pixels = torch.cat([pixels, pixels[-1:].expand(context_frames - len(pixels), -1, -1, -1)])
        try:
            return self.video_vae().encode(pixels).float().cpu()
        finally:
            self.offload("video_vae")

    def encode_audio(self, audio, frame_count, context_frames, start=0):
        audio_t = round(context_frames / 24 * 40)
        if audio is None:
            wave = torch.zeros(1, 2, audio_t * 800)
        else:
            wave = audio["waveform"]
            if wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] < 1:
                raise ValueError("Audio must be a single waveform batch [1, channels, samples]")
            wave = comfy_audio.resample(wave[:1, :2].float().cpu(), audio["sample_rate"], 32000)
            if wave.shape[1] == 1:
                wave = wave.expand(-1, 2, -1)
            first, last = round(start / 24 * 32000), round((start + frame_count) / 24 * 32000)
            wave = wave[..., first:last].clone()
            wave = F.pad(wave, (0, max(0, audio_t * 800 - wave.shape[-1])))[..., :audio_t * 800]
        if not torch.isfinite(wave).all():
            raise ValueError("Audio contains non-finite samples")
        vae = self.load_vae("audio_vae")
        try:
            result = vae.encode(wave.movedim(1, -1)).float().cpu()
            if tuple(result.shape) != (1, 32, 2, audio_t):
                raise ValueError("Select the native MiniMax-H3 audio VAE")
            return result
        finally:
            self.release("audio_vae")

    def frames(self, images, prompt, fps, width, height, seed, steps, start_noise,
               input_size="768p", window_frames=121, sparse=True, conditioning=None, audio=None):
        validate_canvas(width, height)
        if images.ndim != 4 or images.shape[-1] not in (3, 4) or not images.is_floating_point():
            raise ValueError("Provide an RGB/RGBA IMAGE batch of floating-point video frames")
        indices = resampled_frame_indices(len(images), fps)
        count = len(indices)
        plan = list(window_plan(count, window_frames))
        context = aligned_frames(window_frames)
        gib = output_memory_gib(count, width, height)
        info = f"{count} frames at 24 fps (input: {len(images)} at {fps:g} fps); {len(plan)} window(s); output IMAGE alone {gib:.2f} GiB. Model/working memory is additional."
        logger.info("Open-H3: %s", info)
        cond = self.conditioning(prompt, conditioning) if steps else None
        output = torch.empty((count, height, width, 3), dtype=torch.float32,
                             device=model_management.intermediate_device())
        progress = utils.ProgressBar(len(plan))
        for i, (start, stop, context_stop) in enumerate(plan):
            model_management.throw_exception_if_processing_interrupted()
            logger.info("Open-H3 window %d/%d: frames %d-%d", i + 1, len(plan), start, stop - 1)
            pixels = images[start:context_stop] if fps == 24.0 else images[indices[start:context_stop]]
            video = self.encode_images(pixels, width, height, input_size, context)
            del pixels
            video = self.upscale(video, width, height)
            fixed_audio = self.encode_audio(audio, context_stop - start, context, start) if steps else None
            refined = self.sample(video, fixed_audio, cond, (seed + i) % 2**64, steps, start_noise, sparse)
            decoded = self.decode(refined, width, height)
            output[start:stop].copy_(decoded[:stop - start])
            del video, fixed_audio, refined, decoded
            progress.update_absolute(i + 1)
        return output, info

    def native(self, latent, prompt, width, height, seed, steps, start_noise,
               sparse=True, conditioning=None, audio=None):
        validate_canvas(width, height)
        samples = latent["samples"]
        fixed_audio = None
        if samples.is_nested:
            streams = samples.unbind()
            if len(streams) != 2:
                raise ValueError("Connect a native H3 video/audio latent pair")
            video, fixed_audio = streams
        else:
            video = samples
        if video.ndim != 5 or video.shape[:2] != (1, 24) or video.shape[2] not in (2, 7, 12, 17, 22, 27, 32, 37):
            raise ValueError("Native H3 input must be one 5-124 frame video latent. Decode longer latents and use Refine Video.")
        video = video.detach().float().cpu().clone()
        count = 5 + (video.shape[2] - 2) // 5 * 17
        if not torch.isfinite(video).all():
            raise ValueError("Native video latent contains non-finite values")
        if audio is not None or fixed_audio is None:
            fixed_audio = self.encode_audio(audio, count, count)
        else:
            fixed_audio = fixed_audio.detach().float().cpu().clone()
        if tuple(fixed_audio.shape) != (1, 32, 2, round(count / 24 * 40)) or not torch.isfinite(fixed_audio).all():
            raise ValueError("Native audio latent does not match the video duration")
        cond = self.conditioning(prompt, conditioning) if steps else None
        video = self.upscale(video, width, height)
        video = self.sample(video, fixed_audio, cond, seed, steps, start_noise, sparse)
        frames = self.decode(video, width, height)[:count]
        info = f"{count} native frames; output IMAGE alone {output_memory_gib(count, width, height):.2f} GiB. Model/working memory is additional."
        return frames, {"samples": NestedTensor([video, fixed_audio])}, info


def refinement_sigmas(steps=3, start_sigma=None):
    if steps not in SIGMAS:
        raise ValueError("Select the released two-step or three-step schedule")
    if start_sigma is None:
        return SIGMAS[steps]
    if not math.isfinite(start_sigma) or not 0 < start_sigma < 1:
        raise ValueError("Starting noise must be finite and between zero and one")
    # Equal spacing before the release's shift-6 transform. Keep the measured
    # three-step release grid above when no explicit starting noise is supplied.
    raw = start_sigma / (6 - 5 * start_sigma)
    middle = [raw * (1 - i / steps) for i in range(1, steps)]
    return (start_sigma, *(6 * x / (1 + 5 * x) for x in middle), 0.0)


def refine_latents(model, conditioning, video, audio, seed, steps=3, sparse=True, start_sigma=None):
    sigmas = refinement_sigmas(steps, start_sigma)
    patched = model
    if sparse:
        patched = apply_block_sparse_attention(
            model, tau=1.75, topk_ratio=0.0, vsa=False, start_percent=0.0, end_percent=1.0,
            min_tokens=0, dense_blocks=set(), sink_conditioning="exact_kv_and_rows",
            extra_tokens=0, verbose=True,
        )
    device = model_management.get_torch_device()
    noise = torch.randn(video.shape, dtype=torch.float32, device=device,
                        generator=torch.Generator(device=device).manual_seed(seed)).cpu()
    latent = NestedTensor([video.float().cpu(), audio.float().cpu()])
    noises = NestedTensor([noise, torch.zeros_like(audio, device="cpu")])
    mask = NestedTensor([torch.ones_like(video, device="cpu"), torch.zeros_like(audio, device="cpu")])
    progress = utils.ProgressBar(steps)

    def callback(step, x0, x, total):
        model_management.throw_exception_if_processing_interrupted()
        progress.update_absolute(step + 1, total)

    try:
        result = sample.sample_custom(
            patched, noises, 1.0, samplers.ksampler("euler"), torch.tensor(sigmas),
            conditioning, [], latent, noise_mask=mask, callback=callback, seed=seed,
        )
        video_out, audio_out = result.unbind()
        if not torch.allclose(audio_out, audio.float().to(audio_out), rtol=1e-5, atol=1e-5):
            raise RuntimeError("Refinement unexpectedly changed the fixed audio latent")
        return video_out
    finally:
        model_management.unload_model_and_clones(patched)
