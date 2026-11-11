"""Small V3 nodes around native ComfyUI H3 components."""

from comfy_api.latest import io

from .ic import ICRefiner
from .loading import ICModelSelection, ModelSelection, bundle_names, ic_bundle_names, model_names
from .runtime import Refiner


HANDLE = io.Custom("OLM_OPEN_H3_REFINER")
IC_HANDLE = io.Custom("OLM_OPEN_H3_IC_REFINER")
CATEGORY = "Olm/Open-H3 Refiner"
MODEL_HINTS = {
    "base_model": ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models", "Comfy-Org/MiniMax-H3"),
    "text_encoder": ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", "Comfy-Org/MiniMax-H3"),
    "video_vae": ("minimax_h3_video_vae_fp16.safetensors", "vae", "Comfy-Org/MiniMax-H3"),
    "audio_vae": ("minimax_h3_audio_vae_fp32.safetensors", "vae", "Comfy-Org/MiniMax-H3"),
    "upscaler": ("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", "latent_upscale_models", "LBH-123-AI/Minimax_h3_latent_Upscaler/tree/main/minimax_h3_latent_upscaler_3d_conv_v1"),
}


def model_input(kind, preferred, variant="fl2va"):
    example, folder, source = MODEL_HINTS[kind]
    if kind == "base_model":
        example = example.replace("fl2va", variant)
    names = model_names(kind, variant) or [f"(missing: {example})"]
    default = next((n for n in names if preferred in n), names[0])
    tooltip = (f"Recommended file: {example}\nSource: https://huggingface.co/{source}\n"
               f"Place in ComfyUI/models/{folder}/ or a registered extra_model_paths.yaml location. "
               "Existing files can stay on another drive. Restart ComfyUI after adding files or changing model paths.")
    return io.Combo.Input(kind, options=names, default=default, tooltip=tooltip)


def schedule_input():
    return io.DynamicCombo.Input("schedule", options=[
        io.DynamicCombo.Option("Custom", [
            io.Combo.Input("steps", options=[2, 3], default=2),
            io.Float.Input("start_noise", default=0.5, min=0.01, max=0.99, step=0.01,
                           tooltip="Lower noise generally preserves more of the upscaled input; it can still redraw details."),
        ]),
        io.DynamicCombo.Option("NVIDIA two-step", []),
        io.DynamicCombo.Option("NVIDIA three-step", []),
        io.DynamicCombo.Option("Upscaler only", []),
    ])


def selected_schedule(schedule):
    name = schedule["schedule"]
    if name == "Custom":
        return int(schedule["steps"]), float(schedule["start_noise"])
    return {"NVIDIA two-step": (2, None), "NVIDIA three-step": (3, None), "Upscaler only": (0, None)}[name]


def refinement_inputs():
    return [HANDLE.Input("refiner"), io.String.Input("prompt", multiline=True, default=""),
            io.Int.Input("width", default=2560, min=256, max=4096, step=32),
            io.Int.Input("height", default=1440, min=256, max=4096, step=32), schedule_input(),
            io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
            io.Combo.Input("vae_tile_size", options=[256, 192, 128, 384, 512], default=256,
                           tooltip="H3 VAE spatial tile size. Smaller tiles reduce working memory but may alter texture; 256 is native default."),
            io.Boolean.Input("sparse_attention", default=True, advanced=True,
                             tooltip="Native sparse attention, tested on RTX 5090. Disable to use ComfyUI's dense attention."),
            io.Conditioning.Input("conditioning", optional=True, advanced=True,
                                  tooltip="Optional H3 conditioning bypasses this node's text encoding. Otherwise the prompt is encoded locally."),
            io.Audio.Input("audio", optional=True,
                           tooltip="Soundtrack used as fixed conditioning. Output sound can bypass the node unchanged.")]


class OpenH3RefinerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bundles = bundle_names() or ["(missing: Open-H3-Regenerate-2K adapters)"]
        return io.Schema(node_id="OlmOpenH3RefinerLoader", display_name="Open-H3 Refiner Loader (Olm)",
                         category=CATEGORY, description="Select local native H3 components and the released adapter bundle. No automatic downloads.",
                         inputs=[model_input("base_model", "int8_convrot"), model_input("text_encoder", "nvfp4_awq"),
                                 model_input("video_vae", "fp16"), model_input("audio_vae", "fp32"),
                                 model_input("upscaler", "bf16"), io.Combo.Input("adapters", options=bundles,
                                     tooltip="Source: https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K\n"
                                             "Use models/open_h3_refiner/<bundle>/lora/ or a configured adapter root. "
                                             "Required: open_h3_regenerate_2k_lora.safetensors, sharp_curve_branches.safetensors "
                                             "and adaln_t_table.bin. Fetch/extract the table separately with "
                                             "scripts/fetch_sharp_table.py (--help). No automatic downloads."),
                                 io.Combo.Input("text_encoder_device", options=["auto", "cpu"], default="auto", advanced=True,
                                                tooltip="CPU text encoding reduces VRAM demand but is slower and requires substantial RAM.")],
                         outputs=[HANDLE.Output("refiner")])

    @classmethod
    def execute(cls, base_model, text_encoder, video_vae, audio_vae, upscaler, adapters, text_encoder_device="auto"):
        selection = ModelSelection(base_model, text_encoder, video_vae, audio_vae, upscaler, adapters, text_encoder_device)
        selection.resolve()
        return io.NodeOutput(selection)


class OpenH3RefineVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OlmOpenH3RefineVideo", display_name="Open-H3 Refine Video (Olm)", category=CATEGORY,
                         description="Generative video refinement at 24 fps. Other input rates are converted by dropping/repeating frames. Independent windows can show visible boundaries. Final IMAGE memory scales with the whole clip (121 frames at 2K ≈ 5 GiB).",
                         inputs=[io.Image.Input("images"), *refinement_inputs(),
                                 io.Float.Input("fps", default=24.0, min=1, max=120,
                                                tooltip="Actual input frame rate. Other rates are converted to 24 fps by dropping/repeating frames, without motion interpolation. Connect output fps to Create Video."),
                                 io.Combo.Input("input_size", options=["768p", "576p"], default="768p"),
                                 io.Combo.Input("window_frames", options=[121, 90, 56, 22], default=121,
                                                tooltip="Shorter windows reduce attention memory but can reduce quality. The model was tuned for the default ~5-second context.")],
                         outputs=[io.Image.Output("images"), io.String.Output("memory_info"),
                                  io.Float.Output("fps", tooltip="Output frame rate: 24 fps. Connect to Create Video.")])

    @classmethod
    def execute(cls, images, refiner, prompt, fps, width, height, schedule, seed, input_size,
                window_frames, vae_tile_size, sparse_attention=True, conditioning=None, audio=None):
        steps, noise = selected_schedule(schedule)
        with Refiner(refiner.resolve(), refiner.text_encoder_device, vae_tile_size) as engine:
            images, info = engine.frames(images, prompt, fps, width, height, seed, steps, noise,
                                         input_size, int(window_frames), sparse_attention, conditioning, audio)
        return io.NodeOutput(images, info, 24.0)


class OpenH3RefineLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OlmOpenH3RefineLatent", display_name="Open-H3 Refine Latent (Olm)", category=CATEGORY,
                         description="Refine native H3 video/AV latents without re-encoding. Supports the short 5-124 frame H3 grids. Decode longer latents and use Refine Video for windowed processing.",
                         inputs=[io.Latent.Input("latent"), *refinement_inputs()],
                         outputs=[io.Image.Output("images"), io.Latent.Output("latent"), io.String.Output("memory_info")])

    @classmethod
    def execute(cls, latent, refiner, prompt, width, height, schedule, seed, vae_tile_size,
                sparse_attention=True, conditioning=None, audio=None):
        steps, noise = selected_schedule(schedule)
        with Refiner(refiner.resolve(), refiner.text_encoder_device, vae_tile_size) as engine:
            images, latent, info = engine.native(latent, prompt, width, height, seed, steps, noise,
                                                 sparse_attention, conditioning, audio)
        return io.NodeOutput(images, latent, info)


class OpenH3ICLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bundles = ic_bundle_names() or ["(missing: Open-H3-Regenerate-2K-IC adapters)"]
        return io.Schema(node_id="OlmOpenH3ICLoader", display_name="Open-H3 IC Loader (Olm)", category=CATEGORY,
                         description="Select pruned Ref2VA models and the IC bundle. Prepare timestep data once with scripts/prepare_ic.py. No automatic downloads.",
                         inputs=[model_input("base_model", "int8_convrot", "ref2va"),
                                 model_input("text_encoder", "nvfp4_awq"), model_input("video_vae", "fp16"),
                                 model_input("upscaler", "bf16"), io.Combo.Input("adapters", options=bundles,
                                     tooltip="Source: https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC\n"
                                             "IC, HyperFlow and LMS Sharp adapters in lora/; silence_audio_124f.safetensors in assets/; "
                                             "prepared ic_time.safetensors at the bundle root. Use the existing open_h3_refiner model paths."),
                                 io.Combo.Input("text_encoder_device", options=["auto", "cpu"], default="auto", advanced=True,
                                                tooltip="CPU text encoding reduces VRAM demand but is slower and requires substantial RAM.")],
                         outputs=[IC_HANDLE.Output("refiner")])

    @classmethod
    def execute(cls, base_model, text_encoder, video_vae, upscaler, adapters, text_encoder_device="auto"):
        selection = ICModelSelection(base_model, text_encoder, video_vae, upscaler, adapters, text_encoder_device)
        selection.resolve()
        return io.NodeOutput(selection)


class OpenH3ICRefineVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OlmOpenH3ICRefineVideo", display_name="Open-H3 IC Refine Video (Olm)", category=CATEGORY,
                         description="IC refinement with the draft as a clean reference, video-aware prompt encoding and published three/four-step HyperFlow schedules. Longer videos use independent windows with possible visible boundaries. Route original audio directly to Create Video.",
                         inputs=[io.Image.Input("images"), IC_HANDLE.Input("refiner"),
                                 io.String.Input("prompt", multiline=True, default="",
                                                 tooltip="Optional scene description. The enhancement instruction and video reference are supplied automatically."),
                                 io.Float.Input("fps", default=24., min=1, max=120,
                                                tooltip="Actual input rate; drop/repeat conversion produces 24 fps output."),
                                 io.Int.Input("width", default=2560, min=256, max=4096, step=32),
                                 io.Int.Input("height", default=1440, min=256, max=4096, step=32),
                                 io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                                 io.Float.Input("sharp_strength", default=1., min=0., max=2., step=.1,
                                                tooltip="LMS Sharp strength. 1 is NVIDIA's default; 0 disables it. This is separate from IC LoRA strength."),
                                 io.Combo.Input("input_size", options=["768p", "576p"], default="768p"),
                                 io.Combo.Input("window_frames", options=[121, 90, 56, 22], default=121,
                                                tooltip="Default context fits 124 frames. Shorter windows reduce working memory; independent windows may show seams."),
                                 io.Combo.Input("vae_tile_size", options=[256, 192, 128, 384, 512], default=256),
                                 io.Boolean.Input("sparse_attention", default=True, advanced=True,
                                                  tooltip="Native block-sparse attention, tested on RTX 5090. Disable for dense attention."),
                                 io.Float.Input("ic_strength", default=1., min=0., max=2., step=.05,
                                                optional=True, advanced=True,
                                                tooltip="Experimental IC LoRA strength. 1 is the publisher's value. Lower values need not preserve details better. At 0, the Ref2VA reference, HyperFlow and any Sharp remain active."),
                                 io.Combo.Input("sampler", options=["up3", "up4"], default="up3",
                                                optional=True, advanced=True,
                                                tooltip="Published HyperFlow schedules: up3 uses 3 steps from noise 0.835; up4 uses 4 steps from 0.923. The timestep data supports both.")],
                         outputs=[io.Image.Output("images"), io.String.Output("memory_info"), io.Float.Output("fps")])

    @classmethod
    def execute(cls, images, refiner, prompt, fps, width, height, seed, sharp_strength=1.,
                input_size="768p", window_frames=121, vae_tile_size=256, sparse_attention=True,
                ic_strength=1., sampler="up3"):
        with ICRefiner(refiner.resolve(), refiner.text_encoder_device, vae_tile_size) as engine:
            output, info = engine.frames(images, prompt, fps, width, height, seed, sharp_strength,
                                         input_size, int(window_frames), sparse_attention, ic_strength, sampler)
        return io.NodeOutput(output, info, 24.)
