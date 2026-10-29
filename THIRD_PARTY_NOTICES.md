# Third-party notices

`upscaler.py` adapts the 3D-convolution model from
[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler),
copyright 2026 LBH-123-AI, under the [MIT license](licenses/LBH-MIT.txt).
The model structure and normalization follow the copy in NVIDIA's
[Open-H3 source snapshot](https://github.com/NVlabs/Sana/blob/6e4a44002149418a547434b91d7280a8bbdcace3/models/Open-H3-Regenerate-2K/open_h3/upscaler_node.py).
Changes use ComfyUI operations, remove registration and global caches, omit
unused attention and temporal chunking, and replace inference dropout with an
identity while preserving checkpoint keys.

The adapter recipe and sampling schedule follow NVIDIA's same source snapshot.
ComfyUI's model, text encoder, VAE, sampler, quantization, and sparse attention
implementations are imported from the user's installation.

Weights are not distributed with this project. Model and component licenses are
separate from the integration code; see the
[publisher's model card](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K).
