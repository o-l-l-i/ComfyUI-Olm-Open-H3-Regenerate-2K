"""Olm Open-H3 generative video refinement nodes."""

from comfy_api.latest import ComfyExtension, io

from .loading import register_model_paths
from .nodes import OpenH3RefinerLoader, OpenH3RefineLatent, OpenH3RefineVideo


class OpenH3Extension(ComfyExtension):
    async def on_load(self):
        register_model_paths()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [OpenH3RefinerLoader, OpenH3RefineVideo, OpenH3RefineLatent]


async def comfy_entrypoint() -> ComfyExtension:
    return OpenH3Extension()
