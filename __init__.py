from dotenv import load_dotenv
from signature_core.version import __version__

from .nodes import (
    dev_tools,
    io,
    lora,
    numbers,
    platform_io,
    platform_wrapper,
    primitives,
    utils,
)
from .nodes.vision import (
    augmentation,
    color,
    enhance,
    filters,
    misc,
    models,
    morphology,
    transform,
)

load_dotenv()


NODE_CLASS_MAPPINGS = {
    **models.NODE_CLASS_MAPPINGS,
    **transform.NODE_CLASS_MAPPINGS,
    **enhance.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
    **morphology.NODE_CLASS_MAPPINGS,
    **misc.NODE_CLASS_MAPPINGS,
    **augmentation.NODE_CLASS_MAPPINGS,
    **lora.NODE_CLASS_MAPPINGS,
    **io.NODE_CLASS_MAPPINGS,
    **platform_io.NODE_CLASS_MAPPINGS,
    **utils.NODE_CLASS_MAPPINGS,
    **color.NODE_CLASS_MAPPINGS,
    **numbers.NODE_CLASS_MAPPINGS,
    **primitives.NODE_CLASS_MAPPINGS,
    **dev_tools.NODE_CLASS_MAPPINGS,
    **platform_wrapper.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **models.NODE_DISPLAY_NAME_MAPPINGS,
    **transform.NODE_DISPLAY_NAME_MAPPINGS,
    **enhance.NODE_DISPLAY_NAME_MAPPINGS,
    **filters.NODE_DISPLAY_NAME_MAPPINGS,
    **morphology.NODE_DISPLAY_NAME_MAPPINGS,
    **misc.NODE_DISPLAY_NAME_MAPPINGS,
    **augmentation.NODE_DISPLAY_NAME_MAPPINGS,
    **lora.NODE_DISPLAY_NAME_MAPPINGS,
    **io.NODE_DISPLAY_NAME_MAPPINGS,
    **platform_io.NODE_DISPLAY_NAME_MAPPINGS,
    **utils.NODE_DISPLAY_NAME_MAPPINGS,
    **color.NODE_DISPLAY_NAME_MAPPINGS,
    **numbers.NODE_DISPLAY_NAME_MAPPINGS,
    **primitives.NODE_DISPLAY_NAME_MAPPINGS,
    **dev_tools.NODE_DISPLAY_NAME_MAPPINGS,
    **platform_wrapper.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./nodes/web"
NAME = "🔸 Signature Nodes"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MANIFEST"]

MANIFEST = {
    "name": NAME,
    "version": __version__,
    "author": "Marco, Frederico, Anderson",
    "description": "SIG Nodes",
}
