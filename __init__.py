import os
import shutil

from dotenv import load_dotenv
from signature_core.logger import console
from signature_core.version import __version__

# script folder
script_folder = os.path.dirname(os.path.realpath(__file__))
base_comfy_dir = os.path.dirname(os.path.realpath(__file__)).split("custom_nodes")[0]
signature_js = "signature.js"
signature_js_path = os.path.join(script_folder, "nodes/web/")
web_extensions = os.path.join(base_comfy_dir, "web/extensions/")
src = os.path.join(signature_js_path, signature_js)
dst = os.path.join(web_extensions, signature_js)
if os.path.exists(web_extensions):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)

console.log(f"[green]:::> Signature Core version: {__version__}")


from .nodes import (
    augmentations,
    data,
    file,
    image,
    lora,
    mask,
    models,
    numbers,
    platform_io,
    primitives,
    transform,
    utils,
    wrapper,
)

load_dotenv()


NODE_CLASS_MAPPINGS = {
    **image.NODE_CLASS_MAPPINGS,
    **mask.NODE_CLASS_MAPPINGS,
    **file.NODE_CLASS_MAPPINGS,
    **transform.NODE_CLASS_MAPPINGS,
    **numbers.NODE_CLASS_MAPPINGS,
    **primitives.NODE_CLASS_MAPPINGS,
    **augmentations.NODE_CLASS_MAPPINGS,
    **models.NODE_CLASS_MAPPINGS,
    **utils.NODE_CLASS_MAPPINGS,
    **data.NODE_CLASS_MAPPINGS,
    **lora.NODE_CLASS_MAPPINGS,
    **platform_io.NODE_CLASS_MAPPINGS,
    **wrapper.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **image.NODE_DISPLAY_NAME_MAPPINGS,
    **mask.NODE_DISPLAY_NAME_MAPPINGS,
    **file.NODE_DISPLAY_NAME_MAPPINGS,
    **transform.NODE_DISPLAY_NAME_MAPPINGS,
    **numbers.NODE_DISPLAY_NAME_MAPPINGS,
    **primitives.NODE_DISPLAY_NAME_MAPPINGS,
    **augmentations.NODE_DISPLAY_NAME_MAPPINGS,
    **models.NODE_DISPLAY_NAME_MAPPINGS,
    **utils.NODE_DISPLAY_NAME_MAPPINGS,
    **data.NODE_DISPLAY_NAME_MAPPINGS,
    **lora.NODE_DISPLAY_NAME_MAPPINGS,
    **platform_io.NODE_DISPLAY_NAME_MAPPINGS,
    **wrapper.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./nodes/web"
NAME = "🔲 Signature Nodes"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MANIFEST"]

MANIFEST = {
    "name": NAME,
    "version": __version__,
    "author": "Marco, Frederico, Anderson",
    "description": "SIG Nodes",
}
