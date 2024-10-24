import importlib
import inspect
import os
import re
import shutil

from dotenv import load_dotenv
from signature_core.logger import console
from signature_core.version import __version__

load_dotenv()

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


def get_node_class_mappings(nodes_directory: str):
    node_class_mappings = {}
    node_display_name_mappings = {}

    plugin_file_paths = []

    for path, _, files in os.walk(nodes_directory):
        for name in files:
            if not name.endswith(".py"):
                continue
            plugin_file_paths.append(os.path.join(path, name))

    for plugin_file_path in plugin_file_paths:
        plugin_rel_path = plugin_file_path.replace(".py", "").replace(os.sep, ".")
        plugin_rel_path = plugin_rel_path.split("signature-core-nodes.nodes.")[-1]

        try:
            module = importlib.import_module("signature-core-nodes.nodes." + plugin_rel_path)

            for item in dir(module):
                value = getattr(module, item)
                if (
                    not value
                    or not inspect.isclass(value)
                    or not value.__module__.startswith("signature-core-nodes.nodes.")
                ):
                    continue

                if hasattr(value, "FUNCTION"):
                    cleaned_name = item.replace("2", "")
                    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", cleaned_name).lower()
                    key = f"signature_{snake_case}"
                    node_class_mappings[key] = value
                    node_display_name_mappings[key] = f"SIG {item}"
        except ImportError as e:
            console.log(f"[red]Error importing {plugin_rel_path}: {e}")

    return node_class_mappings, node_display_name_mappings


# Get the path to the nodes directory
nodes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes")
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = get_node_class_mappings(nodes_path)

WEB_DIRECTORY = "./nodes/web"
NAME = "🔲 Signature Nodes"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MANIFEST"]

MANIFEST = {
    "name": NAME,
    "version": __version__,
    "author": "Marco, Frederico, Anderson",
    "description": "SIG Nodes",
}


greet_logo = f"""
          ███████████            ██████████████          ███████████████
       ██████     ███████      ██████████████████       ██████████████████
      ████            ████    █████           █████    ████           █████
      ████                    █████                    ████
       █████████               ██████████████          ██████████████
         ████████████████        █████████████████       █████████████████
                     █████                  ███████                ████████
      ██               ███    ███              ████   ████             █████
      ████            ████    █████           █████   ██████           ████
       ██████████████████      ███████████████████      ██████████████████
            █████████             █████████████            █████████████


    Maintained by: Marco, Frederico, Anderson
    Version: {__version__}
"""

console.log(greet_logo)
