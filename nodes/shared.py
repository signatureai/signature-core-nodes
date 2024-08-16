import os

from dotenv import load_dotenv
from signature_core import CURRENT_DIR

load_dotenv()

BASE_COMFY_DIR = os.path.dirname(os.path.realpath(__file__)).split("custom_nodes")[0]
SIGNATURE_NODES_DIR = os.path.dirname(os.path.realpath(__file__)).split("src")[0]
SIGNATURE_CORE_DIR = CURRENT_DIR
SD_SCRIPTS_DIR = os.getenv("SD_SCRIPTS_DIR") or os.path.join(CURRENT_DIR, "submodules/sd-scripts")
LORA_OUTPUT_DIR = os.getenv("LORA_OUTPUT_DIR") or os.path.join(BASE_COMFY_DIR, "models/loras")


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")
