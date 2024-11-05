import gc
import os
import sys

import torch
from dotenv import load_dotenv

load_dotenv()

BASE_COMFY_DIR: str = os.path.dirname(os.path.realpath(__file__)).split("custom_nodes")[0]
SIGNATURE_NODES_DIR: str = os.path.dirname(os.path.realpath(__file__)).split("src")[0]

MAX_INT: int = sys.maxsize
MAX_FLOAT: float = sys.float_info.max


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

sys.path.append(BASE_COMFY_DIR)
import comfy  # type: ignore


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.cleanup_models()
    comfy.model_management.soft_empty_cache()
