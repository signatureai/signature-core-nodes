import os
import sys

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
