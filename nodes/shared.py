import os

from dotenv import load_dotenv

load_dotenv()

BASE_COMFY_DIR: str = os.path.dirname(os.path.realpath(__file__)).split("custom_nodes")[0]
SIGNATURE_NODES_DIR: str = os.path.dirname(os.path.realpath(__file__)).split("src")[0]

MAX_INT: int = 10**1000
MAX_FLOAT: float = 1.7976931348623157e308


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")
