import os
from comfy import model_management # type: ignore
from dotenv import load_dotenv

load_dotenv()

COMFY_DEVICE = model_management.get_torch_device()
BASE_COMFY_DIR = os.getcwd().split('custom_nodes')[0]
SIGNATURE_DIR = os.path.dirname(os.path.realpath(__file__)).split('src')[0]
SD_SCRIPTS_DIR = os.getenv('SD_SCRIPTS_DIR')
LORA_OUTPUT_DIR = os.getenv('LORA_OUTPUT_DIR')

class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")