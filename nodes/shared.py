import os

BASE_COMFY_DIR = os.getcwd().split('custom_nodes')[0]
SIGNATURE_DIR =  os.path.dirname(os.path.realpath(__file__)).split('src')[0]


class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")