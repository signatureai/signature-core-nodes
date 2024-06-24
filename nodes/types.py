from .categories import TYPES_CAT
from .shared import any
class Any2String():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = TYPES_CAT
    def process(self, input):
        return (str(input),)

class Any2Any():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = (any,)
    FUNCTION = "process"
    CATEGORY = TYPES_CAT
    def process(self, input):
        return (input,)


class TextPreview():
    @classmethod
    def INPUT_TYPES(s):  # type: ignore
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = TYPES_CAT
    def process(self, text):
        return {"ui": {"text": text}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "Signature Any2Any": Any2Any,
    "Signature Any2String": Any2String,
    "Signature Text Preview": TextPreview
}