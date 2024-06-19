from signature_core.nodes.categories import LORA_CAT

class SignatureLoraStacker:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "lora_dicts": ("LIST",),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",)
            }
        }

        inputs["optional"] = {
            "lora_stack": ("LORA_STACK",)
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "lora_stacker"
    CATEGORY = LORA_CAT

    def lora_stacker(self, lora_dicts: list, lora_stack=None):
        loras = [None for _ in lora_dicts]

        for idx, lora_dict in enumerate(lora_dicts):
            loras[idx] = (lora_dict["lora_name"], lora_dict["lora_weight"], lora_dict["lora_weight"]) # type: ignore

        # If lora_stack is not None, extend the loras list with lora_stack
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)

NODE_CLASS_MAPPINGS = {
    "signature_LoRA_Stacker": SignatureLoraStacker,
}