import os

import folder_paths  # type: ignore
import torch

# comfy related imports
from comfy import sd, utils  # type: ignore
from signature_core.img.tensor_image import TensorImage
from uuid_extensions import uuid7str

from .categories import LORA_CAT
from .shared import BASE_COMFY_DIR


class ApplyLoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
    )
    RETURN_NAMES = (
        "MODEL",
        "CLIP",
    )
    FUNCTION = "execute"
    CATEGORY = CATEGORY = LORA_CAT

    def execute(
        self,
        **kwargs,
    ):
        model = kwargs.get("model")
        clip = kwargs.get("clip")
        lora_stack = kwargs.get("lora_stack")
        loras = []
        if lora_stack is None:
            return (
                model,
                clip,
            )

        model_lora = model
        clip_lora = clip
        loras.extend(lora_stack)

        for lora in loras:
            lora_name, strength_model, strength_clip = lora

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = utils.load_torch_file(lora_path, safe_load=True)

            model_lora, clip_lora = sd.load_lora_for_models(
                model_lora, clip_lora, lora, strength_model, strength_clip
            )

        return (
            model_lora,
            clip_lora,
        )


class LoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                "switch_1": (["Off", "On"],),
                "lora_name_1": (loras,),
                "model_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_2": (["Off", "On"],),
                "lora_name_2": (loras,),
                "model_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_3": (["Off", "On"],),
                "lora_name_3": (loras,),
                "model_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "execute"
    CATEGORY = LORA_CAT

    def execute(
        self,
        **kwargs,
    ):
        switch_1 = kwargs.get("switch_1")
        lora_name_1 = kwargs.get("lora_name_1")
        model_weight_1 = kwargs.get("model_weight_1")
        clip_weight_1 = kwargs.get("clip_weight_1")
        switch_2 = kwargs.get("switch_2")
        lora_name_2 = kwargs.get("lora_name_2")
        model_weight_2 = kwargs.get("model_weight_2")
        clip_weight_2 = kwargs.get("clip_weight_2")
        switch_3 = kwargs.get("switch_3")
        lora_name_3 = kwargs.get("lora_name_3")
        model_weight_3 = kwargs.get("model_weight_3")
        clip_weight_3 = kwargs.get("clip_weight_3")
        lora_stack = kwargs.get("lora_stack")

        lora_list: list = []
        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])

        if lora_name_1 != "None" and switch_1 == "On":
            lora_list.extend([(lora_name_1, model_weight_1, clip_weight_1)]),  # type: ignore

        if lora_name_2 != "None" and switch_2 == "On":
            lora_list.extend([(lora_name_2, model_weight_2, clip_weight_2)]),  # type: ignore

        if lora_name_3 != "None" and switch_3 == "On":
            lora_list.extend([(lora_name_3, model_weight_3, clip_weight_3)]),  # type: ignore

        return (lora_list,)


class Dict2LoraStack:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "lora_dicts": ("LIST",),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

        inputs["optional"] = {"lora_stack": ("LORA_STACK",)}
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "execute"
    CATEGORY = LORA_CAT

    def execute(self, **kwargs):
        lora_dicts = kwargs.get("lora_dicts")
        if not isinstance(lora_dicts, list):
            raise ValueError("Lora dicts must be a list")
        lora_stack = kwargs.get("lora_stack")
        loras = [None for _ in lora_dicts]

        for idx, lora_dict in enumerate(lora_dicts):
            loras[idx] = (lora_dict["lora_name"], lora_dict["lora_weight"], lora_dict["lora_weight"])  # type: ignore

        # If lora_stack is not None, extend the loras list with lora_stack
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)


class SaveLoraCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": ("STRING", {"default": ""}),
                "repeats": ("INT", {"default": 5, "min": 1}),
                "images": ("IMAGE",),
                "labels": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = LORA_CAT

    def execute(
        self,
        **kwargs,
    ):
        dataset_name = kwargs.get("dataset_name")
        if not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a string")
        repeats = kwargs.get("repeats")
        if not isinstance(repeats, int):
            raise ValueError("Repeats must be an integer")
        images = kwargs.get("images")
        if not isinstance(images, torch.Tensor):
            raise ValueError("Images must be a torch.Tensor")
        labels = kwargs.get("labels")
        if not isinstance(labels, str):
            raise ValueError("Labels must be a string")
        prefix = kwargs.get("prefix")
        if not isinstance(prefix, str):
            raise ValueError("Prefix must be a string")
        suffix = kwargs.get("suffix")
        if not isinstance(suffix, str):
            raise ValueError("Suffix must be a string")

        labels_list = labels.split("\n") if "\n" in labels else [labels]

        root_folder = os.path.join(BASE_COMFY_DIR, "loras_datasets")
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        uuid = uuid7str()
        dataset_folder = os.path.join(root_folder, f"{dataset_name}_{uuid}")
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        images_folder = os.path.join(dataset_folder, f"{repeats}_{dataset_name}")
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        tensor_images = TensorImage.from_BWHC(images)
        for i, img in enumerate(tensor_images):
            # timestamp to be added to the image name

            TensorImage(img).save(os.path.join(images_folder, f"{dataset_name}_{i}.png"))
            # write txt label with the same name of the image
            with open(os.path.join(images_folder, f"{dataset_name}_{i}.txt"), "w") as f:
                label = prefix + labels_list[i % len(labels_list)] + suffix
                f.write(label)

        return (dataset_folder,)


NODE_CLASS_MAPPINGS = {
    "signature_apply_lora_stack": ApplyLoraStack,
    "signature_lora_stack": LoraStack,
    "signature_dict_to_lora_stack": Dict2LoraStack,
    "signature_save_lora_captions": SaveLoraCaptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_apply_lora_stack": "SIG ApplyLoRAStack",
    "signature_lora_stack": "SIG LoRAStack",
    "signature_dict_to_lora_stack": "SIG Dict2LoRAStack",
    "signature_save_lora_captions": "SIG SaveLoRACaptions",
}
