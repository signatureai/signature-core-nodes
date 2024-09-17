import comfy  # type: ignore
import folder_paths  # type: ignore
import torch
from comfy import model_management  # type: ignore
from signature_core.functional.color import rgba_to_rgb
from signature_core.functional.transform import (
    auto_crop,
    cutout,
    rescale,
    resize,
    rotate,
)
from signature_core.img.tensor_image import TensorImage
from spandrel import ImageModelDescriptor, ModelLoader  # type: ignore

from ..categories import TRANSFORM_CAT


class AutoCrop:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_threshold": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 1.00, "step": 0.01}),
                "left_padding": ("INT", {"default": 0}),
                "right_padding": ("INT", {"default": 0}),
                "top_padding": ("INT", {"default": 0}),
                "bottom_padding": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "x", "y", "width", "height")

    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_threshold: float,
        left_padding: int,
        right_padding: int,
        top_padding: int,
        bottom_padding: int,
    ):
        img_tensor = TensorImage.from_BWHC(image)
        mask_tensor = TensorImage.from_BWHC(mask)
        if img_tensor.shape[1] != 3:
            img_tensor = rgba_to_rgb(img_tensor)

        padding = (left_padding, right_padding, top_padding, bottom_padding)
        img_result, mask_result, min_x, min_y, width, height = auto_crop(
            img_tensor, mask_tensor, mask_threshold=mask_threshold, padding=padding
        )
        output_img = TensorImage(img_result).get_BWHC()
        output_mask = TensorImage(mask_result).get_BWHC()

        return (output_img, output_mask, min_x, min_y, width, height)


class Rescale:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
                "interpolation": (
                    ["nearest", "nearest-exact", "bilinear", "bicubic", "box", "hamming", "lanczos"],
                ),
                "antialias": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(
        self,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        factor: float = 2.0,
        interpolation: str = "nearest",
        antialias: bool = True,
    ):
        input_image = (
            TensorImage.from_BWHC(image)
            if isinstance(image, torch.Tensor)
            else TensorImage(torch.zeros((1, 3, 1, 1)))
        )
        input_mask = (
            TensorImage.from_BWHC(mask)
            if isinstance(mask, torch.Tensor)
            else TensorImage(torch.zeros((1, 1, 1, 1)))
        )
        output_image = rescale(input_image, factor, interpolation, antialias).get_BWHC()
        output_mask = rescale(input_mask, factor, interpolation, antialias).get_BWHC()

        return (
            output_image,
            output_mask,
        )


class Resize:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "width": ("INT", {"default": 1024, "min": 32, "step": 2, "max": 40960}),
                "height": ("INT", {"default": 1024, "min": 32, "step": 2, "max": 40960}),
                "mode": (["STRETCH", "FIT", "FILL", "ASPECT"],),
                "interpolation": (["bilinear", "nearest", "bicubic", "area"],),
                "antialias": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(
        self,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        width: int = 1024,
        height: int = 1024,
        mode: str = "default",
        interpolation: str = "nearest",
        antialias: bool = True,
    ):
        input_image = (
            TensorImage.from_BWHC(image)
            if isinstance(image, torch.Tensor)
            else TensorImage(torch.zeros((1, 3, width, height)))
        )
        input_mask = (
            TensorImage.from_BWHC(mask)
            if isinstance(mask, torch.Tensor)
            else TensorImage(torch.zeros((1, 1, width, height)))
        )
        output_image = resize(input_image, width, height, mode, interpolation, antialias).get_BWHC()
        output_mask = resize(input_mask, width, height, mode, interpolation, antialias).get_BWHC()

        return (
            output_image,
            output_mask,
        )


class Rotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0, "max": 360.0, "step": 1.0}),
                "zoom_to_fit": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(
        self,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        angle: float = 0.0,
        zoom_to_fit: bool = False,
    ):
        input_image = (
            TensorImage.from_BWHC(image)
            if isinstance(image, torch.Tensor)
            else TensorImage(torch.zeros((1, 3, 1, 1)))
        )
        input_mask = (
            TensorImage.from_BWHC(mask)
            if isinstance(mask, torch.Tensor)
            else TensorImage(torch.zeros((1, 1, 1, 1)))
        )
        output_image = rotate(input_image, angle, zoom_to_fit).get_BWHC()
        output_mask = rotate(input_mask, angle, zoom_to_fit).get_BWHC()

        return (
            output_image,
            output_mask,
        )


class Cutout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "rgba")
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor):
        tensor_image = TensorImage.from_BWHC(image)
        tensor_mask = TensorImage.from_BWHC(mask, image.device)

        image_rgb, image_rgba = cutout(tensor_image, tensor_mask)

        out_image_rgb = TensorImage(image_rgb).get_BWHC()
        out_image_rgba = TensorImage(image_rgba).get_BWHC()

        return (
            out_image_rgb,
            out_image_rgba,
        )


class UpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        resampling_methods = ["bilinear", "nearest", "bicubic", "area"]

        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "mode": (["rescale", "resize"],),
                "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 100.0, "step": 0.01}),
                "resize_size": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                "resampling_method": (resampling_methods,),
                "tiled_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = TRANSFORM_CAT

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise ValueError("Upscale model must be a single-image model.")

        return out

    def upscale_with_model(self, upscale_model, image, device, tile: int = 512, overlap=32) -> torch.Tensor:
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (tile * tile * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)
        in_img = image.movedim(-1, -3).to(device)

        s = None
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        if not isinstance(s, torch.Tensor):
            raise ValueError("Upscaling failed")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # type: ignore
        return s

    def upscale(
        self,
        image,
        upscale_model,
        mode="rescale",
        resampling_method="bilinear",
        rescale_factor=2,
        resize_size=1024,
        tiled_size=512,
    ):
        # Load upscale model
        up_model = self.load_model(upscale_model)
        device = model_management.get_torch_device()
        up_model.to(device)

        # target size
        _, H, W, _ = image.shape
        target_size = resize_size if mode == "resize" else max(H, W) * rescale_factor
        current_size = max(H, W)
        up_image = image
        while current_size < target_size:
            step = self.upscale_with_model(up_model, up_image, device=device, tile=tiled_size)
            del up_image
            up_image = step.to("cpu")
            _, H, W, _ = up_image.shape
            current_size = max(H, W)

        up_model.to("cpu")
        tensor_image = TensorImage.from_BWHC(up_image)

        if mode == "resize":
            up_image = resize(
                tensor_image, resize_size, resize_size, "ASPECT", resampling_method, True
            ).get_BWHC()
        else:
            # get the max size of the upscaled image
            _, _, H, W = tensor_image.shape
            upscaled_max_size = max(H, W)

            original_image = TensorImage.from_BWHC(image)
            _, _, ori_H, ori_W = original_image.shape
            original_max_size = max(ori_H, ori_W)

            # rescale_factor is the factor to multiply the original max size
            original_target_size = rescale_factor * original_max_size
            scale_factor = original_target_size / upscaled_max_size

            up_image = rescale(tensor_image, scale_factor, resampling_method, True).get_BWHC()

        return (up_image,)


NODE_CLASS_MAPPINGS = {
    "signature_cutout": Cutout,
    "signature_rotate": Rotate,
    "signature_rescale": Rescale,
    "signature_resize": Resize,
    "signature_auto_crop": AutoCrop,
    "signature_upscale_image": UpscaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_cutout": "SIG Cutout",
    "signature_rotate": "SIG Rotate",
    "signature_rescale": "SIG Rescale",
    "signature_resize": "SIG Resize",
    "signature_auto_crop": "SIG Auto Crop",
    "signature_upscale_image": "SIG Upscale Image",
}
