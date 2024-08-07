import torch
from signature_core.functional.morphology import dilation, erosion
from signature_core.img.tensor_image import TensorImage

from ..categories import MISC_CAT


class Bitwise:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "mode": (["and", "or", "xor", "left_shift", "right_shift"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask_1: torch.Tensor, mask_2: torch.Tensor, mode: str):
        input_mask_1 = TensorImage.from_BWHC(mask_1)
        input_mask_2 = TensorImage.from_BWHC(mask_2)
        eight_bit_mask_1 = torch.tensor(input_mask_1 * 255, dtype=torch.uint8)
        eight_bit_mask_2 = torch.tensor(input_mask_2 * 255, dtype=torch.uint8)

        if mode == "and":
            result = torch.bitwise_and(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "or":
            result = torch.bitwise_or(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "xor":
            result = torch.bitwise_xor(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "left_shift":
            result = torch.bitwise_left_shift(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "right_shift":
            result = torch.bitwise_right_shift(eight_bit_mask_1, eight_bit_mask_2)
        else:
            raise ValueError("Invalid mode")

        float_result = result.float() / 255
        output_mask = TensorImage(float_result).get_BWHC()
        return (output_mask,)


class Ones:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "channels": ("INT", {"default": 1, "min": 1, "max": 4}),
                "batch": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, width: int, height: int, channels: int, batch: int):
        step = torch.ones((batch, channels, height, width))
        output_image = TensorImage(step).get_BWHC()
        return (output_image,)


class Zeros:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "channels": ("INT", {"default": 1}),
                "batch": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, width: int, height: int, channels: int, batch: int):
        step = torch.zeros((batch, channels, height, width))
        output_image = TensorImage(step).get_BWHC()
        return (output_image,)


class OnesLike:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_BWHC(image)
        step = torch.ones_like(input_image)
        output_image = TensorImage(step).get_BWHC()
        return (output_image,)


class ZerosLike:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_BWHC(image)
        step = torch.zeros_like(input_image)
        output_image = TensorImage(step).get_BWHC()
        return (output_image,)


class MaskBinaryFilter:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.01, "min": 0.00, "max": 1.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask: torch.Tensor, threshold: float):
        step = TensorImage.from_BWHC(mask)
        step[step > threshold] = 1.0
        step[step <= threshold] = 0.0
        output = TensorImage(step).get_BWHC()
        return (output,)


class MaskInvert:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask: torch.Tensor):
        step = TensorImage.from_BWHC(mask)
        step = 1.0 - step
        output = TensorImage(step).get_BWHC()
        return (output,)


class MaskDistance:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"mask_0": ("MASK",), "mask_1": ("MASK",)}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask_0: torch.Tensor, mask_1: torch.Tensor):
        tensor1 = TensorImage.from_BWHC(mask_0)
        tensor2 = TensorImage.from_BWHC(mask_1)
        dist = torch.Tensor((tensor1 - tensor2).pow(2).sum(3).sqrt().mean())
        return (dist,)


class CreateTrimap:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
                "inner_min_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                "inner_max_threshold": ("INT", {"default": 255, "min": 0, "max": 255}),
                "outer_min_threshold": ("INT", {"default": 15, "min": 0, "max": 255}),
                "outer_max_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "kernel_size": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MASK", "TRIMAP")
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(
        self,
        mask: torch.Tensor,
        inner_min_threshold,
        inner_max_threshold,
        outer_min_threshold,
        outer_max_threshold,
        kernel_size,
    ):
        step = TensorImage.from_BWHC(mask)
        inner_mask = TensorImage(step.clone())
        inner_mask[inner_mask > (inner_max_threshold / 255.0)] = 1.0
        inner_mask[inner_mask <= (inner_min_threshold / 255.0)] = 0.0

        step = TensorImage.from_BWHC(mask)
        inner_mask = erosion(image=inner_mask, kernel_size=kernel_size, iterations=1)

        inner_mask[inner_mask != 0.0] = 1.0

        outter_mask = step.clone()
        outter_mask[outter_mask > (outer_max_threshold / 255.0)] = 1.0
        outter_mask[outter_mask <= (outer_min_threshold / 255.0)] = 0.0
        outter_mask = dilation(image=inner_mask, kernel_size=kernel_size, iterations=5)

        outter_mask[outter_mask != 0.0] = 1.0

        trimap_im = torch.zeros_like(step)
        trimap_im[outter_mask == 1.0] = 0.5
        trimap_im[inner_mask == 1.0] = 1.0
        batch_size = step.shape[0]

        trimap = torch.zeros(
            batch_size, 2, step.shape[2], step.shape[3], dtype=step.dtype, device=step.device
        )
        for i in range(batch_size):
            tar_trimap = trimap_im[i][0]
            trimap[i][1][tar_trimap == 1] = 1
            trimap[i][0][tar_trimap == 0] = 1

        output_0 = TensorImage(trimap_im).get_BWHC()
        output_1 = trimap.permute(0, 2, 3, 1)

        return (
            output_0,
            output_1,
        )


NODE_CLASS_MAPPINGS = {
    "signature_bitwise": Bitwise,
    "signature_ones": Ones,
    "signature_zeros": Zeros,
    "signature_ones_like": OnesLike,
    "signature_zeros_like": ZerosLike,
    "signature_mask_invert": MaskInvert,
    "signature_mask_binary_filter": MaskBinaryFilter,
    "signature_mask_distance": MaskDistance,
    "signature_create_trimap": CreateTrimap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_bitwise": "SIG Bitwise",
    "signature_ones": "SIG Ones",
    "signature_zeros": "SIG Zeros",
    "signature_ones_like": "SIG Ones Like",
    "signature_zeros_like": "SIG Zeros Like",
    "signature_mask_invert": "SIG Mask Invert",
    "signature_mask_binary_filter": "SIG Mask Binary Filter",
    "signature_mask_distance": "SIG Mask Distance",
    "signature_create_trimap": "SIG Create Trimap",
}
