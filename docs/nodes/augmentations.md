# Augmentations Nodes

## RandomCropAugmentation

Applies a random crop augmentation to an image.

This class performs a random crop on an image based on specified dimensions and
percentage.

Methods: execute(\*\*kwargs): Applies the random crop augmentation and returns the
augmented image.

Args: height (int): The height of the image. width (int): The width of the image.
min_window (int): The minimum window size for cropping. max_window (int): The maximum
window size for cropping. percent (float): The percentage of the image to crop.
augmentation: An optional existing augmentation to apply.

Returns: tuple: The augmented image.

### Return Types

- `AUGMENTATION`

::: nodes.augmentations.RandomCropAugmentation

## FlipAugmentation

Applies a flip augmentation to an image.

This class performs a horizontal or vertical flip on an image based on the specified
direction and percentage.

Methods: execute(\*\*kwargs): Applies the flip augmentation and returns the augmented
image.

Args: flip (str): The direction of the flip ('horizontal' or 'vertical'). percent
(float): The percentage of the image to flip. augmentation: An optional existing
augmentation to apply.

Returns: tuple: The augmented image.

### Return Types

- `AUGMENTATION`

::: nodes.augmentations.FlipAugmentation

## ComposeAugmentation

Composes multiple augmentations and applies them to an image and mask.

This class combines multiple augmentations and applies them to an image and mask,
supporting multiple samples and random seeds.

Methods: execute(\*\*kwargs): Applies the composed augmentations and returns the
augmented images and masks.

Args: augmentation: The augmentation to apply. samples (int): The number of samples to
generate. seed (int): The random seed for augmentation. image: The input image to
augment. mask: The input mask to augment.

Returns: tuple: Lists of augmented images and masks.

### Return Types

- `IMAGE`
- `MASK`

::: nodes.augmentations.ComposeAugmentation
