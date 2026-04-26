"""
Grad-CAM heatmap generation for EfficientNetXRay.

Targets the last feature block of EfficientNet-B0 (model.features[-1]).
Returns a blended PIL Image with the jet-colormap heatmap overlaid on the
original grayscale X-ray converted to RGB.

NOTE: GradCAM requires gradient computation.  Call this OUTSIDE any
``torch.no_grad()`` context.  The model must be in eval() mode.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_cam(
    model: torch.nn.Module,
    tensor_input: torch.Tensor,
    class_index: int,
    alpha: float = 0.5,
    original_pil: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap blended onto the original image.

    Parameters
    ----------
    model:
        The loaded EfficientNetXRay in eval mode.
    tensor_input:
        Shape (1, 3, H, W) on the same device as the model.
    class_index:
        Index of the class to explain (argmax of sigmoid probs recommended).
    alpha:
        Blending weight for the heatmap overlay (0 = original, 1 = full CAM).
    original_pil:
        If provided, the heatmap is resized and overlaid on this image.
        If None, a plain jet-colormap CAM image is returned.

    Returns
    -------
    PIL.Image in RGB mode.
    """
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(class_index)]

    # GradCAM needs gradients — use enable_grad explicitly.
    with torch.enable_grad():
        cam_obj = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam_obj(input_tensor=tensor_input, targets=targets)

    # grayscale_cam shape: (batch, H, W) — take first item.
    grayscale_cam = grayscale_cam[0]

    if original_pil is not None:
        # Resize original to match CAM dimensions (224×224) and convert to RGB float.
        resized = original_pil.convert("RGB").resize(
            (grayscale_cam.shape[1], grayscale_cam.shape[0]), Image.BILINEAR
        )
        rgb_array = np.array(resized, dtype=np.float32) / 255.0
    else:
        # Fallback: grey background.
        rgb_array = np.ones(
            (grayscale_cam.shape[0], grayscale_cam.shape[1], 3), dtype=np.float32
        ) * 0.5

    blended = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True, image_weight=1.0 - alpha)
    return Image.fromarray(blended)
