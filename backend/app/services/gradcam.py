"""
Grad-CAM for torchvision ViT-B/16 (token-wise CAM on the last encoder block).

We take gradients of the target logit w.r.t. the last transformer block outputs,
positive-weight the token dimensions, drop the CLS token, reshape patch tokens to
14×14, upsample to 224×224, and overlay on the denormalized input (clinical-style
attention map — interpret as saliency, not a segmentation mask).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _denormalize_tensor_chw(x: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized → uint8 RGB [H,W,3] for display."""
    mean = IMAGENET_MEAN.to(x.device).view(3, 1, 1)
    std = IMAGENET_STD.to(x.device).view(3, 1, 1)
    t = x * std + mean
    t = t.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return (t * 255.0).astype(np.uint8)


def compute_vit_grad_cam(
    model: nn.Module,
    input_chw: torch.Tensor,
    target_class_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        original_rgb: (224, 224, 3) uint8
        heatmap: (224, 224) float in [0, 1]
        overlay_rgb: (224, 224, 3) uint8 — alpha blend of jet colormap + original
    """
    model.eval()
    x = input_chw.unsqueeze(0).to(device, dtype=torch.float32).detach()
    x.requires_grad_(False)

    activations: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        out.retain_grad()
        activations.append(out)

    last_block = model.encoder.layers[-1]
    handle = last_block.register_forward_hook(forward_hook)

    try:
        with torch.enable_grad():
            logits = model(x)
            if target_class_index < 0 or target_class_index >= logits.size(1):
                target_class_index = int(logits.argmax(dim=1).item())
            score = logits[0, target_class_index]
            model.zero_grad(set_to_none=True)
            score.backward()

        if not activations:
            raise RuntimeError("Grad-CAM hook captured no activations")

        act = activations[0]
        grad = act.grad
        if grad is None:
            raise RuntimeError("No gradients on encoder activations (check model forward path)")

        # Token-wise importance: sum over hidden dim (Grad-CAM style on token embeddings).
        cam = (grad * act).sum(dim=-1)
        cam = F.relu(cam)[0]
        patch_cam = cam[1:]
        grid = int(patch_cam.numel() ** 0.5)
        if grid * grid != patch_cam.numel():
            logger.warning("Unexpected token count %s for sqrt grid", patch_cam.numel())
            grid = int(np.sqrt(patch_cam.numel()))

        hm = patch_cam.reshape(grid, grid)
        hm = hm - hm.min()
        maxv = hm.max()
        if float(maxv) > 1e-8:
            hm = hm / maxv
        else:
            hm = torch.ones_like(hm) * 0.25

        hm_up = F.interpolate(
            hm.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap = hm_up.detach().cpu().numpy().astype(np.float32)

        orig_rgb = _denormalize_tensor_chw(input_chw)

        # Teal/clinical colormap via matplotlib-style jet channel mix (numpy only below).
        h = heatmap
        overlay_rgb = _apply_heatmap_overlay(orig_rgb, h, alpha=0.48)

        return orig_rgb, heatmap, overlay_rgb
    finally:
        handle.remove()


def _apply_heatmap_overlay(rgb_uint8: np.ndarray, heatmap_01: np.ndarray, alpha: float) -> np.ndarray:
    """Blend turquoise heatmap over RGB image."""
    h = np.clip(heatmap_01, 0.0, 1.0)
    # Simple "medical" map: dark blue → cyan → yellow (piecewise in RGB)
    c1 = np.stack([h, 0.85 * h + 0.15, 0.25 + 0.55 * (1 - h)], axis=-1)
    c1 = (np.clip(c1, 0, 1) * 255).astype(np.float32)
    base = rgb_uint8.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * c1
    return np.clip(out, 0, 255).astype(np.uint8)


def pil_to_model_tensor(transform, image: Image.Image) -> torch.Tensor:
    """PIL RGB → [3,224,224] normalized tensor (CPU)."""
    t = transform(image.convert("RGB"))
    return t
