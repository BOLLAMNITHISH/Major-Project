"""
Loads torchvision ViT-B/16 fine-tuned weights (.pth) from train_vit_pytorch.py.

Preprocessing matches validation transforms: PIL → Resize(224) → ToTensor → ImageNet normalize.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

from app.config import settings
from app.services.gradcam import compute_vit_grad_cam, pil_to_model_tensor

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchViTPredictor:
    """Thread-safe for inference-only use (eval mode, no gradients)."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._class_names: list[str] = []
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._model: nn.Module
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"ViT checkpoint not found: {path.resolve()}")

        # weights_only=True avoids arbitrary code execution on newer PyTorch versions
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        class_names = ckpt.get("class_names")
        num_classes = ckpt.get("num_classes")
        state = ckpt.get("model_state_dict")
        if not isinstance(class_names, list) or state is None:
            raise ValueError(
                "Checkpoint must contain 'class_names' and 'model_state_dict' "
                "(export from model/train_vit_pytorch.py)."
            )
        if num_classes is None:
            num_classes = len(class_names)

        model = vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, int(num_classes))
        model.load_state_dict(state, strict=True)

        self._model = model.to(self._device)
        self._model.eval()
        self._class_names = [str(c) for c in class_names]
        logger.info(
            "Loaded ViT-B/16 from %s | classes=%s | device=%s",
            path,
            self._class_names,
            self._device,
        )

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        """Returns predicted class name, confidence, and per-class probabilities."""
        rgb = image.convert("RGB")
        x = self._transform(rgb).unsqueeze(0).to(self._device, non_blocking=True)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        label = self._class_names[idx]
        confidence = float(probs[idx])
        all_scores = {self._class_names[i]: float(probs[i]) for i in range(len(self._class_names))}
        return label, confidence, all_scores

    def predict_and_class_index(self, image: Image.Image) -> tuple[str, float, dict[str, float], int]:
        """Same as predict_pil plus argmax index (for Grad-CAM target)."""
        rgb = image.convert("RGB")
        x = self._transform(rgb).unsqueeze(0).to(self._device, non_blocking=True)
        with torch.inference_mode():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        label = self._class_names[idx]
        confidence = float(probs[idx])
        all_scores = {self._class_names[i]: float(probs[i]) for i in range(len(self._class_names))}
        return label, confidence, all_scores, idx

    def grad_cam_from_pil(
        self,
        image: Image.Image,
        target_class_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Grad-CAM for the given class (default: predicted class).

        Returns original_rgb (uint8), heatmap [0,1], overlay_rgb (uint8), target_index used.
        """
        rgb = image.convert("RGB")
        input_chw = pil_to_model_tensor(self._transform, rgb)
        if target_class_index is None:
            _, _, _, target_class_index = self.predict_and_class_index(rgb)
        orig, hm, overlay = compute_vit_grad_cam(
            self._model, input_chw, target_class_index, self._device
        )
        return orig, hm, overlay, target_class_index

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def device(self) -> str:
        return str(self._device)


_predictor: TorchViTPredictor | None = None
_checkpoint_used: str | None = None


def load_predictor(checkpoint_path: Path) -> TorchViTPredictor:
    """Load or reload the ViT from disk (call once at startup)."""
    global _predictor, _checkpoint_used
    resolved = str(checkpoint_path.resolve())
    if _predictor is not None and _checkpoint_used == resolved:
        return _predictor
    _predictor = TorchViTPredictor(checkpoint_path)
    _checkpoint_used = resolved
    return _predictor


def get_predictor() -> TorchViTPredictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialized; call load_predictor() during startup.")
    return _predictor


def model_is_ready() -> bool:
    return _predictor is not None


def get_runtime_checkpoint_path() -> str | None:
    return _checkpoint_used


def get_runtime_device() -> str | None:
    return _predictor.device if _predictor is not None else None
