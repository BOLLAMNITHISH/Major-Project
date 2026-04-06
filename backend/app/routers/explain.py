import base64
import io
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.config import settings
from app.schemas import ExplainResponse
from app.services.predictor import get_predictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explainability"])

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@router.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)) -> ExplainResponse:
    """
    Grad-CAM on the last ViT encoder block: saliency map over patch tokens,
    upsampled and overlaid on the input slice. Shown beside the original image.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        raw = await file.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {settings.max_upload_mb} MB)",
            )
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e!s}") from e

    try:
        predictor = get_predictor()
        label, confidence, all_scores, idx = predictor.predict_and_class_index(image)
        orig_rgb, _hm, overlay_rgb, target_idx = predictor.grad_cam_from_pil(image, idx)
        target_label = predictor.class_names[target_idx]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Grad-CAM failed")
        raise HTTPException(status_code=500, detail=f"Explainability failed: {e!s}") from e

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=140)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=11, color="0.2")
    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Grad-CAM · {target_label}", fontsize=11, color="0.2")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Predicted: {label} ({100 * confidence:.1f}% confidence)",
        fontsize=12,
        fontweight="medium",
        y=1.02,
    )
    fig.patch.set_facecolor("#f8fafc")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return ExplainResponse(
        prediction=label,
        confidence=confidence,
        all_scores=all_scores,
        target_class_index=target_idx,
        target_class_label=target_label,
        side_by_side_png_base64=b64,
    )
