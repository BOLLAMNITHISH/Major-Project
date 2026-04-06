import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import PredictionRecord
from app.schemas import PredictResponse, PredictionHistoryItem
from app.services.predictor import get_predictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="MRI image file"),
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = settings.uploads_dir / safe_name

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb} MB)")

    dest.write_bytes(content)

    try:
        predictor = get_predictor()
        image = Image.open(dest)
        prediction, confidence, all_scores = predictor.predict_pil(image)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass

    record = PredictionRecord(
        original_filename=file.filename,
        predicted_label=prediction,
        confidence=confidence,
        scores_json=json.dumps(all_scores),
    )
    db.add(record)
    await db.commit()

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        all_scores=all_scores,
    )


@router.get("/history", response_model=list[PredictionHistoryItem])
async def history(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> list[PredictionHistoryItem]:
    limit = min(max(limit, 1), 500)
    result = await db.execute(
        select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
    )
    rows = result.scalars().all()
    out: list[PredictionHistoryItem] = []
    for r in rows:
        scores_map = json.loads(r.scores_json)
        out.append(
            PredictionHistoryItem(
                id=r.id,
                filename=r.original_filename,
                prediction=r.predicted_label,
                confidence=r.confidence,
                all_scores={k: float(v) for k, v in scores_map.items()},
                created_at=r.created_at,
            )
        )
    return out
