import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.schemas import EvaluationReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])


def _resolve_evaluation_report_path() -> Path | None:
    if settings.evaluation_report_path:
        p = Path(settings.evaluation_report_path)
        if p.is_file():
            return p
    if settings.vit_checkpoint_path:
        sibling = Path(settings.vit_checkpoint_path).parent / "evaluation_report.json"
        if sibling.is_file():
            return sibling
    fallback = Path("./data/evaluation_report.json")
    if fallback.is_file():
        return fallback
    return None


def _load_report_sync(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/evaluation", response_model=EvaluationReportResponse)
async def get_evaluation_report() -> EvaluationReportResponse:
    path = _resolve_evaluation_report_path()
    if path is None:
        return EvaluationReportResponse(available=False)

    try:
        data = await asyncio.to_thread(_load_report_sync, path)
    except Exception as e:
        logger.warning("Failed to read evaluation report: %s", e)
        return EvaluationReportResponse(available=False)

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        metrics = {k: float(v) for k, v in metrics.items()}

    cm = data.get("confusion_matrix")
    if not isinstance(cm, list):
        cm = None

    names = data.get("class_names")
    if not isinstance(names, list):
        names = None

    val_loss = data.get("val_loss")
    if val_loss is not None:
        val_loss = float(val_loss)

    note = data.get("note")
    if note is not None:
        note = str(note)

    return EvaluationReportResponse(
        available=True,
        source_path=str(path.resolve()),
        class_names=[str(x) for x in names] if names else None,
        metrics=metrics,
        confusion_matrix=cm,
        val_loss=val_loss,
        note=note,
    )
