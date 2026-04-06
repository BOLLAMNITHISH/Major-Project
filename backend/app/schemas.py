from datetime import datetime

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """JSON shape returned by POST /predict."""

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float]


class PredictionHistoryItem(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    all_scores: dict[str, float]
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str | None = None
    checkpoint: str | None = None


class ExplainResponse(BaseModel):
    """Grad-CAM explainability; side-by-side PNG is base64 (no data: prefix)."""

    prediction: str
    confidence: float
    all_scores: dict[str, float]
    target_class_index: int
    target_class_label: str
    side_by_side_png_base64: str
    """Left: original MRI; right: Grad-CAM heatmap overlay (saliency)."""


class EvaluationReportResponse(BaseModel):
    """Validation metrics + confusion matrix for research dashboard."""

    available: bool
    source_path: str | None = None
    class_names: list[str] | None = None
    metrics: dict[str, float] | None = None
    confusion_matrix: list[list[int]] | None = None
    val_loss: float | None = None
    note: str | None = None
