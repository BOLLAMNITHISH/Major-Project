import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import evaluation, explain, inference
from app.schemas import HealthResponse
from app.services.predictor import (
    get_runtime_checkpoint_path,
    get_runtime_device,
    load_predictor,
    model_is_ready,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("./data").mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    await init_db()

    ckpt = settings.vit_checkpoint_path
    if not ckpt:
        raise RuntimeError(
            "VIT_CHECKPOINT_PATH is not set. Point it to a .pth file from "
            "model/train_vit_pytorch.py (e.g. checkpoints_vit_torch/best_model.pth)."
        )
    p = Path(ckpt)
    if not p.is_file():
        raise FileNotFoundError(
            f"VIT_CHECKPOINT_PATH does not exist or is not a file: {p.resolve()}"
        )
    logger.info("Loading ViT weights from %s", p.resolve())
    load_predictor(p)

    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router)
app.include_router(explain.router)
app.include_router(evaluation.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = model_is_ready()
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        version=settings.app_version,
        model_loaded=loaded,
        device=get_runtime_device(),
        checkpoint=get_runtime_checkpoint_path(),
    )
