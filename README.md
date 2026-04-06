# Multiclass Brain Tumor Classification (ViT)

Monorepo: **FastAPI** backend loading a **torchvision ViT-B/16** checkpoint (`.pth` from `model/train_vit_pytorch.py`), **React (Vite + Tailwind)** dashboard, **SQLite** prediction history, and **model/** training scripts.

**Classes (typical):** `glioma`, `meningioma`, `pituitary`, `no_tumor` — must match your trained checkpoint.

---

## Folder structure

```
major-Project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI, CORS, lifespan, /health
│   │   ├── config.py            # Settings (env)
│   │   ├── database.py          # Async SQLAlchemy + SQLite
│   │   ├── models.py            # ORM: PredictionRecord
│   │   ├── schemas.py           # Pydantic models
│   │   ├── routers/
│   │   │   └── inference.py     # POST /predict, GET /history
│   │   └── services/
│   │       └── predictor.py     # PIL + torchvision ViT inference
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/ ...
│   └── vite.config.ts           # Proxies /predict, /history, /health → :8000
├── model/
│   ├── train_vit_pytorch.py   # PyTorch training → best_model.pth
│   ├── train.py               # Optional Hugging Face Trainer
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup commands

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
# Set VIT_CHECKPOINT_PATH to your best_model.pth

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API: `http://127.0.0.1:8000` · OpenAPI: `http://127.0.0.1:8000/docs`

The server **requires** `VIT_CHECKPOINT_PATH` pointing at a valid `.pth` file (see `.env.example`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App: `http://127.0.0.1:5173` (Vite proxies `/predict`, `/history`, and `/health` to the backend.)

### Train the ViT (PyTorch)

```bash
cd model
pip install -r requirements.txt
python train_vit_pytorch.py --data_dir PATH_TO_IMAGEFOLDER --output_dir ./checkpoints_vit_torch --epochs 30
```

Set `VIT_CHECKPOINT_PATH` to `checkpoints_vit_torch/best_model.pth` in `backend/.env`.

---

## API summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, version, model loaded, device, checkpoint path |
| POST | `/predict` | `multipart/form-data` field `file` — MRI image |
| GET | `/history?limit=100` | Recent predictions (SQLite) |

**Example `POST /predict` JSON:**

```json
{
  "prediction": "glioma",
  "confidence": 0.97,
  "all_scores": {
    "glioma": 0.97,
    "meningioma": 0.02,
    "pituitary": 0.005,
    "no_tumor": 0.005
  }
}
```

---

## Requirements files

- **Backend:** `backend/requirements.txt`
- **Training:** `model/requirements.txt`
- **Frontend:** `frontend/package.json`

---

## Notes

- First-time PyTorch / CUDA setup varies by machine.
- This stack is for **research and education**; it is not a medical device.
