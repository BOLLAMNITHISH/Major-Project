"""
Pure PyTorch training script: Vision Transformer (ViT-B/16, 224×224) for brain MRI classification.

Expected data layout (one subfolder per class):
    DATA_ROOT/
        glioma/
        meningioma/
        pituitary/
        no_tumor/

Example:
    python train_vit_pytorch.py --data_dir ./data/brain_mri --epochs 30 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless save of confusion matrix figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16


# -----------------------------------------------------------------------------
# ImageNet normalization: ViT was pretrained on ImageNet; same stats align
# the input distribution with what the backbone expects (before fine-tuning).
# -----------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ViT-B/16 for 4-class brain tumor MRI classification"
    )
    p.add_argument("--data_dir", type=str, required=True, help="Root with class subfolders")
    p.add_argument("--output_dir", type=str, default="./checkpoints_vit_torch")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (head + unfrozen blocks)")
    p.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay (common for ViT)")
    p.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of data for validation")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num_unfrozen_blocks",
        type=int,
        default=2,
        help="How many last transformer encoder blocks to train (rest of backbone frozen)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Reproducibility: same weight init, shuffling, and splits across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageFolderSubset(Dataset):
    """
    Wraps torchvision ImageFolder with a fixed index list so train/val can use
    different augmentation pipelines while sharing the same on-disk dataset.
    """

    def __init__(
        self,
        folder: datasets.ImageFolder,
        indices: list[int],
        transform: transforms.Compose | None,
    ) -> None:
        self.folder = folder
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, target = self.folder.samples[self.indices[i]]
        img = self.folder.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Train: resize to 224×224, augment (flip / rotate / scale), then tensor + normalize.
    Val: deterministic resize + normalize only (no randomness).
    """
    # Training augmentations reduce overfitting on small medical datasets.
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            # RandomAffine: translation + isotropic scale simulates mild zoom and shift
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def stratified_train_val_indices(
    targets: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices with similar class proportions in train and validation."""
    idx = np.arange(len(targets))
    y = np.array(targets)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, stratify=y, random_state=seed, shuffle=True
    )
    return train_idx.tolist(), val_idx.tolist()


def load_vit_finetune_last_layers(
    num_classes: int, num_unfrozen_blocks: int, device: torch.device
) -> nn.Module:
    """
    Load torchvision ViT-B/16 (patch 16, 224×224), matching 'vit-base-patch16-224' family.

    - Replace the classification head for `num_classes`.
    - Freeze early transformer blocks; train the last `num_unfrozen_blocks` blocks,
      the final LayerNorm, and the head. This is a standard 'fine-tune last layers' recipe.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"] is ImageNet; we replace head for our task.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Freeze all parameters first, then selectively enable gradients on the tail.
    for p in model.parameters():
        p.requires_grad = False

    n_blocks = len(model.encoder.layers)
    if num_unfrozen_blocks > n_blocks:
        num_unfrozen_blocks = n_blocks
    start = n_blocks - num_unfrozen_blocks

    # Unfreeze the last K encoder blocks (self-attention + MLP inside each block).
    for i in range(start, n_blocks):
        for p in model.encoder.layers[i].parameters():
            p.requires_grad = True

    # Final encoder LayerNorm before the classification head is part of the 'top' of the network.
    for p in model.encoder.ln.parameters():
        p.requires_grad = True

    for p in model.heads.parameters():
        p.requires_grad = True

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single pass over the training set; returns average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    """
    Validation / test loop: loss plus sklearn metrics (macro-averaged).
    Returns: avg_loss, metrics dict, y_true, y_pred (for confusion matrix).
    """
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(n, 1)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Macro average: unweighted mean over classes (fair when classes are imbalanced).
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return avg_loss, metrics, y_true, y_pred


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion matrix (validation)",
) -> None:
    """Save a heatmap of the confusion matrix for error analysis."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load file paths + labels without transforms; we attach transforms per split below.
    base = datasets.ImageFolder(args.data_dir, transform=None)
    class_names = base.classes
    num_classes = len(class_names)
    if num_classes != 4:
        print(
            f"Warning: expected 4 classes, found {num_classes}. "
            "Metrics and head still use len(classes)."
        )

    train_idx, val_idx = stratified_train_val_indices(base.targets, args.val_ratio, args.seed)
    train_tf, val_tf = build_transforms()
    train_ds = ImageFolderSubset(base, train_idx, train_tf)
    val_ds = ImageFolderSubset(base, val_idx, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_vit_finetune_last_layers(
        num_classes=num_classes,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        device=device,
    )

    # Multi-class classification: CrossEntropyLoss = log_softmax + NLL (standard for ViT logits).
    criterion = nn.CrossEntropyLoss()

    # Only parameters with requires_grad=True receive updates (frozen backbone stays fixed).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} "
            f"P={val_metrics['precision_macro']:.4f} "
            f"R={val_metrics['recall_macro']:.4f} "
            f"F1={val_metrics['f1_macro']:.4f}"
        )

        # Track best checkpoint by macro-F1 on validation.
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_path = out_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved best model (val macro-F1={best_f1:.4f}) -> {best_path}")

            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_val.png")

    # Persist last epoch full weights as requested (.pth).
    last_path = out_dir / "last_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "history": history,
            "args": vars(args),
        },
        last_path,
    )
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Evaluation report for FastAPI GET /evaluation + Plotly dashboard
    best_path = out_dir / "best_model.pth"
    if best_path.is_file():
        try:
            try:
                ckpt_eval = torch.load(best_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt_eval = torch.load(best_path, map_location=device)
            state = ckpt_eval.get("model_state_dict")
            if state is not None:
                eval_model = vit_b_16(weights=None)
                eval_model.heads.head = nn.Linear(
                    eval_model.heads.head.in_features, num_classes
                )
                eval_model.load_state_dict(state)
                eval_model.to(device)
                eval_model.eval()
                v_loss, v_metrics, y_t, y_p = evaluate(
                    eval_model, val_loader, criterion, device
                )
                cm_final = confusion_matrix(
                    y_t, y_p, labels=list(range(num_classes))
                )
                report = {
                    "class_names": class_names,
                    "metrics": v_metrics,
                    "confusion_matrix": cm_final.tolist(),
                    "val_loss": float(v_loss),
                    "note": "Held-out validation split (train_vit_pytorch.py); best checkpoint by macro-F1.",
                }
                report_path = out_dir / "evaluation_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"  Wrote evaluation report for API dashboard -> {report_path}")
        except Exception as ex:
            print(f"  Warning: could not write evaluation_report.json: {ex}")

    print(f"Done. Best val macro-F1={best_f1:.4f}. Artifacts in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
