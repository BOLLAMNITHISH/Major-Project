"""
Train ViT for 4-class brain MRI classification (glioma, meningioma, pituitary, no_tumor).

Expected dataset layout (ImageFolder):
  DATA_ROOT/
    glioma/
    meningioma/
    pituitary/
    no_tumor/

Usage:
  python train.py --data_dir ./data/brain_mri --output_dir ./checkpoints/vit-brain-tumor --epochs 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoImageProcessor, Trainer, TrainingArguments, ViTForImageClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_LABELS = ["glioma", "meningioma", "pituitary", "no_tumor"]
LABEL2ID = {name: i for i, name in enumerate(CLASS_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(CLASS_LABELS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ViT for brain tumor classification")
    p.add_argument("--data_dir", type=str, required=True, help="Root folder with class subfolders")
    p.add_argument("--output_dir", type=str, default="./checkpoints/vit-brain-tumor")
    p.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {data_dir}")

    for name in CLASS_LABELS:
        if not (data_dir / name).is_dir():
            logger.warning("Missing class folder: %s", data_dir / name)

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    raw = load_dataset("imagefolder", data_dir=str(data_dir))
    hf_names = raw["train"].features["label"].names
    hf_id_to_name = {i: hf_names[i] for i in range(len(hf_names))}

    def remap_labels(batch):
        batch["label"] = [LABEL2ID[hf_id_to_name[i]] for i in batch["label"]]
        return batch

    ds = raw["train"].map(remap_labels, batched=True)
    ds = ds.train_test_split(test_size=1.0 - args.train_split, seed=args.seed)

    def preprocess(batch):
        images = [im.convert("RGB") for im in batch["image"]]
        enc = processor(images=images, return_tensors="np")
        return {"pixel_values": enc["pixel_values"], "labels": batch["label"]}

    train_ds = ds["train"].map(preprocess, batched=True, remove_columns=["image"])
    eval_ds = ds["test"].map(preprocess, batched=True, remove_columns=["image"])
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    eval_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    logger.info("Saved model and processor to %s", out_dir)
    logger.info("Set backend env CHECKPOINT_PATH=%s for inference.", out_dir)


if __name__ == "__main__":
    main()
