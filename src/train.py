"""DeiT-Small/16 학습 엔진: 실험 A(선형 분류층), 실험 B(부분 미세조정) 및 다중 시드 반복."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transforms import get_train_transforms, get_val_transforms
from src.models.vit_classifier import (
    create_deit_model,
    freeze_backbone_for_linear_probe,
    unfreeze_last_blocks_for_fine_tuning,
    get_optimizer_param_groups,
    save_tumbler_checkpoint,
    CLASSES,
)
from src.data.dataset import create_dataloaders

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"


def set_seed(seed: int = 42) -> None:
    """모든 난수 발생기를 고정하여 완벽한 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler: GradScaler,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += len(targets)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * len(targets)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    n_total = len(y_true)

    # 공식 스펙: 고정 4종 Macro F1, zero_division=0
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1_per_cls, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_per_cls[i]),
            "support": int(sup[i]),
        }

    return {
        "val_loss": total_loss / n_total,
        "val_macro_f1": float(macro_f1),
        "val_acc": float(acc),
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.array(all_probs),
    }


def run_training_experiment(
    experiment: str = "A",
    seed: int = 42,
    epochs: int | None = None,
    batch_size: int = 8,
    lr: float | None = None,
    lr_blocks: float = 1e-5,
    patience: int = 5,
    pretrained_a_checkpoint: Path | str | None = None,
) -> tuple[dict, Path]:
    """단일 실험 (A 또는 B)을 수행하고 최적 체크포인트를 저장합니다."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting Experiment {experiment} (Seed: {seed}) on {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    print(f"=======================================================")

    # 전처리 및 데이터로더
    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)
    train_loader, val_loader = create_dataloaders(
        METADATA_CSV, train_tf, val_tf, batch_size=batch_size, num_workers=0
    )
    print(f"Data: Train {len(train_loader.dataset)} samples ({len(train_loader)} batches), Val {len(val_loader.dataset)} samples ({len(val_loader)} batches)")

    # 모델 초기화
    model = create_deit_model(pretrained=True, num_classes=len(CLASSES))

    if experiment.upper() == "A":
        # 실험 A: 분류층 학습 (Feature Extractor 동결, 헤드만 학습)
        freeze_backbone_for_linear_probe(model)
        actual_epochs = epochs if epochs is not None else 10
        actual_lr = lr if lr is not None else 1e-3
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=actual_lr,
            weight_decay=0.01,
        )
    elif experiment.upper() == "B":
        # 실험 B: 부분 미세조정 (실험 A 최적 가중치 로드 후 마지막 4블록+norm+head 학습)
        if pretrained_a_checkpoint is not None and Path(pretrained_a_checkpoint).is_file():
            print(f"Loading Experiment A weights: {pretrained_a_checkpoint}")
            ckpt = torch.load(pretrained_a_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
        else:
            print("Note: Experiment A checkpoint not provided or not found. Training B from ImageNet weights.")

        unfreeze_last_blocks_for_fine_tuning(model, num_blocks=4)
        actual_epochs = epochs if epochs is not None else 25
        actual_lr = lr if lr is not None else 1e-4  # head lr
        param_groups = get_optimizer_param_groups(model, lr_blocks=lr_blocks, lr_head=actual_lr, weight_decay=0.01)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 학습 및 조기 종료 트래킹
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = CHECKPOINTS_DIR / f"exp_{experiment.lower()}_seed{seed}_best.pt"

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_macro_f1": [], "val_acc": [],
    }

    start_time = time.time()
    for epoch in range(1, actual_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_macro_f1"].append(val_metrics["val_macro_f1"])
        history["val_acc"].append(val_metrics["val_acc"])

        curr_f1 = val_metrics["val_macro_f1"]
        curr_loss = val_metrics["val_loss"]

        is_improved = (curr_f1 > best_macro_f1) or (abs(curr_f1 - best_macro_f1) < 1e-6 and curr_loss < best_val_loss)
        if is_improved:
            best_macro_f1 = curr_f1
            best_val_loss = curr_loss
            patience_counter = 0
            train_args = {
                "experiment": experiment,
                "seed": seed,
                "epochs": actual_epochs,
                "batch_size": batch_size,
                "lr": actual_lr,
                "lr_blocks": lr_blocks,
            }
            save_tumbler_checkpoint(
                best_ckpt_path, model, epoch, best_macro_f1, best_val_loss, train_args
            )
            marker = " ★ BEST"
        else:
            patience_counter += 1
            marker = f" (Patience: {patience_counter}/{patience})"

        print(
            f"Epoch [{epoch:02d}/{actual_epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Val Loss: {curr_loss:.4f} Macro F1: {curr_f1:.4f} Acc: {val_metrics['val_acc']:.3f}{marker}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    elapsed = time.time() - start_time
    print(f"Experiment {experiment} (Seed {seed}) finished in {elapsed:.1f}s. Best Val Macro F1: {best_macro_f1:.4f}")

    # 학습 곡선 시각화 저장
    plot_training_curves(history, experiment, seed)
    return {"best_macro_f1": best_macro_f1, "best_val_loss": best_val_loss, "history": history}, best_ckpt_path


def plot_training_curves(history: dict, experiment: str, seed: int):
    """Loss 및 Macro F1 학습 곡선을 시각화하여 저장합니다."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title(f"Exp {experiment} (Seed {seed}) - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Macro F1 & Accuracy
    ax2.plot(epochs, history["val_macro_f1"], "g-o", label="Val Macro F1 (Primary)")
    ax2.plot(epochs, history["val_acc"], "c--^", label="Val Accuracy")
    ax2.set_title(f"Exp {experiment} (Seed {seed}) - Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = FIGURES_DIR / f"training_curves_exp_{experiment.lower()}_seed{seed}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curve: {plot_path}")


def compare_and_repeat_seeds():
    """1) Seed 42에서 실험 A와 B 비교 -> 2) 우승 모델 선정 -> 3) 우승 모델 Seed 43, 44 반복 실행."""
    print("=================================================================")
    print(" PHASE 1: Compare Experiment A vs Experiment B on Seed 42")
    print("=================================================================")

    # 1. Exp A (Linear Probe, max 10 epochs)
    res_a, ckpt_a = run_training_experiment(experiment="A", seed=42, epochs=10, batch_size=8, lr=1e-3)

    # 2. Exp B (Partial Fine-tuning, max 25 epochs, starting from Exp A)
    res_b, ckpt_b = run_training_experiment(
        experiment="B", seed=42, epochs=25, batch_size=8, lr=1e-4, lr_blocks=1e-5, pretrained_a_checkpoint=ckpt_a
    )

    f1_a = res_a["best_macro_f1"]
    f1_b = res_b["best_macro_f1"]
    print("\n-----------------------------------------------------------------")
    print(f" COMPARISON RESULT (Seed 42):")
    print(f"  Experiment A (Linear Probe):        Val Macro F1 = {f1_a:.4f}")
    print(f"  Experiment B (Partial Fine-tuning): Val Macro F1 = {f1_b:.4f}")

    winner = "B" if f1_b >= f1_a else "A"
    print(f"  Selected Winner for Repetition:     Experiment {winner}")
    print("-----------------------------------------------------------------")

    # 3. 우승 방식 Seed 43, 44 반복
    all_f1s = [f1_b if winner == "B" else f1_a]
    for rep_seed in [43, 44]:
        print(f"\n Repeating Winner (Exp {winner}) on Seed {rep_seed}...")
        pretrained_ckpt = ckpt_a if winner == "B" else None
        res_rep, _ = run_training_experiment(
            experiment=winner, seed=rep_seed,
            epochs=25 if winner == "B" else 10,
            batch_size=8,
            lr=1e-4 if winner == "B" else 1e-3,
            lr_blocks=1e-5,
            pretrained_a_checkpoint=pretrained_ckpt,
        )
        all_f1s.append(res_rep["best_macro_f1"])

    mean_f1 = np.mean(all_f1s)
    std_f1 = np.std(all_f1s)
    print("\n=================================================================")
    print(f" MULTI-SEED REPETITION SUMMARY (Exp {winner}):")
    print(f"  Seed 42: {all_f1s[0]:.4f}")
    print(f"  Seed 43: {all_f1s[1]:.4f}")
    print(f"  Seed 44: {all_f1s[2]:.4f}")
    print(f"  Mean Val Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
    print(f"  Primary Model for Test Set: Seed 42 Best Checkpoint")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--experiment", choices=["A", "B"], default="A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "compare":
        compare_and_repeat_seeds()
    else:
        run_training_experiment(
            experiment=args.experiment,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
