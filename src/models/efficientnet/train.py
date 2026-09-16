from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from vision_utils import (
    ImagePathDataset,
    build_transforms,
    discover_samples,
    find_classes,
    get_device,
    save_json,
    seed_everything,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImageNet-pretrained EfficientNet-B0")
    parser.add_argument("--data-dir", type=Path, default=Path("img/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = get_device(args.device)

    class_names, class_to_idx = find_classes(args.data_dir)
    all_samples = discover_samples(args.data_dir, class_to_idx)
    train_samples, crawler_test_samples = stratified_split(
        all_samples, args.test_ratio, args.seed
    )
    train_transform, _ = build_transforms(args.image_size)
    train_dataset = ImagePathDataset(train_samples, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Crawler split: train={len(train_samples)}, test={len(crawler_test_samples)}")

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={loss:.4f} | train_acc={accuracy:.4f}"
        )

    checkpoint = {
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": args.image_size,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "crawler_train_root": str(args.data_dir),
        "crawler_train_files": [
            str(path.relative_to(args.data_dir)) for path, _ in train_samples
        ],
        "crawler_test_files": [
            str(path.relative_to(args.data_dir)) for path, _ in crawler_test_samples
        ],
    }
    checkpoint_path = args.output_dir / "efficientnet_b0_final.pt"
    torch.save(checkpoint, checkpoint_path)
    save_json(
        {
            "config": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
            "classes": class_names,
            "train_samples": len(train_samples),
            "crawler_test_samples": len(crawler_test_samples),
            "elapsed_seconds": time.time() - started,
            "history": history,
            "note": "No validation set or test-set-based model selection was used.",
        },
        args.output_dir / "training_history.json",
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print("Next: python src/models/efficientnet/evaluate.py")


if __name__ == "__main__":
    main()
