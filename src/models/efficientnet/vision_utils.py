from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_classes(root: Path) -> tuple[list[str], dict[str, int]]:
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(f"At least two class directories are required under {root}")
    return classes, {name: index for index, name in enumerate(classes)}


def discover_samples(root: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    missing = [name for name in class_to_idx if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"Missing class directories under {root}: {missing}")
    extra = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name not in class_to_idx
    )
    if extra:
        raise ValueError(f"Unexpected class directories under {root}: {extra}")

    for class_name, class_index in class_to_idx.items():
        files = sorted(
            path for path in (root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No images found for class '{class_name}' under {root}")
        samples.extend((path, class_index) for path in files)
    return samples


def stratified_split(
    samples: list[tuple[Path, int]], test_ratio: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    train_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []
    for class_samples in grouped.values():
        rng.shuffle(class_samples)
        test_count = max(1, round(len(class_samples) * test_ratio))
        if test_count >= len(class_samples):
            raise ValueError("Each class needs at least two images for a train/test split")
        test_samples.extend(class_samples[:test_count])
        train_samples.extend(class_samples[test_count:])
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class ImagePathDataset(Dataset):
    def __init__(self, samples: Iterable[tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                # Phone photos commonly store camera orientation in EXIF instead of
                # rotating pixels. Apply it before resizing/cropping so evaluation
                # sees the same upright image as a normal image viewer.
                image = ImageOps.exif_transpose(image).convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc
        return image, label, str(path)


def classification_metrics(confusion: np.ndarray, class_names: list[str]) -> dict:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class = {}
    f1_values = []
    precision_values = []
    recall_values = []
    for i, name in enumerate(class_names):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        support = int(confusion[i, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "samples": total,
        "per_class": per_class,
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_csv(confusion: np.ndarray, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *class_names])
        for name, row in zip(class_names, confusion.tolist()):
            writer.writerow([name, *row])


def save_confusion_image(confusion: np.ndarray, class_names: list[str], path: Path) -> None:
    cell, left, top = 110, 170, 90
    width = left + cell * len(class_names) + 20
    height = top + cell * len(class_names) + 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    small_font = ImageFont.load_default(size=12)
    maximum = max(1, int(confusion.max()))
    draw.text((20, 15), "Confusion matrix (rows=actual, columns=predicted)", fill="black", font=font)
    for index, name in enumerate(class_names):
        x = left + index * cell + cell // 2
        draw.text((x, 55), name, fill="black", font=small_font, anchor="mm")
        y = top + index * cell + cell // 2
        draw.text((left - 10, y), name, fill="black", font=small_font, anchor="rm")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            value = int(confusion[row, col])
            strength = value / maximum
            color = (int(245 - 175 * strength), int(250 - 120 * strength), 255)
            box = (
                left + col * cell,
                top + row * cell,
                left + (col + 1) * cell,
                top + (row + 1) * cell,
            )
            draw.rectangle(box, fill=color, outline=(180, 180, 180))
            draw.text(
                ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
                str(value), fill="black", font=font, anchor="mm"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
