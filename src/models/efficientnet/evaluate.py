from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from vision_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImagePathDataset,
    build_transforms,
    classification_metrics,
    discover_samples,
    get_device,
    save_confusion_csv,
    save_confusion_image,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crawler holdout and real photos")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/efficientnet_b0_final.pt"))
    parser.add_argument("--crawler-root", type=Path, default=Path("img/train"))
    parser.add_argument("--real-root", type=Path, default=Path("img/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cam-per-class", type=int, default=3)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate_dataset(model, loader, device, class_names, output_dir: Path):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    rows = []
    for images, labels, paths in loader:
        probabilities = model(images.to(device)).softmax(dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        for label, prediction, probability, path in zip(labels, predictions, probabilities, paths):
            actual = int(label)
            predicted = int(prediction)
            confusion[actual, predicted] += 1
            rows.append(
                {
                    "path": path,
                    "actual": class_names[actual],
                    "predicted": class_names[predicted],
                    "confidence": float(probability[predicted]),
                    "correct": actual == predicted,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = classification_metrics(confusion, class_names)
    save_json(metrics, output_dir / "metrics.json")
    save_confusion_csv(confusion, class_names, output_dir / "confusion_matrix.csv")
    save_confusion_image(confusion, class_names, output_dir / "confusion_matrix.png")
    return metrics, rows


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        predicted = int(logits.argmax(dim=1).item())
        target = predicted if class_index is None else class_index
        logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), predicted, logits.softmax(dim=1)[0].detach().cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.uint8(np.clip(array, 0, 1) * 255)


def colorize_cam(cam: np.ndarray) -> np.ndarray:
    # Lightweight blue->cyan->yellow->red heat map without matplotlib.
    red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


def save_gradcams(model, samples, transform, device, class_names, output_dir, per_class):
    if per_class <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    # Predictions are part of the filename, so remove stale CAMs from an older
    # evaluation (for example after preprocessing changes) before regenerating.
    for old_image in output_dir.glob("*.jpg"):
        old_image.unlink()
    selected = []
    counts = {index: 0 for index in range(len(class_names))}
    for path, label in samples:
        if counts[label] < per_class:
            selected.append((path, label))
            counts[label] += 1
    gradcam = GradCAM(model, model.features[-1])
    try:
        for number, (path, actual) in enumerate(selected, start=1):
            with Image.open(path) as image:
                tensor = transform(ImageOps.exif_transpose(image).convert("RGB"))
            cam, predicted, probabilities = gradcam(tensor.unsqueeze(0).to(device))
            rgb = tensor_to_rgb(tensor)
            heatmap = colorize_cam(cam)
            overlay = np.uint8(0.55 * rgb + 0.45 * heatmap)
            confidence = probabilities[predicted]
            filename = (
                f"{number:02d}_actual-{class_names[actual]}_"
                f"pred-{class_names[predicted]}_conf-{confidence:.3f}.jpg"
            )
            Image.fromarray(overlay).save(output_dir / filename, quality=95)
    finally:
        gradcam.close()


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name:18s} | n={metrics['samples']:3d} | "
        f"accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
    )


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.crawler_root = args.crawler_root.resolve()
    args.real_root = args.real_root.resolve()
    args.output_dir = args.output_dir.resolve()
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    seed_everything(int(checkpoint.get("seed", 42)))
    class_to_idx = checkpoint["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    _, eval_transform = build_transforms(int(checkpoint.get("image_size", 224)))

    crawler_samples = []
    for relative_path in checkpoint["crawler_test_files"]:
        path = args.crawler_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Split image from checkpoint is missing: {path}")
        crawler_samples.append((path, class_to_idx[Path(relative_path).parts[0]]))
    real_samples = discover_samples(args.real_root, class_to_idx)

    crawler_loader = DataLoader(
        ImagePathDataset(crawler_samples, eval_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    real_loader = DataLoader(
        ImagePathDataset(real_samples, eval_transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    print(f"Device: {device}")
    crawler_metrics, _ = evaluate_dataset(
        model, crawler_loader, device, class_names, args.output_dir / "crawler_holdout"
    )
    real_metrics, _ = evaluate_dataset(
        model, real_loader, device, class_names, args.output_dir / "real_photos"
    )
    print_metrics("Crawler holdout", crawler_metrics)
    print_metrics("Real photos", real_metrics)

    comparison = {
        "crawler_holdout": crawler_metrics,
        "real_photos": real_metrics,
        "accuracy_gap_crawler_minus_real": crawler_metrics["accuracy"] - real_metrics["accuracy"],
        "macro_f1_gap_crawler_minus_real": crawler_metrics["macro_f1"] - real_metrics["macro_f1"],
        "interpretation": (
            "A positive gap indicates performance loss when moving from crawled images "
            "to real photographs. Both sets were evaluated with the same final model."
        ),
    }
    save_json(comparison, args.output_dir / "comparison.json")

    save_gradcams(
        model, crawler_samples, eval_transform, device, class_names,
        args.output_dir / "gradcam" / "crawler_holdout", args.cam_per_class
    )
    save_gradcams(
        model, real_samples, eval_transform, device, class_names,
        args.output_dir / "gradcam" / "real_photos", args.cam_per_class
    )
    print(f"Accuracy gap (crawler - real): {comparison['accuracy_gap_crawler_minus_real']:.4f}")
    print(f"Saved evaluation outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
