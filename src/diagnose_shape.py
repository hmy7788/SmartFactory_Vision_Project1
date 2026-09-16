"""형상 의존성 진단 모듈: 흑백 변환, 로고 가림, 배경 가림(누끼) 마스킹에 따른 예측 변화 분석 및 시각화."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transforms import AspectRatioPadResize, IMAGENET_MEAN, IMAGENET_STD
from src.models.vit_classifier import create_deit_model, CLASSES
from src.cv_utils.segmentation import extract_tumbler_mask, create_masking_variants

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"


def load_model_from_checkpoint(checkpoint_path: Path | str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """체크포인트로부터 학습된 가중치와 메타데이터를 로드합니다."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = create_deit_model(pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def preprocess_numpy_for_model(img_np: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Numpy RGB 이미지를 모델 입력 텐서 (1, 3, 224, 224)로 변환합니다."""
    pil_img = Image.fromarray(img_np)
    pad_tool = AspectRatioPadResize(target_size=target_size)
    padded = pad_tool(pil_img)
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


@torch.no_grad()
def diagnose_shape_reliance(
    checkpoint_path: Path | str,
    output_figure: str = "shape_reliance_diagnosis.png",
) -> dict:
    """검증셋 이미지들에 대해 흑백 변환, 로고 가림, 배경 가림 마스킹을 적용하고 예측 변화를 진단합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    df = pd.read_csv(METADATA_CSV)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"Diagnosing shape reliance on {len(val_df)} validation images using {checkpoint_path}...")

    results = {
        "original": {"correct": 0, "probs": []},
        "grayscale": {"correct": 0, "probs": []},
        "logo_masked": {"correct": 0, "probs": []},
        "bg_masked": {"correct": 0, "probs": []},
    }

    sample_visuals = []

    for idx, row in val_df.iterrows():
        img_path = Path(row["full_path"])
        true_label = row["label"]
        true_idx = CLASSES.index(true_label)

        with Image.open(img_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            mask, _ = extract_tumbler_mask(pil_img)
            variants = create_masking_variants(pil_img, mask)

        variant_preds = {}
        for var_name, var_img in variants.items():
            tensor = preprocess_numpy_for_model(var_img).to(device)
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            is_correct = (pred_idx == true_idx)

            results[var_name]["probs"].append(float(probs[true_idx]))
            if is_correct:
                results[var_name]["correct"] += 1

            variant_preds[var_name] = {
                "pred_label": CLASSES[pred_idx],
                "pred_prob": float(probs[pred_idx]),
                "true_prob": float(probs[true_idx]),
                "image": var_img,
            }

        # 클래스별 대표 1장씩 시각화용 저장
        if len(sample_visuals) < 4 and all(v["label"] != true_label for v in sample_visuals):
            sample_visuals.append({
                "label": true_label,
                "variants": variant_preds,
            })

    total = len(val_df)
    summary = {}
    print("\n--- Shape Reliance Diagnosis Results ---")
    for k in ["original", "grayscale", "logo_masked", "bg_masked"]:
        acc = results[k]["correct"] / total
        mean_true_prob = float(np.mean(results[k]["probs"]))
        summary[k] = {"accuracy": round(acc, 4), "mean_target_prob": round(mean_true_prob, 4)}
        print(f"Condition [{k:12s}]: Accuracy = {acc:.3f} | Mean Target Probability = {mean_true_prob:.3f}")

    # 리포트 마크다운 및 JSON 저장
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "shape_reliance_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 시각화 이미지 생성 (4 클래스 x 4 변형 그리드)
    generate_diagnosis_figure(sample_visuals, output_figure)

    return summary


def generate_diagnosis_figure(sample_visuals: list[dict], filename: str):
    """클래스별 샘플에 대한 원본, 흑백, 로고 가림, 배경 가림 변형과 예측 점수를 그리드로 시각화합니다."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_samples = len(sample_visuals)
    if n_samples == 0:
        return

    cols = ["original", "grayscale", "logo_masked", "bg_masked"]
    col_titles = ["1. Original RGB", "2. Grayscale", "3. Logo Masked", "4. Background Masked"]

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, sample in enumerate(sample_visuals):
        true_lbl = sample["label"]
        for col_idx, col_key in enumerate(cols):
            ax = axes[row_idx, col_idx]
            var_data = sample["variants"][col_key]
            img = var_data["image"]
            pred_lbl = var_data["pred_label"]
            pred_prob = var_data["pred_prob"]

            ax.imshow(img)
            ax.axis("off")
            is_match = (pred_lbl == true_lbl)
            color = "green" if is_match else "red"
            title = f"{col_titles[col_idx]}\nPred: {pred_lbl} ({pred_prob:.1%})\nGT: {true_lbl}"
            ax.set_title(title, fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved diagnosis visual figure: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # 최신 가중치 탐색
        candidates = list(CHECKPOINTS_DIR.glob("*.pt"))
        if candidates:
            ckpt_path = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        else:
            print("No checkpoints found. Please train a model first.")
            sys.exit(1)

    diagnose_shape_reliance(ckpt_path)
