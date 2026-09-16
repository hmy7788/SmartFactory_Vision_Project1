"""단일 이미지 텀블러 형태 예측 CLI 및 복합 시각화 이미지 생성 모듈."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transforms import AspectRatioPadResize, IMAGENET_MEAN, IMAGENET_STD
from src.models.vit_classifier import create_deit_model, CLASSES
from src.cv_utils.segmentation import extract_tumbler_mask, render_checkerboard

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


class TumblerPredictor:
    """학습된 ViT 모델을 로드하여 단일 이미지에 대한 형태 분류 및 시각화를 수행합니다."""

    def __init__(self, checkpoint_path: Path | str, device: str | None = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.classes = ckpt.get("classes", CLASSES)
        self.model = create_deit_model(pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.pad_tool = AspectRatioPadResize(target_size=224)

    @torch.no_grad()
    def predict(self, image: Image.Image | np.ndarray) -> dict:
        """단일 이미지에 대해 label, score, probs, infer_ms를 반환합니다."""
        t0 = time.perf_counter()

        if isinstance(image, np.ndarray):
            # BGR 또는 RGB 체크
            pil_img = Image.fromarray(image)
        else:
            pil_img = ImageOps.exif_transpose(image).convert("RGB")

        # 1. 종횡비 보존 패딩
        padded_pil = self.pad_tool(pil_img)

        # 2. 정규화
        arr = np.array(padded_pil, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        # 3. 추론
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_label = self.classes[top_idx]
        top_score = float(probs[top_idx])

        infer_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "label": top_label,
            "score": round(top_score, 4),
            "probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "infer_ms": round(infer_ms, 2),
            "padded_image": padded_pil,
            "raw_image": pil_img,
        }

    def generate_visual_prediction(self, image_path: Path | str, output_path: Path | str) -> dict:
        """예측 결과와 함께 [원본] - [224x224 패딩 전처리] - [누끼/배경분리] - [확률 차트] 4단 종합 시각화 이미지를 저장합니다."""
        img_path = Path(image_path)
        with Image.open(img_path) as raw_img:
            raw_pil = ImageOps.exif_transpose(raw_img).convert("RGB")

        pred_res = self.predict(raw_pil)
        label = pred_res["label"]
        score = pred_res["score"]
        probs = pred_res["probs"]

        # 누끼/마스크 추출
        mask, rgba = extract_tumbler_mask(raw_pil)
        h, w = np.array(raw_pil).shape[:2]
        checker = render_checkerboard(h, w)
        alpha = (rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
        nukki_visual = (rgba[:, :, :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)

        # Matplotlib 복합 시각화 플롯 (1행 4열)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # 1. 원본
        axes[0].imshow(raw_pil)
        axes[0].set_title(f"1. Raw Image\n({w}x{h})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. 패딩 전처리
        axes[1].imshow(pred_res["padded_image"])
        axes[1].set_title("2. Aspect-Ratio Padded\n(224x224, Neutral Pad)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. 누끼/실루엣
        axes[2].imshow(nukki_visual)
        axes[2].set_title("3. Foreground Extracted\n(Nukki on Checkerboard)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # 4. 확률 바 차트
        cls_names = list(probs.keys())
        cls_probs = [probs[c] for c in cls_names]
        colors = ["#1baf7a" if c == label else "#2a78d6" for c in cls_names]

        y_pos = np.arange(len(cls_names))
        bars = axes[3].barh(y_pos, cls_probs, color=colors, height=0.55)
        axes[3].set_yticks(y_pos)
        axes[3].set_yticklabels(cls_names, fontsize=10, fontweight="bold")
        axes[3].invert_yaxis()
        axes[3].set_xlim(0, 1.05)
        axes[3].set_xlabel("Probability", fontsize=10)
        axes[3].set_title(f"4. Prediction: {label} ({score:.1%})\nInfer: {pred_res['infer_ms']}ms", fontsize=11, fontweight="bold", color="#1baf7a")
        axes[3].grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            val = bar.get_width()
            axes[3].text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150)
        plt.close()
        print(f"Saved visual prediction: {out_p}")
        return pred_res


def main():
    parser = argparse.ArgumentParser(description="Tumbler Shape Classifier CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--save-vis", type=str, default=None, help="Path to save composite visualization image")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = list(CHECKPOINTS_DIR.glob("*.pt"))
        if not candidates:
            print("Error: No checkpoints found in checkpoints/. Please train a model first.")
            sys.exit(1)
        ckpt_path = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    predictor = TumblerPredictor(ckpt_path)

    if args.save_vis:
        res = predictor.generate_visual_prediction(args.image, args.save_vis)
    else:
        with Image.open(args.image) as img:
            res = predictor.predict(img)

    clean_res = {
        "label": res["label"],
        "score": res["score"],
        "probs": res["probs"],
        "infer_ms": res["infer_ms"],
    }
    print(json.dumps(clean_res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
