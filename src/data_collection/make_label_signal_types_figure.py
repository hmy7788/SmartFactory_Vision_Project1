"""라벨 품질 필터링의 3가지 신호 유형(강한/약한/문제없음)을 원본+마스크와 함께 보여준다.

label_quality_report.md의 정의:
    강한 신호  — 룰베이스 + Mask R-CNN 둘 다 폴더 라벨과 다르게 판정 -> 제외
    약한 신호  — 둘 중 하나만 다르게 판정 -> 애매한 케이스일 수 있어 포함 유지
    문제 없음  — 둘 다 폴더 라벨과 동일하게 판정 -> 포함

각 예시마다 원본 사진 + 룰베이스 마스크(get_mask())를 나란히 보여줘서,
"마스크가 왜 저렇게 나왔길래 저런 판정이 나왔는지"까지 짐작할 수 있게 한다.

실행:
    python src/data_collection/make_label_signal_types_figure.py
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rule_based"))
from shape_classifier import get_mask, SHAPE_LABELS_KO  # noqa: E402

import numpy as np  # noqa: E402

# (raw2 내 경로, 폴더 라벨, 룰베이스 판정, Mask R-CNN 판정, 신호 유형) — label_quality_report.md에서 발췌.
EXAMPLES = [
    ("taper_step/64.jpg", "taper_step", "mug", "mug", "강한 신호 (제외)"),
    ("straight/100.jpg", "straight", "mug", "straight", "약한 신호 (포함 유지)"),
    ("straight/1.jpg", "straight", "straight", "straight", "문제 없음 (포함)"),
]

SIGNAL_COLORS = {
    "강한 신호 (제외)": "#B00020",
    "약한 신호 (포함 유지)": "#B8860B",
    "문제 없음 (포함)": "#1B5E20",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw2-root", default="data/raw2")
    parser.add_argument("--out", default="reports/figures/data_pipeline/label_signal_types_examples.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(8, 4 * len(EXAMPLES)))
    fig.suptitle("라벨 품질 필터링 — 신호 유형별 예시 (원본 + 룰베이스 마스크)", fontsize=15, fontweight="bold")

    for row, (rel_path, folder_label, rule_pred, maskrcnn_pred, signal) in enumerate(EXAMPLES):
        path = os.path.join(args.raw2_root, rel_path)
        print(f"처리 중: {path} ({signal})")
        image_bgr = imread_unicode(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = get_mask(image_bgr)

        color = SIGNAL_COLORS[signal]

        ax_img = axes[row, 0]
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title(
            f"[{signal}]\n"
            f"폴더 라벨: {SHAPE_LABELS_KO[folder_label]} / "
            f"룰베이스: {SHAPE_LABELS_KO[rule_pred]} / "
            f"Mask R-CNN: {SHAPE_LABELS_KO[maskrcnn_pred]}",
            fontsize=10, color=color, loc="left")

        ax_mask = axes[row, 1]
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.axis("off")
        ax_mask.set_title("룰베이스 마스크", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
