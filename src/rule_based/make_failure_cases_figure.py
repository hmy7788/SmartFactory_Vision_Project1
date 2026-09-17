"""룰베이스 마스크 파이프라인의 대표적인 실패 사례 3종을 원본+마스크로 보여준다.

docs/troubleshooting.md에서 실제로 진단됐던 케이스들:
    1) 저대비 배경 — 흰색/크림색 물체 + 흰색/연회색 배경, Otsu가 거의 못 잡음
    2) 복잡한 배경 — 그림자·받침대 등이 마스크에 같이 섞여 폭 프로파일이 왜곡됨
    3) 손잡이 구멍 메움 — 손잡이 그립 구멍이 segmentation 단계에서 이미 메워져 폭이 부풀려짐

실행:
    python src/rule_based/make_failure_cases_figure.py
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
from shape_classifier import classify_shape, get_mask, SHAPE_LABELS_KO  # noqa: E402

CASES = [
    {
        "path": "data/preprocess/straight/148.jpg",
        "true_cls": "straight",
        "title": "① 저대비 배경",
        "desc": "흰색 물체 + 흰색/연회색 배경\n마스크 전경 비율 2.6%까지 하락\n(명도 대비가 거의 없어 구분 불가, mug로 오분류)",
    },
    {
        "path": "data/preprocess/taper_step/116.jpg",
        "true_cls": "taper_step",
        "title": "② 비스듬히 눕혀진 촬영 각도",
        "desc": "물체가 세로가 아니라 대각선으로 누워 촬영됨\n세로 10구간 분할 전제 자체가 깨짐\n(아래/위 폭비 22.48로 폭주, mug로 오분류)",
    },
    {
        "path": "data/preprocess/taper_smooth/161.jpg",
        "true_cls": "taper_smooth",
        "title": "③ 임계값 경계 케이스",
        "desc": "마스크는 깨끗하지만 연속/단차 경계가 애매\n단차 비율이 임계값(0.29) 근처에서 살짝 넘어감\n(세그멘테이션이 아니라 판정 로직 자체의 한계)",
    },
]


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/figures/rule_based/failure_cases.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(len(CASES), 2, figsize=(9, 5.0 * len(CASES)), constrained_layout=True)
    fig.suptitle("룰베이스 마스크 파이프라인의 대표적인 실패 사례", fontsize=16, fontweight="bold")

    for row, case in enumerate(CASES):
        print(f"처리 중: {case['path']}")
        image_bgr = imread_unicode(case["path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = get_mask(image_bgr)
        pred = classify_shape(mask)["shape"] if mask.any() else "none"

        ax_img = axes[row, 0]
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title(f"{case['title']}\n{case['desc']}", fontsize=10, color="#B00020", loc="left")

        ax_mask = axes[row, 1]
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.axis("off")
        pred_ko = SHAPE_LABELS_KO.get(pred, pred)
        true_ko = SHAPE_LABELS_KO[case["true_cls"]]
        mark = "O" if pred == case["true_cls"] else "X"
        ax_mask.set_title(f"마스크 -> 판정: {pred_ko} ({mark}, 실제: {true_ko})", fontsize=10)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
