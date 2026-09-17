"""ViT(DeiT-Small) 완전 베이스라인 vs 개선점 적용(taehyun 설계)의 Test2 성능 비교 막대그래프를 만든다.

둘 다 data/test1_orientation_backup(126장) 기준 실측치.

실행:
    python src/deep_learning/vit/make_baseline_vs_improved_figure.py
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# (라벨, Test2 정확도, Test2 Macro-F1) — data/test1_orientation_backup(126장) 기준 실측치.
DATA = [
    ("완전 베이스라인\n(패딩/증강/부분freeze/차등LR 없음)", 0.857, 0.856),
    ("개선점 적용\n(패딩+증강+부분freeze+차등LR+WD)", 0.738, 0.729),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/figures/vit/baseline_vs_improved.png")
    args = parser.parse_args()

    labels = [d[0] for d in DATA]
    acc = [d[1] * 100 for d in DATA]
    macro_f1 = [d[2] for d in DATA]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(DATA))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], acc, width, label="Test2 정확도(%)", color="#4C72B0")
    ax2 = ax.twinx()
    bars2 = ax2.bar([i + width / 2 for i in x], macro_f1, width, label="Test2 Macro-F1", color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test2 정확도(%)")
    ax2.set_ylabel("Test2 Macro-F1")
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 1.0)
    ax.set_title("ViT(DeiT-Small) 완전 베이스라인 vs 개선점 적용\n(data/test1_orientation_backup 126장 기준)",
                  fontsize=13, fontweight="bold")

    for b, v in zip(bars1, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    for b, v in zip(bars2, macro_f1):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)

    ax.text(0.5, 0.97, "※ 개선점 적용이 오히려 하락 — 부분 freeze가 도메인 적응을 제한한 것으로 추정",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#B00020",
            bbox=dict(boxstyle="round", fc="#FDECEA", ec="#B00020"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
