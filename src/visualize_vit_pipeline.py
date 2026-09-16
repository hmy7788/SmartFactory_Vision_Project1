"""PPT 슬라이드용 DeiT-Small 텀블러 형태 분류 파이프라인 전체 구조도 생성 스크립트."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor("#fdfdfd")
    ax.set_facecolor("#fdfdfd")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(8, 8.5, "ViT (DeiT-Small/16) 기반 텀블러 4종 형태 분류 파이프라인 구조", 
            fontsize=18, fontweight="bold", ha="center", va="center", color="#1a1a1a")
    ax.text(8, 8.1, "RTX 4050 6GB 최적화 · 종횡비 보존 224 패딩 · 차등 학습률 부분 미세조정 (Partial Fine-Tuning)", 
            fontsize=11, ha="center", va="center", color="#555555")

    # Colors
    c_prep = "#e3f2fd"      # Light Blue (전처리)
    c_patch = "#e8f5e9"     # Light Green (패치)
    c_frozen = "#eeeeee"    # Gray (동결)
    c_train = "#fff3e0"     # Orange (학습)
    c_head = "#f3e5f5"      # Purple (헤드)
    c_border_blue = "#1976d2"
    c_border_green = "#388e3c"
    c_border_gray = "#757575"
    c_border_orange = "#f57c00"
    c_border_purple = "#7b1fa2"

    # Box 1: 입력 및 전처리
    box1 = patches.FancyBboxPatch((0.5, 2.2), 3.2, 5.2, boxstyle="round,pad=0.2", 
                                  fc=c_prep, ec=c_border_blue, lw=2)
    ax.add_patch(box1)
    ax.text(2.1, 7.0, "[1] 입력 및 기하 전처리", fontsize=13, fontweight="bold", ha="center", color=c_border_blue)
    ax.text(2.1, 6.4, "• 원본 텀블러 사진 (다양한 크기)", fontsize=10, ha="center", color="#333")
    ax.text(2.1, 5.9, "• EXIF 자동 회전 보정", fontsize=10, ha="center", color="#333")
    ax.text(2.1, 5.4, "• 100% 종횡비 보존 리사이즈", fontsize=10, fontweight="bold", ha="center", color="#0d47a1")
    ax.text(2.1, 4.9, "• 128 중립 회색 패딩 안착\n  (224 × 224 × 3)", fontsize=10, ha="center", color="#333")
    ax.text(2.1, 4.1, "• 몸통 찌그러짐 0% 방지\n• 임의 크롭/신체 가림 방지", fontsize=9.5, style="italic", ha="center", color="#555")
    ax.text(2.1, 3.2, "• 데이터 증강 (Train):\n  ±10° 회전, 좌우반전, ColorJitter", fontsize=9.5, ha="center", color="#333")

    # Arrow 1 -> 2
    ax.annotate("", xy=(4.1, 4.8), xytext=(3.7, 4.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#1976d2"))

    # Box 2: 패치 분할 & 임베딩
    box2 = patches.FancyBboxPatch((4.2, 2.8), 2.8, 4.0, boxstyle="round,pad=0.2", 
                                  fc=c_patch, ec=c_border_green, lw=2)
    ax.add_patch(box2)
    ax.text(5.6, 6.4, "[2] 패치 분할 & 임베딩", fontsize=12, fontweight="bold", ha="center", color=c_border_green)
    ax.text(5.6, 5.7, "• 16 × 16 픽셀 분할", fontsize=10, fontweight="bold", ha="center", color="#1b5e20")
    ax.text(5.6, 5.2, "• 총 196개 패치 토큰\n  (14 × 14 격자)", fontsize=10, ha="center", color="#333")
    ax.text(5.6, 4.4, "• Linear Projection\n  (차원 D = 384)", fontsize=10, ha="center", color="#333")
    ax.text(5.6, 3.6, "• [CLS] 토큰 (1개) 결합\n• 1D 학습 위치 임베딩 추가\n  -> (197, 384) 텐서", fontsize=9.5, ha="center", color="#333")

    # Arrow 2 -> 3
    ax.annotate("", xy=(7.4, 4.8), xytext=(7.0, 4.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#388e3c"))

    # Box 3: DeiT-Small 백본 (동결 vs 학습 구분)
    box3_outer = patches.FancyBboxPatch((7.5, 1.2), 4.8, 6.2, boxstyle="round,pad=0.2", 
                                        fc="#ffffff", ec="#333333", lw=2.5)
    ax.add_patch(box3_outer)
    ax.text(9.9, 7.0, "[3] DeiT-Small/16 트랜스포머 백본 (12 블록)", fontsize=13, fontweight="bold", ha="center", color="#1a1a1a")

    # Sub-box 3A: 동결 블록 (1~8)
    box3_frozen = patches.FancyBboxPatch((7.7, 4.4), 4.4, 2.2, boxstyle="round,pad=0.15", 
                                         fc=c_frozen, ec=c_border_gray, lw=1.5, ls="--")
    ax.add_patch(box3_frozen)
    ax.text(9.9, 6.1, "Blocks 1 ~ 8 (하위 8개 블록) [동결 : Freeze]", fontsize=11, fontweight="bold", ha="center", color="#424242")
    ax.text(9.9, 5.5, "• Meta ImageNet-1k 사전학습 지식 보존 (1,495만 파라미터)", fontsize=9.5, ha="center", color="#555")
    ax.text(9.9, 5.0, "• 패치 간 국소 엣지 및 기본 윤곽선 특징 추출\n• 파라미터 동결로 소규모 데이터 과적합 원천 차단", fontsize=9.5, ha="center", color="#333")

    # Sub-box 3B: 부분 미세조정 블록 (9~12)
    box3_train = patches.FancyBboxPatch((7.7, 1.5), 4.4, 2.6, boxstyle="round,pad=0.15", 
                                        fc=c_train, ec=c_border_orange, lw=2)
    ax.add_patch(box3_train)
    ax.text(9.9, 3.6, "Blocks 9 ~ 12 + Norm [부분 미세조정 : Fine-Tuning]", fontsize=11, fontweight="bold", ha="center", color=c_border_orange)
    ax.text(9.9, 3.0, "• 저속 미세조정 (lr = 1e-5, AdamW) (710만 파라미터)", fontsize=9.5, fontweight="bold", ha="center", color="#bf360c")
    ax.text(9.9, 2.4, "• 텀블러 전역 기하학 조합 (Global Self-Attention)", fontsize=9.5, ha="center", color="#333")
    ax.text(9.9, 1.8, "• 단차선(Step), 상하 직경 변화율, 종횡비 고수준 형태 학습", fontsize=9.5, ha="center", color="#333")

    # Arrow 3 -> 4
    ax.annotate("", xy=(12.7, 4.8), xytext=(12.3, 4.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#f57c00"))

    # Box 4: 최종 분류 헤드 및 출력
    box4 = patches.FancyBboxPatch((12.8, 1.8), 2.8, 5.6, boxstyle="round,pad=0.2", 
                                  fc=c_head, ec=c_border_purple, lw=2)
    ax.add_patch(box4)
    ax.text(14.2, 7.0, "[4] 텀블러 4종 분류 헤드", fontsize=12, fontweight="bold", ha="center", color=c_border_purple)
    ax.text(14.2, 6.4, "• [CLS] 토큰 벡터 추출 (384-d)", fontsize=9.5, ha="center", color="#333")
    ax.text(14.2, 5.9, "• Linear Layer (384 -> 4)", fontsize=10, fontweight="bold", ha="center", color="#4a148c")
    ax.text(14.2, 5.4, "• 헤드 전용 학습 (lr = 1e-4)", fontsize=9.5, ha="center", color="#bf360c")
    ax.text(14.2, 4.8, "• Softmax 확률 도출", fontsize=9.5, ha="center", color="#333")

    # Class pills
    classes_info = [
        ("straight", "직선 원통형 (벽면 평행)", "#e1f5fe"),
        ("taper_smooth", "연속 테이퍼 (매끄러운 빗면)", "#e8f5e9"),
        ("taper_step", "단차 테이퍼 (꺾임 단차선)", "#fff3e0"),
        ("mug", "머그형 (높이/지름 비율<=1.5)", "#fce4ec"),
    ]
    for i, (cname, cdesc, bg) in enumerate(classes_info):
        y_pos = 4.0 - i * 0.55
        rect = patches.FancyBboxPatch((13.0, y_pos - 0.15), 2.4, 0.45, boxstyle="round,pad=0.08", 
                                      fc=bg, ec="#999", lw=1)
        ax.add_patch(rect)
        ax.text(14.2, y_pos + 0.08, f"'{cname}'", fontsize=9.5, fontweight="bold", ha="center")

    # Bottom Summary Banner
    summary_box = patches.FancyBboxPatch((0.5, 0.3), 15.1, 0.7, boxstyle="round,pad=0.1", 
                                         fc="#263238", ec="none")
    ax.add_patch(summary_box)
    ax.text(8.05, 0.65, "핵심 경쟁력: CNN 경량 모델(MobileNet 71.4%)의 한계를 Self-Attention 전역 윤곽 인식으로 돌파하여 통합 테스트 87.4% / F1 0.88 달성", 
            fontsize=10.5, fontweight="bold", color="#ffffff", ha="center", va="center")

    plt.tight_layout()
    out_path = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\SmartFactory_Vision_Project1\reports\figures\vit_pipeline_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved pipeline architecture diagram to: {out_path}")

if __name__ == "__main__":
    create_pipeline_diagram()
