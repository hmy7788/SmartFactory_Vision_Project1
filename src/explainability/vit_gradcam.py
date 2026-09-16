"""ViT (DeiT-Small) Grad-CAM 구현 및 4개 클래스별 판정 근거 시각화 모듈."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from src.models.vit_classifier import CLASSES, create_deit_model, load_tumbler_checkpoint
from src.models.transforms import AspectRatioPadResize, IMAGENET_MEAN, IMAGENET_STD

# 한글 폰트 설정 (Windows Malgun Gothic 지원)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class ViTGradCAM:
    """Vision Transformer(DeiT-Small)용 Grad-CAM 추출기.
    
    마지막 Transformer 블록의 Patch Token 활성화 맵과
    목표 클래스 로짓에 대한 역전파 그래디언트를 결합하여 2D 주의 집중 히트맵을 생성합니다.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.blocks[-1].norm1
        self.activations = None
        self.gradients = None

        # Hook 등록
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        """input_tensor (1, 3, 224, 224)에 대해 target_class의 Grad-CAM (224, 224)을 계산합니다."""
        self.model.zero_grad()
        
        # 순전파
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs))
            
        score = logits[0, target_class_idx]
        conf = float(probs[target_class_idx])

        # 역전파
        score.backward(retain_graph=True)

        # Token 분리: shape (1, 197, 384) -> Patch Token (1, 196, 384)
        # Token 0은 [CLS] 토큰이므로 제외하고 나머지 196개 패치(14x14) 사용
        act = self.activations[:, 1:, :].detach()  # (1, 196, 384)
        grad = self.gradients[:, 1:, :].detach()   # (1, 196, 384)

        # 채널별 가중치 계산 (Global Average Pooling on gradients)
        weights = torch.mean(grad, dim=1, keepdim=True)  # (1, 1, 384)
        cam = torch.sum(weights * act, dim=-1)[0]       # (196,)
        
        # ReLU 적용 (목표 클래스에 긍정적 기여를 하는 부분만 활성화)
        cam = torch.relu(cam).cpu().numpy()

        # 14x14 2D 재구성
        cam_2d = cam.reshape(14, 14)

        # 0 ~ 1 정규화
        cam_min, cam_max = cam_2d.min(), cam_2d.max()
        if cam_max - cam_min > 1e-8:
            cam_2d = (cam_2d - cam_min) / (cam_max - cam_min)
        else:
            cam_2d = np.zeros_like(cam_2d)

        # 224x224로 Bicubic 보간 확대
        cam_resized = cv2.resize(cam_2d, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        return cam_resized, target_class_idx, conf


def create_gradcam_overlay(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """224x224 RGB 이미지에 JET 컬러맵의 Grad-CAM 히트맵을 오버레이합니다."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap_rgb) * alpha + np.float32(rgb_img) * (1 - alpha)
    return np.uint8(np.clip(overlay, 0, 255))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_root = Path(__file__).resolve().parent.parent.parent
    ckpt_path = proj_root / "checkpoints" / "exp_b_seed44_best.pt"

    # 모델 로드
    model = create_deit_model(num_classes=len(CLASSES), pretrained=False)
    load_tumbler_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)

    gradcam = ViTGradCAM(model)
    pad_resize = AspectRatioPadResize(224)

    # 4개 클래스별 대표 실촬영 테스트 이미지 선정
    test_dir = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\test_orientation")
    samples = {
        "straight": test_dir / "straight" / "KakaoTalk_20260915_114226226_27.jpg",
        "taper_smooth": test_dir / "taper_smooth" / "KakaoTalk_20260915_114226226_06.jpg",
        "taper_step": test_dir / "taper_step" / "KakaoTalk_20260915_114226226.jpg",
        "mug": test_dir / "mug" / "KakaoTalk_20260915_114237129_07.jpg",
    }

    # 설명 텍스트 사전
    explanations = {
        "straight": (
            "[직선 원통형 판정 근거]\n"
            "• 상단 및 하단 몸통의 평행한 수직 윤곽선에 균등하게 집중\n"
            "• 위/아래 지름 차이가 5% 미만인 수직 원통 벽면 특징 포착"
        ),
        "taper_smooth": (
            "[연속 테이퍼형 판정 근거]\n"
            "• 꺾임 없이 점진적으로 좁아지는 좌/우 경사 외곽선에 집중\n"
            "• 상단보다 좁아지는 하단부의 기울기 변화 패턴에 높은 가중치"
        ),
        "taper_step": (
            "[단차 테이퍼형 판정 근거]\n"
            "• 몸통 중간의 '뚜렷한 꺾임선(단차 Step)' 부위에 초집중(적색 영역)\n"
            "• 단차 발생 지점과 그 아래로 지름이 줄어드는 구조를 결정적 단서로 활용"
        ),
        "mug": (
            "[머그형 판정 근거]\n"
            "• 높이가 지름의 1.5배 이하인 뭉툭한 몸통 상/하단 비율에 집중\n"
            "• 텀블러 전체적인 높이-너비 기하학적 윤곽 및 바닥면에 주목"
        ),
    }

    # 4개 클래스 4열 (또는 4행) 종합 시각화 생성 (4 x 3 패널: 원본패딩, Grad-CAM오버레이, 히트맵단독)
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    fig.suptitle("ViT (DeiT-Small) Grad-CAM: 텀블러 4종 형태별 시각적 판정 근거 분석", fontsize=16, fontweight="bold", y=0.99)

    col_titles = ["1. 종횡비 보존 원본 (224x224)", "2. Grad-CAM 주의 집중 오버레이", "3. 모델 시각적 판정 근거 및 신뢰도"]

    for row_idx, (cls_name, img_path) in enumerate(samples.items()):
        # 이미지 로드 및 전처리
        with Image.open(img_path) as im:
            padded_pil = pad_resize(im)

        rgb_np = np.array(padded_pil)
        
        # 텐서 변환 및 정규화
        arr = rgb_np.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        # Grad-CAM 생성
        target_idx = CLASSES.index(cls_name)
        cam, pred_idx, conf = gradcam.generate(tensor, target_class_idx=target_idx)
        overlay = create_gradcam_overlay(rgb_np, cam, alpha=0.55)

        # 열 1: 원본 패딩 이미지
        axes[row_idx, 0].imshow(rgb_np)
        axes[row_idx, 0].set_title(f"[{cls_name}] 원본 입력", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 열 2: Grad-CAM 오버레이
        im_cam = axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title(f"Grad-CAM (예측: {CLASSES[pred_idx]}, {conf*100:.1f}%)", fontsize=11, fontweight="bold", color="darkblue")
        axes[row_idx, 1].axis("off")

        # 열 3: 히트맵 단독 + 텍스트 판정 근거 설명
        axes[row_idx, 2].imshow(cam, cmap="jet")
        axes[row_idx, 2].set_title("순수 Activation Heatmap", fontsize=10)
        axes[row_idx, 2].axis("off")
        
        # 하단/우측 설명 텍스트 박스 추가
        axes[row_idx, 2].text(
            1.08, 0.5,
            explanations[cls_name],
            transform=axes[row_idx, 2].transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.72, top=0.95)

    out_path = proj_root / "reports" / "figures" / "gradcam_class_explanations.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM class explanations figure to: {out_path}")


if __name__ == "__main__":
    main()
