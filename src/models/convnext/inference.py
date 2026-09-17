"""
inference.py
ConvNeXt-Tiny 텀블러 형태 분류 — 추론 모듈

사용법 (import):
    from inference import TumblerClassifier
    clf = TumblerClassifier("./checkpoints_v5/best_convnext_v5.pth")
    result = clf.predict("image.jpg")

사용법 (CLI):
    python inference.py --image image.jpg
    python inference.py --image image.jpg --model ./checkpoints_v5/best_convnext_v5.pth
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['mug', 'straight', 'taper_smooth', 'taper_step']
CLASS_DISPLAY = {
    'mug':          'Mug (손잡이형 머그)',
    'straight':     'Straight (원통형 스트레이트)',
    'taper_smooth': 'Smooth Taper (완만한 테이퍼형)',
    'taper_step':   'Step Taper (계단식 테이퍼형)',
}
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── 메인 클래스 ────────────────────────────────────────────────────────────────
class TumblerClassifier:
    """
    ConvNeXt-Tiny 기반 텀블러 형태 분류기.

    Parameters
    ----------
    model_path : str | Path
        학습된 가중치 파일 경로 (best_convnext_v5.pth)
    device : str | None
        'cuda' / 'cpu' / None(자동 선택)
    """

    def __init__(self, model_path: str = "best_convnext_v5.pth",
                 device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self._load_model(model_path)

        # GradCAM hook
        self._gradients   = None
        self._activations = None
        target_layer = self.model.features[7]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    # ── 모델 로드 ───────────────────────────────────────────────────────────────
    def _load_model(self, model_path: str):
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, len(CLASS_NAMES)
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    # ── GradCAM hooks ───────────────────────────────────────────────────────────
    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── 핵심 추론 메서드 ────────────────────────────────────────────────────────
    def predict(self, image_input, return_gradcam: bool = False) -> dict:
        """
        이미지를 받아 분류 결과를 반환합니다.

        Parameters
        ----------
        image_input : str | Path | PIL.Image.Image | np.ndarray
            이미지 경로 또는 이미 로드된 이미지
        return_gradcam : bool
            True이면 GradCAM 히트맵과 오버레이 이미지를 결과에 포함

        Returns
        -------
        dict
            {
                'class':       str,          # 예측 클래스 이름 (예: 'mug')
                'display':     str,          # 한글 표시 이름
                'confidence':  float,        # 0~100 (%)
                'probabilities': {           # 전체 클래스 확률
                    'mug': float, 'straight': float, ...
                },
                # return_gradcam=True 시 추가:
                'gradcam':     np.ndarray,   # (224,224) 0~1 히트맵
                'overlay':     np.ndarray,   # (224,224,3) uint8 오버레이 이미지
            }
        """
        pil_img = self._to_pil(image_input)

        # ── 전처리 ──
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        # ── 추론 (no_grad) ──
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        result = {
            'class':         pred_class,
            'display':       CLASS_DISPLAY[pred_class],
            'confidence':    confidence,
            'probabilities': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        }

        # ── GradCAM (선택) ──
        if return_gradcam:
            cam     = self._generate_gradcam(pil_img, pred_idx)
            img_np  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
            overlay = self._overlay_cam(img_np, cam)
            result['gradcam'] = cam
            result['overlay'] = overlay

        return result

    # ── GradCAM 생성 ────────────────────────────────────────────────────────────
    def _generate_gradcam(self, pil_img: Image.Image, class_idx: int) -> np.ndarray:
        """예측 클래스에 대한 GradCAM 히트맵 반환 (224×224, 0~1)"""
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(self.device).requires_grad_(True)
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    # ── 오버레이 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _overlay_cam(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """GradCAM 히트맵을 원본 이미지에 오버레이"""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    # ── 입력 변환 유틸 ──────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        return Image.open(image_input).convert("RGB")


# ── CLI 실행 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="텀블러 형태 분류기")
    parser.add_argument("--image", required=True, help="이미지 파일 경로")
    parser.add_argument("--model", default="./checkpoints_v5/best_convnext_v5.pth",
                        help="모델 가중치 경로")
    parser.add_argument("--gradcam", action="store_true",
                        help="GradCAM 히트맵 저장 여부")
    args = parser.parse_args()

    print(f"\n📂 이미지: {args.image}")
    print(f"🧠 모델:   {args.model}\n")

    clf    = TumblerClassifier(model_path=args.model)
    result = clf.predict(args.image, return_gradcam=args.gradcam)

    print(f"✅ 예측:     {result['display']}")
    print(f"📊 신뢰도:   {result['confidence']:.1f}%\n")
    print("── 전체 확률 ──────────────────────────")
    for name, prob in sorted(result['probabilities'].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {CLASS_DISPLAY[name]:<30}  {prob*100:5.1f}%  {bar}")

    if args.gradcam and 'overlay' in result:
        out_path = Path(args.image).stem + "_gradcam.jpg"
        overlay_bgr = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, overlay_bgr)
        print(f"\n🔥 GradCAM 저장: {out_path}")
