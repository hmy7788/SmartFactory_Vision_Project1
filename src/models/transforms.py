"""종횡비 보존 패딩 전처리 및 증강 모듈, 시각적 검증 유틸리티."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NEUTRAL_PAD_COLOR = (128, 128, 128)


class AspectRatioPadResize:
    """텀블러의 몸통 종횡비를 100% 보존하면서 target_size x target_size 캔버스 중앙에 배치하고
    남는 영역을 중립색(회색)으로 패딩합니다.
    """

    def __init__(self, target_size: int = 224, fill_color: tuple[int, int, int] = NEUTRAL_PAD_COLOR):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        # EXIF 자동 회전 보정
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        # 고화질 보간법으로 리사이즈
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)

        # 224x224 중립색 캔버스 생성 후 중앙 배치
        canvas = Image.new("RGB", (self.target_size, self.target_size), self.fill_color)
        paste_x = (self.target_size - nw) // 2
        paste_y = (self.target_size - nh) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas


def get_train_transforms(target_size: int = 224) -> T.Compose:
    """학습용 전처리: 종횡비 보존 패딩 + 약한 회전/반전/색상 증강."""
    return T.Compose([
        AspectRatioPadResize(target_size=target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, fill=NEUTRAL_PAD_COLOR),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(target_size: int = 224) -> T.Compose:
    """검증/테스트용 전처리: 결정적(Deterministic) 종횡비 보존 패딩 및 정규화."""
    return T.Compose([
        AspectRatioPadResize(target_size=target_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """정규화된 (C, H, W) 텐서를 [0, 255] uint8 RGB numpy 배열 (H, W, C)로 역변환."""
    t = tensor.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr


def create_transform_preview(image: Image.Image | np.ndarray) -> np.ndarray:
    """사용자가 전처리 및 증강 과정을 눈으로 확인할 수 있도록
    [원본] - [224x224 패딩] - [증강 샘플 1] - [증강 샘플 2] 4장을 나란히 합성한 이미지를 생성합니다.
    """
    import cv2

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = ImageOps.exif_transpose(image).convert("RGB")

    # 1. 원본 (224x224 바운딩 박스 내 비율 유지 리사이즈로 표시)
    orig_w, orig_h = pil_img.size
    pad_tool = AspectRatioPadResize(224)
    padded_img = pad_tool(pil_img)

    # 2. 증강 샘플 2개 생성
    aug_tool = get_train_transforms(224)
    aug1 = denormalize_tensor(aug_tool(pil_img))
    aug2 = denormalize_tensor(aug_tool(pil_img))

    # PIL -> Numpy
    orig_np = np.array(padded_img)
    orig_raw_np = np.array(pil_img)

    # 시각화 텍스트 오버레이
    def label_panel(img_np: np.ndarray, text: str, subtext: str = "") -> np.ndarray:
        p = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.putText(p, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(p, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if subtext:
            cv2.putText(p, subtext, (8, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(p, subtext, (8, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        return p

    p1 = label_panel(orig_raw_np, "1. Raw Input", f"{orig_w}x{orig_h}")
    p2 = label_panel(orig_np, "2. Aspect-Pad 224", "Neutral (128) Pad")
    p3 = label_panel(aug1, "3. Augment #1", "Flip / Rotate / Jitter")
    p4 = label_panel(aug2, "4. Augment #2", "Randomized")

    grid = np.hstack([p1, p2, p3, p4])
    return grid
