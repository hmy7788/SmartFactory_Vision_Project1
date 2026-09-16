"""텀블러 누끼따기(배경 분리) 및 실루엣 마스크 추출, 시각화 유틸리티."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


def imread_unicode(path: Path | str) -> np.ndarray | None:
    """Windows에서 한글 및 비ASCII 경로를 안전하게 지원하는 OpenCV 이미지 로더."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            bytes_data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        # PIL 폴백
        try:
            with Image.open(path) as pil_img:
                rgb = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None


def imwrite_unicode(path: Path | str, img: np.ndarray) -> bool:
    """Windows에서 한글 및 비ASCII 경로를 안전하게 지원하는 OpenCV 이미지 저장기."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    success, encoded = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(encoded)
        return True
    return False


def extract_tumbler_mask(
    image: np.ndarray | Image.Image,
    margin_ratio: float = 0.05,
    iter_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """이미지에서 텀블러 객체를 분리하여 이진 마스크와 RGBA 누끼 이미지를 추출합니다.

    Args:
        image: RGB uint8 numpy 배열 (H, W, 3) 또는 PIL 이미지.
        margin_ratio: GrabCut 초기 사각형 마진 비율 (0.05 = 5%).
        iter_count: GrabCut 반복 횟수.

    Returns:
        binary_mask: (H, W) uint8 배열 (255=텀블러, 0=배경).
        nukki_rgba: (H, W, 4) uint8 배열 (배경은 알파=0 투명 처리).
    """
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image.copy()

    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        rgba = np.dstack([rgb, mask])
        return mask, rgba

    # 대용량 이미지 고속 처리를 위한 리스케일링 (최대 448px)
    max_dim = max(h, w)
    if max_dim > 448:
        scale = 448.0 / max_dim
        proc_w = max(1, int(round(w * scale)))
        proc_h = max(1, int(round(h * scale)))
        proc_rgb = cv2.resize(rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        proc_w, proc_h = w, h
        proc_rgb = rgb

    # 1. GrabCut 초기화 사각형 정의 (중앙 집중 텀블러 구도)
    mx = max(1, int(proc_w * margin_ratio))
    my = max(1, int(proc_h * margin_ratio))
    rect = (mx, my, max(1, proc_w - 2 * mx), max(1, proc_h - 2 * my))

    mask_gc = np.zeros((proc_h, proc_w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            proc_rgb,
            mask_gc,
            rect,
            bgd_model,
            fgd_model,
            iter_count,
            cv2.GC_INIT_WITH_RECT,
        )
        # GC_FGD(1) 또는 GC_PR_FGD(3)를 전경으로 취급
        proc_binary = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        # GrabCut 실패 시 밝기/대비 기반 폴백
        gray = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, proc_binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # 원본 해상도로 마스크 복원
    if scale != 1.0:
        binary_mask = cv2.resize(proc_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        binary_mask = proc_binary

    # 2. 모폴로지 연산으로 구멍 메우기 및 잡음 제거
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_large)

    # 가장 큰 외곽 윤곽선만 유지 (주 텀블러 객체)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(refined_mask, [largest], -1, 255, thickness=cv2.FILLED)
        binary_mask = refined_mask

    # 3. 투명 배경 RGBA 누끼 이미지 생성
    alpha = binary_mask.copy()
    nukki_rgba = np.dstack([rgb, alpha])

    return binary_mask, nukki_rgba


def render_checkerboard(h: int, w: int, cell_size: int = 16) -> np.ndarray:
    """투명 배경 시각화를 위한 체커보드 패턴 생성."""
    rows = (h + cell_size - 1) // cell_size
    cols = (w + cell_size - 1) // cell_size
    board = np.indices((rows, cols)).sum(axis=0) % 2
    board = np.repeat(np.repeat(board, cell_size, axis=0), cell_size, axis=1)[:h, :w]
    bg = np.where(board == 0, 220, 255).astype(np.uint8)
    return np.dstack([bg, bg, bg])


def create_nukki_preview(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    nukki_rgba: np.ndarray | None = None,
    target_size: tuple[int, int] = (250, 250),
) -> np.ndarray:
    """[원본] | [실루엣 마스크] | [배경 분리 누끼] | [윤곽선 오버레이] 4분할 시각화 이미지를 반환합니다."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    rgb = image.copy()
    h, w = rgb.shape[:2]

    if nukki_rgba is None:
        nukki_rgba = np.dstack([rgb, mask])

    # 1. 체커보드 배경 위에 누끼 합성
    checker = render_checkerboard(h, w)
    alpha = (nukki_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
    nukki_on_checker = (nukki_rgba[:, :, :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)

    # 2. 실루엣 마스크 (3채널 RGB)
    mask_rgb = np.dstack([mask, mask, mask])

    # 3. 윤곽선 오버레이
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # 녹색 외곽선

    # 라벨링 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (10, 25), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 25), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.resize(out, target_size, interpolation=cv2.INTER_AREA)

    p1 = add_label(rgb, "1. Original RGB")
    p2 = add_label(mask_rgb, "2. Silhouette Mask")
    p3 = add_label(nukki_on_checker, "3. Extracted Foreground")
    p4 = add_label(overlay, "4. Contour Overlay")

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    combined = np.vstack([top, bottom])
    return combined


def create_masking_variants(
    image: np.ndarray | Image.Image,
    mask: np.ndarray | None = None,
    neutral_color: tuple[int, int, int] = (128, 128, 128),
) -> dict[str, np.ndarray]:
    """형상 의존성 진단을 위한 4가지 변형 이미지를 생성합니다.

    Returns:
        dict: {
            'original': RGB 원본,
            'grayscale': 흑백 변환 (3채널),
            'logo_masked': 텀블러 중심 로고/패턴 가림 (회색 패치),
            'bg_masked': 배경을 중립색으로 가리고 텀블러 몸통만 보존
        }
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    rgb = image.copy()
    h, w = rgb.shape[:2]

    if mask is None:
        mask, _ = extract_tumbler_mask(rgb)

    # 1. 흑백 변환 (RGB 3채널 복제)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_3ch = np.dstack([gray, gray, gray])

    # 2. 로고 영역 가림: 텀블러 바운딩 박스 중심부 가림
    logo_masked = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # 중심 40% 폭, 30% 높이 영역 가림
        patch_w = max(4, int(bw * 0.45))
        patch_h = max(4, int(bh * 0.30))
        cx, cy = bx + bw // 2, by + bh // 2
        px1 = max(0, cx - patch_w // 2)
        py1 = max(0, cy - patch_h // 2)
        px2 = min(w, cx + patch_w // 2)
        py2 = min(h, cy + patch_h // 2)
        logo_masked[py1:py2, px1:px2] = neutral_color
    else:
        # 윤곽 없을 경우 이미지 정중앙 가림
        logo_masked[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)] = neutral_color

    # 3. 배경 가림 (오직 텀블러 몸통만 남기고 배경은 중립색으로 채움)
    bg_masked = np.full_like(rgb, neutral_color)
    fg_indices = (mask > 127)
    bg_masked[fg_indices] = rgb[fg_indices]

    return {
        "original": rgb,
        "grayscale": gray_3ch,
        "logo_masked": logo_masked,
        "bg_masked": bg_masked,
    }
