"""OpenCV 룰베이스 트랙: 텀블러 형태(4종) 분류.

판정 순서(README "분류 체계" 기준, 반드시 이 순서):
1) 높이/지름 <= MUG_HEIGHT_TO_DIAMETER_MAX  -> mug
2) 아래폭/위폭 >= STRAIGHT_BOTTOM_TOP_RATIO_MIN (거의 수직) -> straight
3) 폭 감소가 한 구간에 몰리지 않고 고르게 분산 -> taper_smooth
4) 폭 감소가 한 구간에 뚜렷하게 몰림(꺾임) -> taper_step

손잡이 유무(detect_handle)는 판정에 전혀 관여하지 않는다 — "실루엣이 손잡이보다
우선한다"는 규칙을 코드 구조로 강제하기 위해, classify_shape()의 분기 조건에는
handle_detected를 절대 사용하지 않는다. 손잡이 정보는 설명 텍스트 등 부가 정보로만
반환한다.
"""

import cv2
import numpy as np

# 임계값 — 실측 데이터 확보 후 재조정 필요 (TUNE_ME)
MUG_HEIGHT_TO_DIAMETER_MAX = 1.5
STRAIGHT_BOTTOM_TOP_RATIO_MIN = 0.95

SHAPE_LABELS_KO = {
    "mug": "머그형",
    "straight": "직선 원통형",
    "taper_smooth": "연속 테이퍼형",
    "taper_step": "단차 테이퍼형",
}


def describe_result_ko(result: dict) -> str:
    """classify_shape()의 반환값을 사람이 읽기 좋은 한글 설명으로 바꾼다."""
    shape_ko = SHAPE_LABELS_KO.get(result["shape"], result["shape"])
    profile_str = ", ".join(f"{w:.1f}" for w in result["width_profile"])
    handle_ko = "예" if result["handle_detected"] else "아니오"
    return (
        f"형태            : {shape_ko} ({result['shape']})\n"
        f"높이/지름 비율   : {result['height_diameter_ratio']:.2f}\n"
        f"아래폭/위폭 비율 : {result['bottom_top_ratio']:.2f}\n"
        f"손잡이 감지      : {handle_ko}\n"
        f"폭 프로파일(위→아래) : [{profile_str}]"
    )
STEP_JUMP_RATIO_THRESHOLD = 0.4
HANDLE_DEPTH_RATIO_THRESHOLD = 0.08


def get_mask(image: np.ndarray) -> np.ndarray:
    """이미지에서 전경(텀블러) 이진 마스크를 추출한다.

    Otsu 임계값 + 최대 외곽 컨투어 기반의 단순한 방법이라, 배경이 단일 색이고
    대비가 뚜렷할 때만 안정적으로 동작한다. 실제 촬영 프로토콜은 배경을 3종
    이상 요구하므로, 실사진 검수 단계에서 GrabCut 등 더 견고한 세그멘테이션으로
    교체·보완이 필요하다 (알려진 한계, 이번 스코프 밖).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def compute_width_profile(mask: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """마스크를 세로로 n_bins등분해 구간별 폭을 반환한다.

    index 0 = 몸통 위쪽(입구) 방향, index -1 = 아래쪽(바닥) 방향.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("mask에 전경 픽셀이 없습니다")

    y_min, y_max = ys.min(), ys.max()
    edges = np.linspace(y_min, y_max + 1, n_bins + 1)
    widths = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        band = (ys >= edges[i]) & (ys < edges[i + 1])
        if not np.any(band):
            continue
        band_xs = xs[band]
        widths[i] = float(band_xs.max() - band_xs.min() + 1)
    return widths


def detect_handle(mask: np.ndarray, depth_ratio_threshold: float = HANDLE_DEPTH_RATIO_THRESHOLD) -> bool:
    """convexHull + convexityDefects로 비대칭 돌출부(손잡이)를 검출한다.

    알려진 한계: 단차 테이퍼형(taper_step)의 꺾임 지점도 깊은 concavity를 만들어
    True로 잡힐 수 있다 (실제 손잡이와 기하학적으로 구분이 안 됨). classify_shape()는
    이 값을 판정에 쓰지 않으므로 분류 결과에는 영향 없음 — 설명 텍스트 등 부가
    정보로 쓸 때만 주의. 실측 데이터로 재조정 필요.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return False

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return False
    hull_indices = np.sort(hull_indices, axis=0)

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return False
    if defects is None:
        return False

    _, _, w, h = cv2.boundingRect(contour)
    scale = max(w, h)
    if scale == 0:
        return False

    for defect in defects.reshape(-1, 4):
        depth_fixed_point = defect[3]
        depth = depth_fixed_point / 256.0
        if depth / scale >= depth_ratio_threshold:
            return True
    return False


def _median_smooth(profile: np.ndarray, window: int = 5) -> np.ndarray:
    """폭 프로파일에 1D median filter를 적용해, 손잡이 돌출부처럼 한 구간만 튀는
    값을 억제한다 (다중 구간에 걸친 진짜 단차는 그대로 유지됨). 손잡이 존재
    여부로 분기하는 게 아니라, 몸통 실루엣 측정 자체를 손잡이에 견고하게
    만드는 방식이라 "실루엣이 손잡이보다 우선" 원칙과 어긋나지 않는다.
    """
    pad = window // 2
    padded = np.pad(profile, pad, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(profile))])


def classify_shape(mask: np.ndarray, n_bins: int = 10) -> dict:
    """폭 프로파일만으로 형태(4종)를 판정한다. handle_detected는 참고 정보일 뿐 판정에 쓰이지 않는다."""
    profile = compute_width_profile(mask, n_bins=n_bins)
    smoothed = _median_smooth(profile)

    ys, _ = np.nonzero(mask)
    height = float(ys.max() - ys.min() + 1)
    diameter = float(smoothed.max())
    height_diameter_ratio = height / diameter if diameter > 0 else float("inf")

    top_width = float(smoothed[0])
    bottom_width = float(smoothed[-1])
    bottom_top_ratio = bottom_width / top_width if top_width > 0 else 0.0

    handle_detected = detect_handle(mask)

    if height_diameter_ratio <= MUG_HEIGHT_TO_DIAMETER_MAX:
        shape = "mug"
    elif bottom_top_ratio >= STRAIGHT_BOTTOM_TOP_RATIO_MIN:
        shape = "straight"
    else:
        deltas = smoothed[:-1] - smoothed[1:]  # 양수 = 아래로 가며 폭이 줄어듦
        total_drop = float(smoothed[0] - smoothed[-1])
        step_ratio = float(deltas.max() / total_drop) if total_drop > 0 else 0.0
        shape = "taper_step" if step_ratio > STEP_JUMP_RATIO_THRESHOLD else "taper_smooth"

    return {
        "shape": shape,
        "height_diameter_ratio": height_diameter_ratio,
        "bottom_top_ratio": bottom_top_ratio,
        "width_profile": profile.tolist(),
        "handle_detected": handle_detected,
    }
