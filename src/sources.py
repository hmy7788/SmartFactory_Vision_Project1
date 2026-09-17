"""입력 소스: 사진 폴더와 웹캠. 웹캠은 실제로 열리는지 확인해서 없으면 목록에서 뺀다."""
import sys
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FOLDER, WEBCAM = "폴더", "웹캠"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

try:  # 웹캠이 없을 때 OpenCV가 찍는 경고 줄이기
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def _open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # 윈도우 기본(MSMF)은 여는 데 수 초
    return cv2.VideoCapture(index, backend)


def webcam_available(index: int = 0) -> bool:
    cap = _open_camera(index)
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


def available_sources(probe: Callable[[], bool] = webcam_available) -> list[str]:
    """폴더는 항상, 웹캠은 probe()가 True일 때만."""
    return [FOLDER, WEBCAM] if probe() else [FOLDER]


def list_images(folder: Path | str) -> list[str]:
    folder = Path(folder)
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_image(path: Path | str) -> np.ndarray:
    """cv2.imread와 같은 BGR uint8. 윈도우 한글 경로도 되게 imdecode."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path}: 이미지로 디코딩 실패")
    return image


def folder_frames(folder: Path | str) -> Iterator[tuple[str, np.ndarray]]:
    folder = Path(folder)
    for name in list_images(folder):
        yield name, read_image(folder / name)


def webcam_frames(index: int = 0) -> Iterator[tuple[str, np.ndarray]]:
    """읽기가 실패할 때까지. 소비를 멈추면 close()로 카메라를 놓는다."""
    cap = _open_camera(index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield f"webcam/{datetime.now():%Y%m%d_%H%M%S_%f}.jpg", frame
    finally:
        cap.release()
