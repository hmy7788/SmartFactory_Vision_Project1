"""텀블러 형태 판정 데모.

저장소 루트에서:
    streamlit run app/demo.py
    streamlit run app/demo.py -- --checkpoint checkpoints/model

흐름:
  1. 실시간 카메라 피드 — 매 프레임 추론해서 클래스·확신도 오버레이
  2. [📸 캡처] 버튼 → 해당 프레임 확정
  3. 오른쪽에 GradCAM 시각화 + 클래스 해설 표시
  4. [🔄 다시 찍기] 버튼 → 실시간 모드로 복귀
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.explanation import profiles
from src.gradcam import GradCAM, overlay_cam
from src.pipeline import Pipeline, preprocess

# ── 설정 ────────────────────────────────────────────────────────────────────
REVIEW_THR  = 0.70
TITLE       = "텀블러 형태 판정 데모"
CAM_INDEX   = 0       # 웹캠 인덱스 (여러 개면 0, 1, 2... 변경)
FPS_DELAY   = 0.05    # 실시간 루프 간격(초) — 줄이면 빨라짐


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model"))
    return p.parse_known_args(sys.argv[1:])[0]


# ── 모델 & GradCAM 초기화 (캐시) ────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 불러오는 중...")
def load_resources(checkpoint_dir: str, meta_mtime: float):
    pipe = Pipeline(checkpoint_dir)
    # ConvNeXt: features[7] 이 마지막 블록
    # 다른 아키텍처면 target_layer를 바꿔야 함
    try:
        target_layer = pipe.model.features[7]
    except (AttributeError, IndexError):
        target_layer = list(pipe.model.children())[-2]   # fallback
    grad_cam = GradCAM(pipe.model, target_layer)
    return pipe, grad_cam


# ── 한 프레임 GradCAM 생성 ───────────────────────────────────────────────────
def run_gradcam(pipe: Pipeline, grad_cam: GradCAM,
                image_bgr: np.ndarray, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    cam     : (H, W) 히트맵 (0~1)
    overlay : (H, W, 3) RGB uint8 오버레이 이미지
    """
    import torch
    x = torch.from_numpy(preprocess(image_bgr, pipe.meta)).unsqueeze(0).requires_grad_(True)
    cam     = grad_cam.generate(x, class_idx)
    img_rgb = cv2.cvtColor(
        cv2.resize(image_bgr, (pipe.meta["img_size"], pipe.meta["img_size"])),
        cv2.COLOR_BGR2RGB
    )
    ovl = overlay_cam(img_rgb, cam)
    return cam, ovl


# ── 실시간 프레임 오버레이 ───────────────────────────────────────────────────
def draw_overlay(frame_bgr: np.ndarray, pred: dict | None,
                 review_thr: float, class_profiles: dict) -> np.ndarray:
    """클래스명·확신도를 프레임 상단에 그린다."""
    img = frame_bgr.copy()
    if pred is None:
        return img
    label = pred.get("label", "")
    score = pred.get("score", 0.0)
    display = profiles.headline(class_profiles, label)
    color   = (0, 220, 0) if score >= review_thr else (0, 165, 255)

    cv2.rectangle(img, (0, 0), (img.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(img, display,            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(img, f"{score:.0%}",     (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return img


# ── 판정 결과 카드 ───────────────────────────────────────────────────────────
def render_verdict(pred: dict, review_thr: float, class_profiles: dict) -> str | None:
    """판정 상태 표시. 판정 완료면 클래스 키 반환, 아니면 None."""
    label = pred.get("label")
    score = pred.get("score", 0.0)
    probs = pred.get("probs", {})
    infer_ms = pred.get("infer_ms")

    title, _, state = profiles.decide(class_profiles, label, score, review_thr)

    if state == "ok":
        st.markdown(f"### 🟢 {title}")
        st.success(f"확신도 **{score:.0%}** ≥ 기준 {review_thr:.0%}")
    else:
        cands = profiles.top_candidates(probs, profiles=class_profiles)
        st.markdown("### 🟡 판정 보류")
        st.warning(
            f"확신도 **{score:.0%}** < 기준 {review_thr:.0%}"
            + (f"  \n후보: {cands}" if cands else "")
        )

    # 전체 확률 바
    st.markdown("**전체 확률**")
    for c in sorted(probs, key=lambda k: -probs[k]):
        p    = probs[c]
        name = profiles.headline(class_profiles, c)
        st.progress(float(p), text=f"{'▶ ' if c == label else '　'}{name}  {p:.1%}")

    if infer_ms:
        st.caption(f"처리 시간: {infer_ms:.0f} ms")

    return label if state == "ok" else None


# ── 클래스 해설 ──────────────────────────────────────────────────────────────
def render_profile(label: str | None, class_profiles: dict) -> None:
    """판정 완료일 때만 클래스 해설을 표시한다."""
    if label is None:
        return
    info = class_profiles.get(label)
    if not info:
        return
    st.markdown("---")
    if info["rule"]:
        st.caption(f"📏 판정 규칙 — {info['rule']}")
    for field in profiles.FIELDS:
        if info[field]:
            st.markdown(f"**{profiles.FIELD_LABELS[field]}**  \n{info[field]}")


# ════════════════════════════════════════════════════════════════════════════
#  앱 시작
# ════════════════════════════════════════════════════════════════════════════
args = parse_args()
st.set_page_config(page_title=TITLE, layout="wide")

ss = st.session_state
ss.setdefault("live_mode", True)          # True=실시간, False=캡처 확인 중
ss.setdefault("captured_bgr", None)       # 캡처된 BGR 프레임
ss.setdefault("captured_result", None)    # 캡처된 추론 결과

CLASS_PROFILES = profiles.load_profiles()

# 모델 로드
pipe, grad_cam, model_error = None, None, None
try:
    meta_path = Path(args.checkpoint) / "meta.json"
    pipe, grad_cam = load_resources(
        args.checkpoint,
        meta_path.stat().st_mtime if meta_path.is_file() else 0.0
    )
except Exception as e:
    model_error = f"{type(e).__name__}: {e}"

classes = list(pipe.classes) if pipe else []

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    review_thr = st.slider("확신도 기준", 0.25, 0.99, REVIEW_THR, step=0.01,
                           help="이 값 이상이면 판정 완료, 미만이면 판정 보류")
    cam_index  = st.number_input("웹캠 번호", min_value=0, max_value=5,
                                 value=CAM_INDEX, step=1,
                                 help="카메라가 안 열리면 0, 1, 2 순으로 바꿔보세요")
    st.markdown("---")
    with st.expander("읽는 법"):
        st.markdown(
            "- 🟢 **판정 완료** — 확신도 ≥ 기준\n"
            "- 🟡 **판정 보류** — 확신도 < 기준\n\n"
            "텀블러가 화면 중앙에 크게 나오게 비추면 정확도가 올라갑니다."
        )
    if pipe is not None:
        info = pipe.info()
        with st.expander("모델 정보"):
            st.caption(
                f"{info['arch']}  \n"
                f"입력 {info['input']}px · {info['resize']}  \n"
                f"{info['weights']} ({info['size_mb']} MB)"
            )

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.title(f"🥤 {TITLE}")

if model_error:
    st.error(f"모델 로드 실패: `{args.checkpoint}` 에 meta.json과 가중치를 넣어 주세요.")
    with st.expander("자세한 원인"):
        st.text(model_error)
    st.stop()

st.markdown("---")

# ── 버튼 행 ──────────────────────────────────────────────────────────────────
btn_col1, btn_col2, _ = st.columns([1, 1, 5])

if ss.live_mode:
    if btn_col1.button("📸 캡처", type="primary", use_container_width=True):
        ss.live_mode = False          # 실시간 루프 탈출 → 캡처 결과 표시
        st.rerun()
else:
    if btn_col1.button("🔄 다시 찍기", use_container_width=True):
        ss.live_mode      = True
        ss.captured_bgr   = None
        ss.captured_result = None
        st.rerun()

st.markdown("---")

# ── 메인 레이아웃 ─────────────────────────────────────────────────────────────
live_col, result_col = st.columns([1, 1], gap="large")

# ══ 왼쪽: 카메라 피드 ════════════════════════════════════════════════════════
with live_col:
    if ss.live_mode:
        st.subheader("📷 실시간 인식")
        frame_slot  = st.empty()
        status_slot = st.empty()

        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(f"웹캠 {int(cam_index)}번을 열 수 없습니다. 사이드바에서 번호를 바꿔보세요.")
            st.stop()

        # ── 실시간 루프 ──────────────────────────────────────────────────────
        while ss.live_mode:
            ret, frame = cap.read()
            if not ret:
                status_slot.warning("프레임을 읽지 못했습니다.")
                time.sleep(0.1)
                continue

            result = pipe.safe_predict(frame.copy())
            pred   = result.get("pred")

            # 오버레이 그려서 표시
            display_frame = draw_overlay(frame, pred, review_thr, CLASS_PROFILES)
            frame_slot.image(display_frame, channels="BGR", use_container_width=True)

            # 현재 추론 결과를 세션에 저장 (캡처 시 사용)
            ss.captured_bgr    = frame.copy()
            ss.captured_result = result

            time.sleep(FPS_DELAY)

            # Streamlit 버튼 클릭 감지 — 버튼 상태가 바뀌면 rerun 되므로
            # live_mode가 False가 되면 루프 종료
            if not ss.live_mode:
                break

        cap.release()

    else:
        # 캡처 모드 — 왼쪽엔 캡처된 원본 이미지 표시
        st.subheader("📷 캡처된 이미지")
        if ss.captured_bgr is not None:
            img_rgb = cv2.cvtColor(ss.captured_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
        else:
            st.info("캡처된 이미지가 없습니다.")

# ══ 오른쪽: GradCAM + 판정 결과 + 해설 ═══════════════════════════════════════
with result_col:
    if ss.live_mode:
        st.subheader("📊 판정 결과")
        st.info("실시간 인식 중입니다. 📸 캡처 버튼을 누르면 상세 결과가 여기에 나옵니다.")

    else:
        st.subheader("📊 판정 결과")

        result = ss.captured_result
        frame  = ss.captured_bgr

        if result is None or frame is None:
            st.info("캡처된 결과가 없습니다.")
        else:
            pred = result.get("pred")

            if pred is None:
                st.error("🔴 판정 실패")
                with st.expander("원인"):
                    st.text(result.get("error") or "사유 미기록")
            else:
                # ── GradCAM 생성 ───────────────────────────────────────────
                label     = pred["label"]
                class_idx = pipe.classes.index(label)

                with st.spinner("GradCAM 생성 중..."):
                    try:
                        cam, ovl = run_gradcam(pipe, grad_cam, frame, class_idx)
                        st.image(ovl, caption="🔥 GradCAM — 빨간색 영역이 모델이 집중한 부분",
                                 use_container_width=True)
                    except Exception as e:
                        st.warning(f"GradCAM 생성 실패: {e}")

                st.markdown("---")

                # ── 판정 결과 ──────────────────────────────────────────────
                confirmed_label = render_verdict(pred, review_thr, CLASS_PROFILES)

                # ── 클래스 해설 ────────────────────────────────────────────
                render_profile(confirmed_label, CLASS_PROFILES)