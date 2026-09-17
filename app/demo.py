"""텀블러 형태 → 건축 양식 판정 (보는 사람용 화면).

저장소 루트에서:
    streamlit run app/demo.py
    streamlit run app/demo.py -- --checkpoint checkpoints/model --folder data/test

구성
  - 모델은 checkpoints/<폴더>/meta.json + 가중치 하나. 고르는 화면이 없다.
  - 이력 저장 없음. 지표·로그 없음 — 보는 사람이 알 필요 없는 것.
  - 사진 한 장 올리는 흐름이 첫 화면. 카메라·폴더는 두 번째 탭.
  - 판정 아래에 그 양식의 해설(src/explanation/class_profiles.yaml).

판정 규칙(src/explanation/profiles.decide)은 모델 출력과 다른 층이다 — 확신도에 기준을 걸어
사후에 정한다. 기준 미만이면 '판정 보류'로 두고 어떤 양식의 해설도 내지 않는다(화면이 자기를
반박하지 않게).
"""
import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # 저장소 루트 (app/의 부모)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import sources  # noqa: E402
from src.explanation import profiles  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

REVIEW_THR = 0.70     # 판정 확신 기준 기본값
MAX_RECORDS = 5000

TITLE = "텀블러 형태로 읽는 건축 양식"
SUBTITLE = ("텀블러의 실루엣을 건축물의 형태로 보고, 어떤 양식 계열인지 판정한 뒤 그 양식의 배경을 설명합니다. "
            "이 프로젝트에서 텀블러는 건축 양식 분류의 대리 도메인입니다.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model"))
    p.add_argument("--folder", default=str(ROOT / "data" / "test"))
    return p.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner=False)
def detect_sources() -> list[str]:
    return sources.available_sources()


@st.cache_resource(show_spinner="판정 모델 불러오는 중")
def get_pipeline(checkpoint_dir: str, meta_mtime: float) -> Pipeline:
    return Pipeline(checkpoint_dir)


def on_start() -> None:
    st.session_state.running = True
    st.session_state.records.clear()
    st.session_state.shown_label = None


def on_stop() -> None:
    st.session_state.running = False


# ---------------------------------------------------------------- 시작
args = parse_args()
st.set_page_config(page_title=TITLE, layout="wide")
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("records", deque(maxlen=MAX_RECORDS))
ss.setdefault("last_image", None)
ss.setdefault("shown_label", None)

CLASS_PROFILES = profiles.load_profiles()

pipe, model_error = None, None
try:
    meta_path = Path(args.checkpoint) / "meta.json"
    pipe = get_pipeline(args.checkpoint, meta_path.stat().st_mtime if meta_path.is_file() else 0.0)
except Exception as e:   # meta 없음·가중치 없음·키 불일치·torch 없음 전부 여기로
    model_error = f"{type(e).__name__}: {e}"
classes = list(pipe.classes) if pipe else []

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("판정 설정")
    review_thr = st.slider("판정 확신 기준", 0.25, 0.99, REVIEW_THR, step=0.01, key="thr",
                           disabled=ss.running,
                           help="모델의 확신도가 이 값 이상일 때만 양식을 확정합니다. "
                                "미만이면 '판정 보류'로 두고 후보만 보여줍니다.")
    with st.expander("이 화면 읽는 법"):
        st.markdown(
            "- 🟢 **판정 완료** — 확신도가 기준 이상. 양식과 해설이 나옵니다.\n"
            "- 🟡 **판정 보류** — 확신도가 기준 미만. 확실하지 않을 땐 답하지 않고 후보만 보여줍니다.\n"
            "- 🔴 **판정 실패** — 사진을 처리하지 못했습니다.\n"
            "- 확신도는 '모델이 헷갈리는 정도'입니다. 물체가 너무 작거나 배경이 복잡하면 "
            "확신하면서 틀릴 수도 있으니, 텀블러가 화면에 크게 나오게 찍어 주세요.")
    if pipe is not None:
        info = pipe.info()
        with st.expander("모델 정보"):
            st.caption(f"{info['arch']} · 입력 {info['input']}px · {info['resize']} · "
                       f"{info['weights']} ({info['size_mb']} MB)")
            st.caption(f"클래스 순서: {info['classes']}")
    if not CLASS_PROFILES:
        st.caption(f"해설 파일이 없습니다: {profiles.PROFILE_PATH.name}")
    elif classes:
        _missing = profiles.missing_classes(CLASS_PROFILES, classes)
        if _missing:
            st.caption(f"해설이 없는 양식: {', '.join(_missing)}")

# ---------------------------------------------------------------- 본문
st.title(TITLE)
st.caption(SUBTITLE)
if model_error:
    st.warning(f"판정 모델이 아직 준비되지 않았습니다. `{args.checkpoint}`에 meta.json과 가중치를 넣어 주세요.")
    with st.expander("자세한 원인"):
        st.text(model_error)

tab_photo, tab_live = st.tabs(["사진으로 판정", "카메라·폴더로 판정"])


def probs_bar(probs: dict | None, top: str | None) -> str:
    if not probs or not classes:
        return ""
    width = max(len(profiles.headline(CLASS_PROFILES, c)) for c in classes)
    lines = []
    for c in classes:
        v = float(probs.get(c, 0.0))
        name = profiles.headline(CLASS_PROFILES, c)
        lines.append(f"{'>' if c == top else ' '} {name:<{width}} {v:5.1%} {'█' * int(round(v * 24))}")
    return "\n".join(lines)


def render_verdict(container, record: dict) -> str | None:
    """판정 카드를 그리고, 양식이 확정됐으면 그 클래스 키를, 아니면 None을 돌려준다."""
    pred = record.get("pred")
    with container.container():
        if pred is None:
            st.error("🔴 판정 실패 — 사진을 처리하지 못했습니다.")
            with st.expander("자세한 원인"):
                st.text(record.get("error") or "사유 미기록")
            return None
        label, score = pred.get("label"), pred.get("score")
        title, _, state = profiles.decide(CLASS_PROFILES, label, score, review_thr)
        if state == "ok":
            st.markdown(f"### 🟢 {title}")
            st.success(f"판정 완료 · 확신도 {score:.0%} (기준 {review_thr:.0%})")
        else:
            cands = profiles.top_candidates(pred.get("probs"), profiles=CLASS_PROFILES)
            st.markdown("### 🟡 판정 보류")
            st.warning(f"확신도 {0 if score is None else score:.0%}가 기준 {review_thr:.0%}에 못 미칩니다."
                       + (f"  후보: {cands}" if cands else ""))
        bar = probs_bar(pred.get("probs"), label)
        if bar:
            st.text(bar)
        if record.get("infer_ms") is not None:
            st.caption(f"처리 {record['infer_ms']:.0f} ms")
        return label if state == "ok" else None


def render_profile(container, key: str | None) -> None:
    """양식 해설. 보류했을 땐 아무 양식의 설명도 내지 않는다."""
    with container.container():
        if key is None:
            return
        info = CLASS_PROFILES.get(key)
        if not info:
            st.caption(f"{key}의 해설이 {profiles.PROFILE_PATH.name}에 없습니다.")
            return
        if info["rule"]:
            st.caption(f"판정 규칙 — {info['rule']}")
        cols = st.columns(len(profiles.FIELDS), gap="medium")
        for col, field in zip(cols, profiles.FIELDS):
            if info[field]:
                col.markdown(f"**{profiles.FIELD_LABELS[field]}**  \n{info[field]}")


def make_record(image: np.ndarray, filename: str) -> dict:
    return {"filename": filename, "t": time.perf_counter(), **pipe.safe_predict(image.copy())}


# ---------------------------------------------------------------- 탭 1: 사진 한 장
with tab_photo:
    up = st.file_uploader("텀블러 사진을 올려 주세요 (jpg · png · bmp)", type=["jpg", "jpeg", "png", "bmp"],
                          key="upload")
    photo_slot = st.empty()
    if up is None:
        photo_slot.info("사진을 올리면 이 자리에 판정과 해설이 나옵니다.")
    elif pipe is None:
        photo_slot.warning("판정 모델이 준비되지 않아 판정할 수 없습니다.")
    else:
        image = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            photo_slot.error("이미지로 읽지 못했습니다. 다른 파일로 시도해 주세요.")
        else:
            record = make_record(image, up.name)
            with photo_slot.container():
                left, right = st.columns([1, 1], gap="large")
                left.image(image, channels="BGR", width="stretch")
                key = render_verdict(right, record)
                render_profile(st, key)

# ---------------------------------------------------------------- 탭 2: 카메라·폴더
with tab_live:
    c1, c2, c3 = st.columns([1, 2, 1])
    options = detect_sources()
    source = c1.radio("입력", options, key="source", horizontal=True, disabled=ss.running)
    folder = c2.text_input("사진 폴더", args.folder, key="folder",
                           disabled=ss.running or source != sources.FOLDER)
    delay_ms = c3.slider("한 장 간격 (ms)", 100, 1000, 300, step=50, key="delay")
    if sources.WEBCAM not in options:
        st.caption("카메라를 찾지 못해 입력 목록에서 뺐습니다. 카메라가 있는 기기에서는 자동으로 나타납니다.")
        if st.button("카메라 다시 찾기", key="rescan", disabled=ss.running):
            detect_sources.clear()
            st.rerun()
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("시작", key="start", type="primary", on_click=on_start,
              disabled=ss.running or pipe is None, width="stretch")
    b2.button("정지", key="stop", on_click=on_stop, disabled=not ss.running, width="stretch")

    main_col, side_col = st.columns([3, 2], gap="large")
    image_slot = main_col.empty()
    verdict_slot = side_col.empty()
    summary_slot = side_col.empty()
    profile_slot = st.empty()


def show_summary() -> None:
    records = list(ss.records)
    with summary_slot.container():
        if not records:
            return
        counts, held, failed = Counter(), 0, 0
        for r in records:
            pred = r.get("pred")
            if pred is None:
                failed += 1
                continue
            _, _, state = profiles.decide(CLASS_PROFILES, pred.get("label"), pred.get("score"), review_thr)
            if state == "ok":
                counts[pred["label"]] += 1
            else:
                held += 1
        parts = [f"{profiles.headline(CLASS_PROFILES, c)} {n}장" for c, n in counts.most_common()]
        if held:
            parts.append(f"보류 {held}장")
        if failed:
            parts.append(f"실패 {failed}장")
        st.caption(f"지금까지 {len(records)}장 · " + " · ".join(parts))


def show_live(record: dict | None) -> None:
    if record is None:
        return
    key = render_verdict(verdict_slot, record)
    if key != ss.shown_label:        # 같은 양식이 이어지면 해설을 다시 그리지 않는다
        ss.shown_label = key
        render_profile(profile_slot, key)


# ---------------------------------------------------------------- 실행
frames = None
if ss.running and pipe is None:
    ss.running = False
if ss.running and source == sources.FOLDER:
    folder_path = Path(folder)
    if not folder_path.is_dir() or not sources.list_images(folder_path):
        st.error(f"사진이 있는 폴더가 아닙니다: {folder_path}")
        ss.running = False
    else:
        frames = sources.folder_frames(folder_path)
elif ss.running:
    frames = sources.webcam_frames()

if ss.running and frames is not None:
    try:
        for filename, image in frames:
            record = make_record(image, filename)
            ss.records.append(record)
            ss.last_image = image
            image_slot.image(image, channels="BGR", width="stretch")
            show_live(record)
            show_summary()
            time.sleep(delay_ms / 1000)
    finally:
        frames.close()
    ss.running = False
    st.rerun()

if ss.last_image is None:
    image_slot.info("입력을 고르고 시작을 누르면 여기에 사진과 판정이 나옵니다.")
else:
    image_slot.image(ss.last_image, channels="BGR", width="stretch")
    show_live(ss.records[-1] if ss.records else None)
show_summary()
