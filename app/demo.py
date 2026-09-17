"""텀블러 형태 → 건축 양식 판정 (보는 사람용 화면).

저장소 루트에서:
    streamlit run app/demo.py
    streamlit run app/demo.py -- --checkpoint checkpoints/model

구성
  - 모델은 checkpoints/<폴더>/meta.json + 가중치 하나. 고르는 화면이 없다.
  - 이력 저장 없음. 지표·로그 없음 — 보는 사람이 알 필요 없는 것.
  - 사진 한 장 올리는 흐름이 첫 화면. 실시간(카메라)이 두 번째 탭.
  - 판정 아래에 그 양식의 해설(src/explanation/class_profiles.yaml).

판정 규칙(src/explanation/profiles.decide)은 모델 출력과 다른 층이다 — 확신도에 기준을 걸어
사후에 정한다. 기준 미만이면 '판정 보류'로 두고 어떤 양식의 해설도 내지 않는다(화면이 자기를
반박하지 않게).
"""
import argparse
import html as html_lib
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

from src import gradcam_reason  # noqa: E402
from src import live as live_stream  # noqa: E402
from src import sources  # noqa: E402
from src.explanation import profiles  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

REVIEW_THR = 0.70     # 판정 확신 기준 기본값
MAX_RECORDS = 5000
UNSET = "__unset__"   # '이번 실행에서 아직 해설을 안 그렸다' 표시. None은 '보류라서 해설 없음'과 겹친다

TITLE = "텀블러 형태로 읽는 건축 양식"
SUBTITLE = ("텀블러의 실루엣을 건축물의 형태로 보고, 어떤 양식 계열인지 판정한 뒤 그 양식의 배경을 설명합니다. "
            "이 프로젝트에서 텀블러는 건축 양식 분류의 대리 도메인입니다.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model"))
    return p.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner="카메라를 찾는 중입니다…")
def find_camera() -> int | None:
    """이 컴퓨터에서 열리는 첫 카메라 번호. 노트북엔 보통 하나뿐이라 고를 일이 없다.

    번호를 손으로 고르게 두지 않는 이유: 노트북에 적외선(Windows Hello) 카메라가 같이 달려 있으면
    번호가 0이 아닐 수 있는데, 그건 쓰는 사람이 알 바가 아니다. 열리는 것을 찾아서 쓴다.
    """
    return sources.first_camera()


@st.cache_resource(show_spinner="판정 모델 불러오는 중")
def get_pipeline(checkpoint_dir: str, meta_mtime: float) -> Pipeline:
    return Pipeline(checkpoint_dir)


def on_start() -> None:
    st.session_state.running = True
    st.session_state.records.clear()
    st.session_state.shown_label = UNSET
    st.session_state.captured = None


def on_stop() -> None:
    st.session_state.running = False


def on_snap() -> None:
    """캡처 = 영상을 멈추고 그 순간의 한 장을 붙잡는다.

    멈추는 이유: 화면이 돌아가는 동안엔 어떤 버튼도 누를 수 없다(파이썬이 프레임 루프에 붙잡혀 있다).
    버튼을 누르는 순간 루프가 끊기므로, 어차피 멈춘다. 그러면 그 한 장을 차분히 볼 수 있게 한다.
    """
    st.session_state.snap_request = True
    st.session_state.running = False


# ---------------------------------------------------------------- 시작
args = parse_args()
st.set_page_config(page_title=TITLE, layout="wide")
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("records", deque(maxlen=MAX_RECORDS))
ss.setdefault("last_image", None)
ss.setdefault("shown_label", UNSET)
ss.setdefault("cam_index", 0)
ss.setdefault("judge", None)
ss.setdefault("captured", None)      # (이미지, 판정) — 실시간 화면에서 캡처한 한 장
ss.setdefault("snap_request", False)
ss.shown_label = UNSET   # 본 실행마다 리셋: 화면이 새로 그려졌으니 해설도 다시 그려야 한다

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
    judge_interval = st.slider("실시간 판정 간격 (초)", 0.5, 4.0, live_stream.MIN_INTERVAL, step=0.5,
                               key="interval",
                               help="실시간 탭에서 몇 초에 한 번 판정할지. 짧게 두면 CPU를 계속 쓰느라 "
                                    "영상이 끊기거나 카메라 연결이 죽습니다. 영상이 불안정하면 늘려 주세요.")
    show_gradcam = st.checkbox("판정 근거 히트맵 보기", value=True, key="gradcam",
                               help="모델이 사진의 어디를 보고 판단했는지 색으로 보여줍니다(GradCAM). "
                                    "사진 한 장과 캡처한 장면에만 그립니다 — 실시간으로 매 프레임 그리기엔 무겁습니다.")
    with st.expander("이 화면 읽는 법"):
        st.markdown(
            "- 🟢 **판정 완료** — 확신도가 기준 이상. 양식과 해설이 나옵니다.\n"
            "- 🟡 **판정 보류** — 확신도가 기준 미만. 확실하지 않을 땐 답하지 않고 후보만 보여줍니다.\n"
            "- 🔴 **판정 실패** — 사진을 처리하지 못했습니다.\n"
            "- 확신도는 '모델이 헷갈리는 정도'입니다. 물체가 너무 작거나 배경이 복잡하면 "
            "확신하면서 틀릴 수도 있으니, 텀블러가 화면에 크게 나오게 찍어 주세요.")
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

tab_photo, tab_live = st.tabs(["사진으로 판정", "실시간 판정"])


BAR_ACCENT = "#FF4B4B"   # 기준을 넘긴 1등에만. .streamlit/config.toml의 primaryColor와 같은 값
BAR_MUTED = "#898781"    # 그 외 전부 — 맥락이지 주인공이 아니다


def probs_chart(probs: dict | None, top: str | None, threshold: float) -> str:
    """클래스별 확신도 막대. 등수 순 · 1등만 색 · 기준선 하나.

    읽는 사람이 할 일은 둘이다 — 어느 형태로 봤나, 기준을 넘었나. 그래서
      - 확신도 순으로 세운다: 1·2등 차이가 길이로 바로 보인다(전엔 클래스 고정 순서라 눈으로 찾아야 했다)
      - **기준을 넘긴 1등에만** 색을 준다: 색이 곧 '판정했다'는 뜻이다. 못 넘으면 전부 회색 —
        화면에 색이 없으면 아무것도 확정하지 않았다는 말이라, 색이 데이터보다 앞서가지 않는다
      - 세로선으로 기준을 긋는다: '넘었나'가 판정의 전부인데 숫자를 비교해야만 알 수 있었다
    막대마다 다른 색을 주지는 않는다 — 길이가 이미 말한 것을 색으로 또 말하는 꼴이라서.
    """
    if not probs:
        return ""
    thr = max(0.0, min(100.0, float(threshold) * 100))
    rows = sorted(probs.items(), key=lambda kv: -float(kv[1]))
    out = [f'<div style="font-size:.76rem;opacity:.55;text-align:right;margin:0 0 3px 0;">'
           f'세로선 = 판정 기준 {thr:.0f}%</div>']
    for name, value in rows:
        pct = max(0.0, min(100.0, float(value) * 100))
        lead = name == top
        confirmed = lead and pct >= thr      # 기준을 넘긴 1등만 색을 받는다
        label = html_lib.escape(profiles.headline(CLASS_PROFILES, name))
        out.append(
            f'<div style="margin:0 0 9px 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;'
            f'font-size:.86rem;opacity:{"1" if lead else ".6"};font-weight:{600 if lead else 400};">'
            f'<span>{"▸ " if lead else ""}{label}</span>'
            f'<span style="font-variant-numeric:tabular-nums;">{pct:.1f}%</span></div>'
            f'<div style="position:relative;height:10px;border-radius:5px;'
            f'background:rgba(128,128,128,.16);margin-top:4px;">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:{pct:.2f}%;'
            f'background:{BAR_ACCENT if confirmed else BAR_MUTED};border-radius:5px;"></div>'
            f'<div style="position:absolute;left:{thr:.2f}%;top:-3px;bottom:-3px;width:2px;'
            f'background:rgba(128,128,128,.7);"></div>'
            f'</div></div>')
    return "".join(out)


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
        chart = probs_chart(pred.get("probs"), label, review_thr)
        if chart:
            st.markdown(chart, unsafe_allow_html=True)
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


def render_gradcam(container, image: np.ndarray, record: dict) -> None:
    """판정 근거 히트맵. 사진 한 장·캡처 한 장에만 그린다 (역전파까지 돌아 실시간엔 무겁다)."""
    if not show_gradcam or pipe is None:
        return
    pred = record.get("pred")
    if pred is None:
        return
    with container.container():
        out, err = pipe.safe_explain(image, pred.get("label"))
        if err:
            st.caption(f"근거 히트맵을 그리지 못했습니다 — {err}")
            return
        st.image(out["overlay"], caption="판정 근거 — 빨간 곳이 모델이 보고 판단한 부분", width="stretch")
        try:
            why = gradcam_reason.explain(out["cam"], out["label"],
                                         float(pred.get("score") or 0.0), review_thr)
        except Exception as e:
            st.caption(f"근거 문장을 만들지 못했습니다 — {type(e).__name__}: {e}")
            return
        st.markdown(f"**주목한 곳** — {why['focus_region']}  ·  집중도 {why['attention_pct']}%")
        for line in why["reason_lines"]:
            st.markdown(line)
        st.caption(why["confidence_str"])
        st.caption("이 문장은 '모델이 어디를 봤나'를 옮긴 것이지 정답의 증거가 아닙니다. "
                   "텀블러가 아니라 배경이나 손이 빨갛다면 맞혔더라도 우연일 수 있습니다.")


def render_capture(image: np.ndarray, record: dict, title: str = "📸 캡처 판정") -> None:
    """캡처한 한 장: 사진 · 근거 히트맵 · 판정 · 해설."""
    st.markdown(f"#### {title}")
    left, right = st.columns([1, 1], gap="large")
    left.image(image, channels="BGR", width="stretch")
    key = render_verdict(right, record)
    render_gradcam(left, image, record)
    render_profile(st, key)


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
                render_gradcam(left, image, record)
                render_profile(st, key)

# ---------------------------------------------------------------- 탭 2: 실시간
# 카메라는 이 컴퓨터 것 하나다. cv2로 직접 열고 프레임을 읽어 화면에 그린다.
#
# 브라우저 카메라(webrtc)는 뺐다 — 팀원 기기에서 찍게 하려던 것인데, 붙이는 데마다 환경을 타서
# (연결 지연, 몇 초 뒤 끊김, 터널 뒤에서 컴포넌트 로드 실패) 시연에 쓸 만큼 안정적이지 않았다.
# 코드는 src/live.py에 그대로 남아 있다 — 되살리려면 그 파일과 이 자리의 분기를 다시 붙이면 된다.

with tab_live:
    if pipe is None:
        st.warning("판정 모델이 준비되지 않아 실시간 판정을 할 수 없습니다.")
    else:
        st.caption("이 컴퓨터에 연결된 카메라로 판정합니다. 영상은 카메라 속도대로 흐르고, "
                   f"판정은 {judge_interval:.1f}초에 한 번씩 갱신됩니다.")
        ss.cam_index = find_camera()
        if ss.cam_index is None:
            st.warning("카메라를 찾지 못했습니다. 다른 앱(줌·팀즈 등)이 쓰고 있으면 먼저 닫아 주세요. "
                       "윈도우 설정에서 '데스크톱 앱이 카메라에 액세스하도록 허용'도 확인해 주세요.")
            if st.button("카메라 다시 찾기", key="rescan"):
                find_camera.clear()
                st.rerun()
        b1, b2, b3, _ = st.columns([1, 1, 1.4, 2.6])
        b1.button("시작", key="start", type="primary", on_click=on_start,
                  disabled=ss.running or ss.cam_index is None, width="stretch")
        b2.button("정지", key="stop", on_click=on_stop, disabled=not ss.running, width="stretch")
        b3.button("📸 캡처해서 판정", key="snaplocal", on_click=on_snap,
                  disabled=not ss.running, width="stretch",
                  help="영상을 멈추고 그 순간의 한 장을 판정합니다 (근거 히트맵 포함).")
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


# ---------------------------------------------------------------- 실행: 이 컴퓨터 카메라
if ss.running and pipe is not None:
    # 영상과 판정을 떼어놓는다. 프레임마다 판정하면 한 장에 0.3초씩 걸려 영상이 뚝뚝 끊긴다.
    # 화면은 들어오는 대로 갱신하고, 판정은 judge_interval에 한 번만 — 그 사이엔 직전 판정을 둔다.
    frames = sources.webcam_frames(ss.cam_index)
    last_judge = last_show = 0.0
    try:
        for filename, image in frames:
            now = time.perf_counter()
            if now - last_show >= 0.1:        # 화면 갱신은 10fps면 충분하다 (그림을 매번 보내는 비용)
                last_show = now
                ss.last_image = image
                image_slot.image(image, channels="BGR", width="stretch")
            if now - last_judge >= judge_interval:
                last_judge = now
                record = make_record(image, filename)
                ss.records.append(record)
                show_live(record)
                show_summary()
    finally:
        frames.close()
    ss.running = False
    st.rerun()

if pipe is not None:
    if ss.snap_request:
        # 캡처 버튼은 영상 루프를 끊고 여기로 온다. 마지막으로 화면에 나온 그 장면을 붙잡는다.
        ss.snap_request = False
        if ss.last_image is None:
            ss.captured = None
        else:
            frozen = ss.last_image.copy()
            ss.captured = (frozen, make_record(frozen, "capture"))
    if ss.last_image is None:
        image_slot.info("시작을 누르면 여기에 화면과 판정이 나옵니다.")
    else:
        image_slot.image(ss.last_image, channels="BGR", width="stretch")
        show_live(ss.records[-1] if ss.records else None)
    show_summary()
    with tab_live:      # 탭 안에 그린다 — 밖에 두면 '사진으로 판정' 탭에서도 보인다
        if ss.captured is not None and not ss.running:
            render_capture(*ss.captured)
