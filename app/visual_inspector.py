"""대화형 시각화 인스펙터 대시보드 (Streamlit 기반):
전처리, 누끼따기(배경 분리), 마스킹 진단, 모델 예측 확률을 실시간으로 확인하는 웹 UI.
실행 방법: streamlit run app/visual_inspector.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transforms import AspectRatioPadResize
from src.cv_utils.segmentation import extract_tumbler_mask, render_checkerboard, create_masking_variants
from src.predict import TumblerPredictor
from src.models.vit_classifier import CLASSES

RAW_DIR = PROJECT_ROOT.parent / "raw"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"

st.set_page_config(page_title="Tumbler Shape Inspector", layout="wide", page_icon="🥤")

st.title("🥤 텀블러 형태 분류 및 이미지 처리 시각 검증 대시보드")
st.markdown("전처리(종횡비 패딩), 누끼따기(배경 분리/실루엣), 마스킹(로고/배경 가림) 및 ViT 모델 예측을 실시간으로 직접 확인합니다.")

# ---------------------------------------------------------------- 사이드바
st.sidebar.header("⚙️ 설정 및 입력 선택")

# 체크포인트 선택
ckpts = list(CHECKPOINTS_DIR.glob("*.pt"))
if ckpts:
    ckpt_names = [p.name for p in sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)]
    selected_ckpt = st.sidebar.selectbox("모델 체크포인트", ckpt_names)
    ckpt_path = CHECKPOINTS_DIR / selected_ckpt
else:
    ckpt_path = None
    st.sidebar.warning("⚠️ 학습된 체크포인트가 없습니다. 모델 학습 후 추론이 활성화됩니다.")

input_mode = st.sidebar.radio("이미지 소스", ["데이터셋 탐색", "내 사진 업로드"])

target_image = None
image_title = ""

if input_mode == "내 사진 업로드":
    uploaded = st.sidebar.file_uploader("텀블러 사진 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        target_image = Image.open(uploaded)
        image_title = uploaded.name
else:
    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV)
        cls_choice = st.sidebar.selectbox("클래스 선택", CLASSES)
        sub_df = df[df["label"] == cls_choice].reset_index(drop=True)
        img_names = sub_df["filename"].tolist()
        sel_file = st.sidebar.selectbox("이미지 선택", img_names)
        sel_row = sub_df[sub_df["filename"] == sel_file].iloc[0]
        img_path = Path(sel_row["full_path"])
        if img_path.exists():
            target_image = Image.open(img_path)
            image_title = f"{cls_choice} / {sel_file} (Product Group: {sel_row['product_group']})"
    else:
        st.sidebar.info("데이터 검수를 먼저 실행하면 메타데이터 탐색이 가능합니다.")

# ---------------------------------------------------------------- 메인 화면
if target_image is not None:
    pil_rgb = ImageOps.exif_transpose(target_image).convert("RGB")
    w, h = pil_rgb.size
    st.subheader(f"📌 대상 이미지: {image_title} ({w} × {h})")

    # 1. 전처리 및 누끼따기 파이프라인 시각화
    st.markdown("### 1️⃣ 전처리 및 누끼따기(배경 분리) 단계별 시각화")
    col1, col2, col3, col4 = st.columns(4)

    # 1) 원본
    with col1:
        st.markdown("**① 원본 이미지 (RGB)**")
        st.image(pil_rgb, use_container_width=True)

    # 2) 224x224 종횡비 보존 패딩
    pad_tool = AspectRatioPadResize(224)
    padded_pil = pad_tool(pil_rgb)
    with col2:
        st.markdown("**② 종횡비 보존 패딩 (224×224)**")
        st.image(padded_pil, use_container_width=True, caption="왜곡 없이 중립색(회색) 패딩")

    # 3) 누끼따기 (RGBA on Checkerboard)
    with st.spinner("배경 분리(누끼) 추출 중..."):
        mask, rgba = extract_tumbler_mask(pil_rgb)
    checker = render_checkerboard(h, w)
    alpha = (rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
    nukki_visual = (rgba[:, :, :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)

    with col3:
        st.markdown("**③ 배경 분리 누끼 (Foreground)**")
        st.image(nukki_visual, use_container_width=True, caption="배경 제거된 텀블러 객체")

    # 4) 실루엣 마스크 & 윤곽선
    with col4:
        st.markdown("**④ 실루엣 마스크 (Silhouette)**")
        st.image(mask, use_container_width=True, caption="형태 판정용 이진 실루엣")

    st.divider()

    # 2. 모델 예측 결과
    if ckpt_path is not None and ckpt_path.is_file():
        st.markdown("### 2️⃣ 모델 형태 분류 및 신뢰도")
        predictor = TumblerPredictor(ckpt_path)
        pred_res = predictor.predict(pil_rgb)

        pred_col1, pred_col2 = st.columns([1, 2])
        with pred_col1:
            st.metric("예측 형태 (Predicted Shape)", pred_res["label"])
            st.metric("신뢰도 (Confidence Score)", f"{pred_res['score'] * 100:.2f}%")
            st.caption(f"추론 시간: {pred_res['infer_ms']} ms")

        with pred_col2:
            st.markdown("**클래스별 예측 확률 분포**")
            chart_df = pd.DataFrame({
                "Shape Class": list(pred_res["probs"].keys()),
                "Probability": list(pred_res["probs"].values()),
            })
            st.bar_chart(chart_df.set_index("Shape Class"))

        st.divider()

        # 3. 형상 의존성 마스킹 실시간 테스트
        st.markdown("### 3️⃣ 형상 의존성 마스킹 테스트 (색상/로고 vs 순수 형태)")
        st.markdown("모델이 로고나 특정 색상에만 반응하는지, 몸통 윤곽 형태를 보는지 검증합니다.")

        variants = create_masking_variants(pil_rgb, mask)
        m_col1, m_col2, m_col3 = st.columns(3)

        # 1) 흑백 변환
        with m_col1:
            st.markdown("**[테스트 A] 흑백 변환 (Grayscale)**")
            st.image(variants["grayscale"], use_container_width=True)
            res_gray = predictor.predict(variants["grayscale"])
            st.info(f"예측: **{res_gray['label']}** ({res_gray['score'] * 100:.1f}%)")

        # 2) 로고 마스킹
        with m_col2:
            st.markdown("**[테스트 B] 중심 로고 가림 (Logo Masked)**")
            st.image(variants["logo_masked"], use_container_width=True)
            res_logo = predictor.predict(variants["logo_masked"])
            st.info(f"예측: **{res_logo['label']}** ({res_logo['score'] * 100:.1f}%)")

        # 3) 배경 마스킹 (누끼만 유지)
        with m_col3:
            st.markdown("**[테스트 C] 배경 가림 (Nukki Only)**")
            st.image(variants["bg_masked"], use_container_width=True)
            res_bg = predictor.predict(variants["bg_masked"])
            st.info(f"예측: **{res_bg['label']}** ({res_bg['score'] * 100:.1f}%)")
else:
    st.info("👈 좌측 사이드바에서 이미지를 선택하거나 사진을 업로드해 주세요.")
