# streamlit2 — 텀블러 형태 판정 데모

팀원 제작 UI + ConvNeXt-Tiny v5 모델 연동 버전.

## 실행 방법

```bash
# 저장소 루트(streamlit2/)에서
pip install -r requirements.txt
streamlit run app/demo.py

# 또는 윈도우에서
run_demo.bat
```

## 모델 파일 넣는 법

`checkpoints/model/` 폴더에 가중치 파일을 넣으면 됩니다.

```
checkpoints/model/
├── meta.json              ← 이미 설정 완료 (ConvNeXt-Tiny v5)
└── best_convnext_v5.pth   ← 학습한 사람한테 받아서 여기 복사
```

meta.json은 이미 ConvNeXt-Tiny에 맞게 설정되어 있습니다.
가중치 파일만 넣으면 바로 실행됩니다.

## 파일 구조

```
streamlit2/
├── app/
│   └── demo.py                       # Streamlit 화면 (메인)
├── src/
│   ├── pipeline.py                   # 모델 로드 & 추론
│   ├── sources.py                    # 이미지 입력 (업로드/웹캠/폴더)
│   └── explanation/
│       ├── profiles.py               # 해설 로더 & 판정 규칙
│       └── class_profiles.yaml       # 클래스별 해설 텍스트
├── checkpoints/model/
│   ├── meta.json                     # 모델 설정 (ConvNeXt-Tiny v5)
│   ├── meta.example.json             # 설정 예시
│   └── best_convnext_v5.pth          # 가중치 (별도 전달)
├── requirements.txt
├── run_demo.bat                      # 윈도우 실행 스크립트
└── README.md
```

## ConvNeXt vs 원본(MobileNetV3) 변경 사항

| 항목 | 원본 | 이 버전 |
|------|------|---------|
| `arch` | mobilenet_v3_small | convnext_tiny |
| `resize` | letterbox | stretch |
| `weights` | best.pt | best_convnext_v5.pth |
| `demo.py` width 인자 | `width="stretch"` (에러) | `use_container_width=True` |
