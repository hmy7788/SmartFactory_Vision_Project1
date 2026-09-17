# convnext — 팀 ConvNeXt-Tiny 분류기

- `inference.py`: 학습한 팀원이 준 추론 모듈 (원본 그대로). GradCAM 히트맵 생성 기능 포함.
- 가중치: `checkpoints/model/best_convnext_v5.pth` (111MB, git 제외 — 팀원한테 직접 받는다)
- 대시보드는 이 파일을 직접 쓰지 않고 `src/pipeline.py`가 `checkpoints/model/meta.json`을 읽어 같은 방식으로 복원한다.
  `meta.json`의 값은 이 파일의 TRANSFORM·CLASS_NAMES·_load_model에서 뽑은 것 — 여기가 바뀌면 meta.json도 맞춘다.
