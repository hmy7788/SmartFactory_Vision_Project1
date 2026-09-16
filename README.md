# 텀블러 4종 형태 분류 - ViT

텀블러 몸통의 윤곽과 비율을 읽어 네 형태를 분류하는 Vision Transformer 프로젝트입니다. 사전학습된 `DeiT-Small/16`을 미세조정하며, 핵심 지표는 네 클래스의 F1을 동등하게 평균한 **Macro F1**입니다.

## 분류 클래스

| 클래스 | 형태 기준 |
|---|---|
| `straight` | 몸통 아래 지름이 위 지름의 95% 이상이고 윤곽이 거의 수직 |
| `taper_smooth` | 꺾임 없이 몸통이 점진적으로 좁아짐 |
| `taper_step` | 몸통에 뚜렷한 단차 또는 꺾임이 한 번 이상 있음 |
| `mug` | 손잡이와 무관하게 높이가 지름의 1.5배 이하 |

## 구성

```text
src/
  data/              데이터 검사, 중복 제거, 제품 그룹 분할
  models/            DeiT-Small 분류기와 전처리
  cv_utils/          GrabCut 기반 실루엣 추출
  explainability/    ViT Grad-CAM
  train.py           선형 분류층 학습과 부분 미세조정
  predict.py         단일 이미지 추론과 시각화
  evaluate_*.py      웹, 실촬영, 통합 평가
data/
  metadata.csv       웹 학습·검증 데이터의 라벨, 그룹, 분할 정보
  real_test/         4종 12장 실촬영 파일럿과 메타데이터
checkpoints/
  exp_b_seed43_best.pt  최종 선택 가중치
reports/
  figures/           혼동행렬, 학습곡선, Grad-CAM, 전처리 시각화
  combined_test_*    통합 평가 결과
```

웹 학습 원본 이미지는 수집 출처와 용량 문제로 저장소에 포함하지 않습니다. `data/metadata.csv`의 `full_path`를 현재 보유한 `raw/` 데이터 경로에 맞춘 뒤 학습을 실행하세요.

## 설치와 실행

Python 3.10 이상과 CUDA가 가능한 PyTorch 환경을 사용했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 데이터 검사·메타데이터 생성
python -m src.data.inspect_and_clean
python -m src.data.split_dataset

# 실험 A: 분류층만 학습 / 실험 B: 마지막 Transformer 블록 미세조정
python -m src.train --mode compare

# 최종 가중치로 단일 이미지 예측
python -m src.predict --image path\to\tumbler.jpg --checkpoint checkpoints\exp_b_seed43_best.pt
```

`--mode compare`는 seed 42에서 선형 분류층 학습과 부분 미세조정을 비교한 뒤, 더 좋은 방식을 seed 43·44에서 반복합니다. 학습과 모델 선택에는 검증 Macro F1만 사용합니다.

## 통합 데이터 평가 결과

최종 선택 모델은 `exp_b_seed43`입니다. 웹 검증 89장과 학습에 사용하지 않은 리비전 실촬영 126장을 합친 215장 통합 평가에서 아래 결과를 얻었습니다.

| 평가셋 | 사진 수 | Macro F1 | Accuracy | 비고 |
|---|---:|---:|---:|---|
| 웹 검증 | 89 | 0.9678 | 0.9663 | 제품 그룹을 분리한 웹 이미지 |
| 리비전 실촬영 | 126 | 0.8161 | 0.8095 | 팀원 휴대폰 촬영 이미지 |
| **통합 평가** | **215** | **0.8784** | **0.8744** | 웹 검증 + 실촬영 |

| 클래스 | 사진 수 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `straight` | 64 | 0.8525 | 0.8125 | 0.8320 |
| `taper_smooth` | 52 | 0.7869 | 0.9231 | 0.8496 |
| `taper_step` | 55 | 0.9796 | 0.8727 | 0.9231 |
| `mug` | 44 | 0.9091 | 0.9091 | 0.9091 |

웹 이미지보다 실촬영에서 성능이 낮아졌으며, 특히 `straight`와 `taper_smooth`가 서로 혼동되는 경향이 확인됐습니다. 자세한 혼동행렬과 오분류 목록은 [통합 평가 보고서](reports/combined_test_report.md), 데이터 분할은 [분할 요약](reports/data_split_summary.json), 시각 자료는 [reports/figures](reports/figures)에서 확인할 수 있습니다.

## 재현성

- 클래스 순서는 `straight`, `taper_smooth`, `taper_step`, `mug`로 고정합니다.
- Macro F1은 고정 네 클래스를 대상으로 `zero_division=0`으로 계산합니다.
- 종횡비를 유지한 224×224 패딩 전처리를 학습·추론에 공통 적용합니다.
- 동일 제품·파생 이미지는 `product_group` 단위로 학습·검증 분할합니다.

## 부가 도구

- [시각 검수 결과](VISUAL_INSPECTION_RESULTS.md): 전처리, Grad-CAM, 속도, 형태 의존성 진단
- [네이버 쇼핑 수집기](docs/naver_tumbler.md): 공식 API 기반 후보 수집과 검수·내보내기
- [SSG 수집기](docs/ssg_crawler.md): 공개 검색 카드 수집과 재개 처리
