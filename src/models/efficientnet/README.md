# EfficientNet-B0 image classification

아래 명령은 **저장소 루트**에서 실행합니다.

`img/train`의 크롤링 이미지만 학습에 사용하고, 이를 클래스별로 train/holdout-test로
층화 분할합니다. `img/test`의 실촬영 이미지는 학습에 전혀 사용하지 않습니다.

현재 인식된 클래스는 `mug`, `straight`, `taper_smooth`, `taper_step`입니다.

최종 실험 결과와 Grad-CAM 분석은 [결과 보고서](../../../reports/efficientnet/TRAINING_EVALUATION_REPORT.md)에 정리되어 있습니다.
원본 이미지와 학습된 체크포인트는 이 브랜치에 포함하지 않습니다.

## 1. 설치

```powershell
python -m pip install -r requirements.txt
```

첫 실행에서는 torchvision이 ImageNet pretrained EfficientNet-B0 가중치를 내려받습니다.

## 2. 학습

```powershell
python src/models/efficientnet/train.py --epochs 15 --batch-size 32
```

기본 설정은 크롤링 이미지를 클래스별 80:20으로 나눕니다. 별도의 validation set과
test 성능 기반 early stopping은 사용하지 않으며, 고정 epoch의 마지막 모델을 저장합니다.
GPU가 있으면 자동으로 CUDA를 사용합니다. GPU 메모리가 부족하면 `--batch-size 16`으로
낮추면 됩니다.

주요 결과:

- `outputs/efficientnet_b0_final.pt`: 최종 모델과 재현 가능한 holdout 파일 목록
- `outputs/training_history.json`: 학습 loss/accuracy 기록

## 3. 두 테스트셋 평가와 Grad-CAM

```powershell
python src/models/efficientnet/evaluate.py --cam-per-class 4
```

평가 결과는 `outputs/evaluation`에 저장됩니다.

- `crawler_holdout/metrics.json`: train에서 떼어둔 크롤링 test 성능
- `real_photos/metrics.json`: 별도 실촬영 test 성능
- 각 폴더의 `predictions.csv`, `confusion_matrix.csv`, `confusion_matrix.png`
- `comparison.json`: 두 도메인의 accuracy/macro-F1 차이
- `gradcam/crawler_holdout`, `gradcam/real_photos`: 클래스별 Grad-CAM overlay

Grad-CAM에서 뜨거운 색 영역이 모델 예측에 크게 기여한 부분입니다. 물체 자체가 아닌
배경, 워터마크, 손, 촬영대 등에 집중한다면 크롤링 데이터의 편향을 학습한 신호입니다.

## 자주 쓰는 옵션

```powershell
# 크롤링 holdout 비율을 25%로 변경
python src/models/efficientnet/train.py --test-ratio 0.25 --epochs 20

# CPU를 명시적으로 사용
python src/models/efficientnet/train.py --device cpu --batch-size 16
python src/models/efficientnet/evaluate.py --device cpu
```
