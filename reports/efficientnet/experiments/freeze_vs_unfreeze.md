# EfficientNet-B0: Freeze vs. Unfreeze 비교

## 결론

사전학습 특징 추출부를 15 epoch 동안 계속 고정한 `freeze/head-only` 방식은, 같은 학습률에서 모델 전체를 학습한 `unfreeze/full fine-tuning` 방식보다 크롤링 holdout과 실촬영 test 성능이 낮았다. 특히 [기준선](../TRAINING_EVALUATION_REPORT.md) 학습률 `3e-4`에서 실촬영 정확도 차이는 14.29%p였다.

| 학습률 | 학습 방식 | 학습 가능 파라미터 | 최종 train 정확도 | 크롤링 holdout 정확도 / Macro-F1 | 실촬영 정확도 / Macro-F1 |
|---:|---|---:|---:|---:|---:|
| `1e-4` | Freeze: head-only | 5,124 / 4,012,672 | 82.85% | 86.82% (112/129) / 86.89% | 69.05% (87/126) / 67.64% |
| `1e-4` | Unfreeze: 전체 학습 | 4,012,672 / 4,012,672 | 97.86% | 96.90% (125/129) / 97.02% | 79.37% (100/126) / 79.34% |
| `3e-4` | Freeze: head-only | 5,124 / 4,012,672 | 91.23% | 94.57% (122/129) / 94.66% | 73.81% (93/126) / 73.38% |
| **`3e-4`** | **Unfreeze: 전체 학습** | **4,012,672 / 4,012,672** | **99.22%** | **97.67% (126/129) / 97.73%** | **88.10% (111/126) / 87.50%** |

## 실험 방법

- 공통 조건: ImageNet 사전학습 EfficientNet-B0, 크롤링 train 513장·holdout 129장, 실촬영 test 126장, seed 42, 15 epoch, batch 32, AdamW(weight decay `1e-4`), CosineAnnealingLR, 기존 약한 증강과 CenterCrop 평가. Validation·test 기반 모델 선택 없이 마지막 epoch 모델 사용.
- Freeze: `features` 파라미터의 gradient를 비활성화하고, 학습 중 특징 추출부를 eval 모드로 유지해 BatchNorm running statistics도 고정. 분류기 파라미터 5,124개만 optimizer에 전달.
- Unfreeze: 사전학습 가중치에서 시작해 특징 추출부와 분류기를 모두 업데이트. `1e-4` 및 `3e-4` 전체 학습 결과는 [학습률 벤치마크](learning_rate_benchmark.md)와 동일.
- 두 freeze 체크포인트에서 특징 추출부 가중치·BatchNorm 통계가 ImageNet 사전학습 상태와 동일함을 확인. 같은 학습률의 비교 모델 간 train/holdout 실제 파일 목록·순서, 클래스 인덱스, seed, 증강 및 평가 전처리도 일치함을 검증.

## 실촬영 클래스별 F1

| 클래스 | Support | Freeze `1e-4` | Unfreeze `1e-4` | Freeze `3e-4` | Unfreeze `3e-4` |
|---|---:|---:|---:|---:|---:|
| `mug` | 22 | 71.11% | 80.95% | 78.05% | 82.05% |
| `straight` | 39 | 72.90% | 81.93% | 77.23% | 88.61% |
| `taper_smooth` | 32 | 51.06% | 68.75% | 67.86% | 82.35% |
| `taper_step` | 33 | 75.47% | 85.71% | 70.37% | 96.97% |

두 freeze 모델 모두 실촬영 `straight` 39장을 전부 맞혔지만 다른 클래스를 `straight`로 과도하게 예측했다. Freeze `3e-4`에서는 실제 `taper_smooth` 11장과 `taper_step` 8장을 `straight`로 예측했다. Freeze `1e-4`에서는 각각 16장과 7장이었다. `1e-4`에서 전체 미세조정은 freeze보다 실촬영 정확도 10.32%p, `3e-4`에서는 14.29%p 높았다.

## 해석상의 제한

이 비교의 freeze는 **15 epoch 내내 고정**하는 방법이다. 팀원들이 사용한 'head-only 워밍업 후 unfreeze' 2단계 학습은 이 보고서에서 실험하지 않았으므로 같은 결론을 적용할 수 없다. 모든 결과는 seed 42의 단일 학습이며, 실촬영 test의 반복 사용에 따른 선택 편향을 고려해야 한다. 실험용 코드·체크포인트·그래프·이미지별 예측은 이 브랜치에 포함하지 않았고, 기준선 학습 코드는 변경하지 않았다.
