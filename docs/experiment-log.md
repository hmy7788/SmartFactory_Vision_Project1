# 실험 결과 로그

4갈래 트랙(룰베이스 / Mask R-CNN / ResNet / EfficientNet)의 실험 결과를 트랙별로 기록한다. **검증 대상은 시스템**이므로, 정확도 숫자만이 아니라 구현 난이도·추론 속도·실패 패턴도 함께 남긴다.

새 실험을 기록할 때는 각 트랙 표에 행을 추가한다. 형식:

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| YYYY-MM-DD | 이름 | 예: ResNet-18 baseline, backbone freeze | 0.00 | 0.00 | 혼동 클래스, 특이사항 등 |

---

## 룰베이스 (`src/rule_based/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-14 | Claude | 최초 구현: get_mask/compute_width_profile/detect_handle/classify_shape + 합성 마스크 10종 pytest | N/A (실데이터 없음) | N/A | 합성 마스크 4종(+손잡이 변형) 전부 올바른 라벨로 분류(10/10 테스트 통과). 실사진 확보 전까지는 정확도 숫자 의미 없음 — get_mask의 세그멘테이션 견고성이 다음 병목 |
| 2026-09-14 | Claude | 실사진 4장(클래스당 1장) 수동 테스트 — 연속테이퍼형/직선원통형/단차테이퍼형/머그형 각 1장 | 2/4 정확(50%, n=4로 통계적 의미 없음) | - | **연속테이퍼형·직선원통형**: 단색 배경+고대비 → 마스크 깨끗, 정확히 분류. **단차테이퍼형**: 그림자가 마스크에 붙고 아래쪽 단차 부분이 통째로 누락됐는데도 라벨은 우연히 맞음(height/diameter=1.51, bottom/top=0.92 — 둘 다 임계값 바로 옆이라 불안정). **머그형**: 나무 배경과 코르크 받침 색이 비슷해 마스크가 중간에서 잘리고 손잡이도 누락 → straight로 오분류. get_mask()의 배경분리 한계가 예상대로 실제 데이터에서 터짐 — GrabCut 등 개선 필요 (다음 실험에서 처리) |
| 2026-09-14 | Claude | get_mask()에 GrabCut 정제 추가(Otsu 결과를 시드로 사용) 후 같은 4장 재테스트 | 2/4 정확(동일) | - | 단차테이퍼형의 위험한 경계값이 완화됨(h/d 1.51→1.58, b/t 0.92→0.89, 임계값에서 더 멀어짐) + 연속테이퍼형 마스크 디테일 향상. **머그형(코르크=나무 바닥 색 충돌)과 단차테이퍼형(그림자 융착)은 GrabCut으로도 해결 안 됨** — 색상 기반 알고리즘의 근본 한계, 촬영 조건(배경 대비·그림자 없는 조명) 문제에 가까움. 이런 어려운 케이스는 결국 Mask R-CNN(dl1) 트랙이 처리하도록 설계된 부분 |
| 2026-09-15 | Claude | `data/raw2`(수집·미검증 상태)에서 클래스당 무작위 10장(n=40, seed=42) 샘플링 후 `classify_shape()` 일괄 실행 — `notebooks/rule_based_raw2_sample.ipynb` | 폴더 라벨 기준 일치율: straight 100%(10/10), mug 100%(10/10), taper_smooth 60%(6/10), **taper_step 30%(3/10)**, 전체 72.5%(29/40) | - | **taper_step 6/10이 mug로 오분류** — 원인 확정: `get_mask()`가 몸통을 놓치고 금속 뚜껑/테두리만 마스크로 잡는 경우가 반복됨 → 높이가 과소측정되어 height/diameter≤1.5(mug 조건)를 충족해버림. 그림자/색충돌 케이스(9/14 실험)와 별개로, **taper_step 특유의 "몸통 놓침" 실패 모드**가 구조적으로 반복됨을 n=40에서 확인. 마스크가 몸통까지 제대로 잡힌 3장은 전부 정확히 분류됨 — 로직 자체는 문제없고 세그멘테이션이 병목. taper_smooth의 오분류(mug 2, taper_step 2)는 원인 미분석. **주의**: raw2 라벨은 미검수 상태라 일치율에 "엉뚱한 이미지가 섞여서 생긴 불일치"도 일부 포함될 수 있음(둘을 분리 못 함) |
| 2026-09-15 | Claude | `get_mask()`에 Canny 엣지 기반 마스크(`_rough_mask_canny`)를 Otsu와 OR로 합쳐 GrabCut 시드로 사용하도록 개선 후 동일 40장 재실행 | straight 90%(9/10), taper_smooth 50%(5/10), **taper_step 60%(6/10)**, mug 100%(10/10), 전체 **75.0%(30/40)** | - | 원인 진단: 실패 6장 중 3장(101/65/83.jpg)이 흰색 텀블러+흰색 배경이라 Otsu 전경 비율이 1.3~5.6%까지 떨어짐(명도 대비 자체가 없음) — Canny는 명도 절대값이 아니라 옅은 경계선을 보므로 이 케이스에 강함. **taper_step 30%→60%로 2배 개선, 전체도 순개선(72.5%→75%)**. 단, straight·taper_smooth에서 각 1건씩 새 오분류 발생(Canny가 배경 텍스처를 같이 잡는 부작용으로 추정, 미분석) — trade-off 있는 개선. 여전히 실패: 70.jpg(클로즈업 사진, 데이터 문제이지 알고리즘 문제 아님), 63.jpg(복잡한 야외 배경) |
| 2026-09-15 | Claude | `classify_shape()` 임계값을 실측(data/raw2 445장, Mask R-CNN 마스크 기준)으로 재조정 — `notebooks/threshold_calibration.ipynb`. 각 갈림길을 이진분류로 놓고 정확도 최대화 지점 탐색: MUG_HEIGHT_TO_DIAMETER_MAX 1.5→1.5(유지), STRAIGHT_BOTTOM_TOP_RATIO_MIN 0.95→**0.92**, STEP_JUMP_RATIO_THRESHOLD 0.4→**0.29** | Mask R-CNN 마스크 기준(445장): 84.5%→**88.0%**. 룰베이스 마스크 기준(445장): 66.3%→65.8%(거의 변화 없음) | - | Mask R-CNN 파이프라인에서 taper_step 75.7%→88.8%로 크게 개선(전체 3.5%p↑), taper_smooth는 90.0%→83.0%로 소폭 하락(trade-off). 룰베이스는 Mask R-CNN 마스크로 캘리브레이션한 값이라 전이가 완벽하진 않지만(-0.5%p) 손해가 거의 없어 공통 채택. **부수 효과**: 합성 테스트 `test_taper_smooth_with_handle_stays_taper_smooth`가 새 임계값에 걸려 실패 → 원인은 임계값이 아니라 테스트용 손잡이 돌기가 폭 구간 경계에 걸쳐 median smoothing으로 안 지워지는 인공적 단차를 만든 것으로 확인, `synthetic.py`에서 손잡이 위치를 구간 안쪽으로 조정해 해결 (`docs/troubleshooting.md` 참고). `classify_shape()`가 `step_ratio`를 항상 반환하도록 리팩터링(이전엔 taper 분기에서만 계산됨) |

## Mask R-CNN (`src/deep_learning/dl1_maskrcnn/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `segment.py` 최초 구현(COCO 사전학습 `maskrcnn_resnet50_fpn_v2`, 제로샷·fine-tuning 없음) + `rule_based_raw2_sample.ipynb`와 동일한 40장(seed=42)에 룰베이스와 나란히 비교 — `notebooks/maskrcnn_vs_rule_based.ipynb` | - (raw2, 미검증 라벨 기준 참고용) | - | **전체 75%(룰베이스) → 85%(Mask R-CNN)**. 클래스별: straight 90→100%, taper_smooth 50→70%, **taper_step 60→80%**(목표했던 개선), mug 100→90%(소폭 회귀). 추론 속도는 GPU(RTX 4050)에서 평균 150ms/장으로 룰베이스(CPU, 503ms/장, GrabCut이 병목)보다도 빠름. cup/bottle/vase/wine glass/bowl 중 점수 높은 COCO 검출을 사용 — 텀블러가 COCO 클래스에 없어서 근접 카테고리로 대체 |
| 2026-09-15 | Claude | 룰베이스 마스크를 pseudo-label 삼아 fine-tuning 시도 (배경/텀블러 2클래스, 평가용 40장은 학습에서 제외, `classify_shape()`가 폴더 라벨과 맞은 것만 채택) — `notebooks/maskrcnn_finetune_rulebase_pseudolabels.ipynb`. 학습 266장(3 epoch, 1197s, loss 0.376→0.140), GPU RTX 4050 | 동일 40장(held-out) 재평가: 78%(31/40) | - | **제로샷(85%)보다 오히려 나빠짐(78%) — 우려했던 리스크가 실제로 발생.** straight 100→80%, taper_smooth 70→60%, taper_step 80→70% 전부 회귀, mug만 90→100%(가장 많이 채택된 클래스라 치우침 발생). 원인: (1) 클래스별 pseudo-label 채택 수가 심하게 불균형(mug 88 / straight 81 / taper_smooth 64 / **taper_step 33** — 룰베이스가 약한 클래스일수록 채택 수도 적어져서 학습 데이터가 mug 쪽으로 편향), (2) 룰베이스 마스크의 거친 경계가 COCO 학습으로 얻은 정밀한 마스크 품질을 깎아먹은 것으로 추정. **결론: 이 체크포인트는 폐기, dl1 트랙은 제로샷 버전(`segment.py`의 기본 `load_model()`)을 계속 사용.** 체크포인트는 `checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth`에 남겨두되 실사용 안 함(참고 기록용) |

## ResNet-18/50 (`src/deep_learning/dl2_resnet/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `train.py` 최초 구현 — ResNet-18 ImageNet 사전학습, 백본 freeze(FC layer만 학습), `data/preprocess`(642장, 강한 신호 제외) 층화 80/20 분할, 증강에 `RandomPerspective` 포함(촬영 각도 왜곡을 직접 겨냥), 20 epoch, GPU RTX 4050(145s) | **91.3%**(best epoch 12) | **58.9%**(33/56, data/test 실촬영) | **압도적 개선**: 같은 Test 세트에서 룰베이스 18%, Mask R-CNN 39% 대비 두 배 가까이 높음 — 기하학적 규칙(`classify_shape`) 대신 학습된 시각 패턴을 쓰는 게 촬영 각도 왜곡에 훨씬 강하다는 가설이 실측으로 확인됨. Val→Test 하락폭은 32.4%p로 여전히 크지만(진짜 domain shift), 절대 정확도 자체가 크게 앞섬. Val 클래스별 F1: straight 0.928, taper_smooth 0.903, taper_step 0.857, mug 0.955 — 고르게 좋음. **Test에서는 mug만 1/6(17%)로 급락**(4장이 taper_step으로 오분류) — 표본이 6장뿐이라 노이즈 가능성 있음, 확대 검증 필요. 그래프: `reports/figures/resnet18_{training_curves,val_confusion_matrix,test_confusion_matrix}.png`, 체크포인트: `checkpoints/resnet18_shape.pth` |

## EfficientNet-B0 (`src/deep_learning/efficientnet/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| _(아직 실험 없음)_ | | | | | |

---

## 4갈래 종합 비교

`notebooks/03_model_comparison.ipynb`에서 정기적으로 산출되는 비교 결과를 요약한다. Train(웹 이미지)→Test(실사용 이미지) 간 domain shift로 인한 정확도 하락은 의도된 평가이니, 하락 폭 자체를 비교 지표로 다룬다.

| 날짜 | 트랙 | Val→Test 정확도 하락폭 | 주요 혼동 클래스 쌍 | 비고 |
|---|---|---|---|---|
| 2026-09-15 | 룰베이스 | raw2(≈val 성격) 65.8% → **Test 18%** (47.7%p 하락) | straight/taper_smooth/taper_step 전부 → mug로 대량 오분류 | `notebooks/test_set_evaluation.ipynb`. data/test는 직접 촬영한 진짜 Test 세트(56장). 원인: (1) 위에서 내려다본 촬영 각도 때문에 원근 왜곡으로 직선이 테이퍼져 보이고 키가 눌려 보임 — README가 이미 예견한 위험(\"각도에 따라 왜곡/소실\"), (2) 일부 taper_step은 원래도 짧고 통통한 디자인이라 mug 경계(h/d≤1.5)에 가까워서 살짝만 눌려도 넘어감. 코드 버그 아님 — 2D 폭 프로파일 방식의 근본 한계, 촬영 각도(정면 비율)를 프로토콜대로 지키거나 원근 보정이 필요 |
| 2026-09-15 | Mask R-CNN | raw2 88.0% → **Test 39%** (49.0%p 하락) | straight → taper_step/taper_smooth로 분산, mug → taper_step 대량 오분류 | 같은 원인(촬영 각도). Mask R-CNN의 마스크 자체는 룰베이스보다 낫지만, 마스크 이후의 `classify_shape()`(폭 프로파일 기반)는 두 트랙이 공유하므로 각도 왜곡에는 똑같이 취약함 — 마스크 품질 문제가 아니라 기하학적 가정(정면 촬영)이 깨진 것 |
| 2026-09-15 | ResNet-18 | preprocess val 91.3% → **Test 58.9%** (32.4%p 하락) | mug → taper_step 대량 오분류(6장 중 4장, 표본 작음) | **같은 domain shift인데도 하락 후 절대 정확도가 룰베이스(18%)·Mask R-CNN(39%)보다 훨씬 높음.** 기하 규칙 대신 학습된 시각 패턴을 쓰는 게 촬영 각도 왜곡에 강하다는 가설 확인. 각도 왜곡을 겨냥한 `RandomPerspective` 증강 포함. 4트랙 중 처음으로 "회피"가 아니라 "정면 돌파"에 가까운 결과 |
