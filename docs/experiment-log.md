# 실험 결과 로그

4갈래 트랙(룰베이스 / Mask R-CNN / ResNet / EfficientNet)의 실험 결과를 트랙별로 기록한다. **검증 대상은 시스템**이므로, 정확도 숫자만이 아니라 구현 난이도·추론 속도·실패 패턴도 함께 남긴다.

## 현재 순위 요약 (2026-09-15, `data/test` 104장 기준)

같은 실제 촬영 Test 세트(직접 촬영, 클래스당 10~38장, 클래스: straight/taper_smooth/taper_step/mug)로 전부 재평가한 최신 스냅샷. 아래 표만 보면 지금 가장 나은 파이프라인이 뭔지 바로 알 수 있다 — 아래쪽 트랙별 표는 그 결과에 이르기까지의 실험 과정(시행착오 포함) 기록.

| 순위 | 트랙 | Test 정확도 | Macro-F1 | 비고 |
|---|---|---|---|---|
| 🥇 1 | **ResNet-18 (`--no-freeze-backbone`)** | **79.8%** (83/104) | **0.769** | 백본까지 fine-tuning, 차등 LR. 전 실험 통틀어 최고. 체크포인트: `checkpoints/resnet18_shape.pth` |
| 2 | Mask R-CNN (제로샷) | 46.2% (48/104) | 0.414 | COCO 사전학습 그대로, fine-tuning 안 함(`src/deep_learning/dl1_maskrcnn/segment.py`) |
| 3 | Mask R-CNN (파인튜닝) | 15.4% (16/104) | 0.152 | 룰베이스 pseudo-label로 fine-tuning — **실사용 안 함**, 실패 사례로 기록만 유지 |
| 4 | 룰베이스 | 11.5% (12/104) | 0.104 | `src/rule_based/`. 거의 전부 mug로 쏠리는 구조적 편향 확인 |
| - | EfficientNet-B0 | 미착수 | - | |
| - | ResNet-50 | 미착수(18 vs 50 비교 아직 안 함) | - | |

정확도와 macro-F1 순위가 그대로 일치한다는 점이 중요하다 — 즉 룰베이스·Mask R-CNN 파인튜닝의 낮은 정확도가 "머그형에만 몰아서 찍어 숫자만 맞춘" 클래스 불균형 편법이 아니라, **전 클래스에 걸쳐 고르게 성능이 나쁘다**는 뜻(Macro-F1은 표본 수와 무관하게 클래스별 F1을 동일 가중치로 평균내므로). 특히 룰베이스는 mug precision이 0.077로 극히 낮아(recall 0.600은 높지만) "애매하면 mug로 찍는" 편향이 수치로도 그대로 드러난다.

**핵심 교훈**: 이 프로젝트의 제일 큰 domain shift 원인은 **촬영 각도(원근 왜곡)** — 위에서 내려다보고 찍으면 직선이 테이퍼져 보이고 키가 눌려 보여서, 폭 프로파일 같은 **기하학적 규칙에 의존하는 트랙(룰베이스, 그리고 그 규칙을 공유하는 Mask R-CNN)일수록 크게 무너진다.** ResNet처럼 **학습된 시각 패턴**(색상·질감·손잡이 모양·맥락 등)을 쓰는 방식이 이 왜곡에 훨씬 강하다는 게 이번 실험들로 반복 확인됨. Grad-CAM으로 봐도 ResNet은 배경이 아니라 물체 본체·손잡이에 정확히 집중하고 있었다(`reports/figures/resnet18/gradcam.png`).

---

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
| 2026-09-15 | Claude | `data/test` 확대판(104장, test2 병합)으로 재평가 — `src/rule_based/evaluate_test.py` 신규 작성(다른 트랙과 같은 패턴: confusion matrix를 `reports/figures/rule_based/`에 저장) | - | **11.5%(12/104)** | 이전 56장 기준(18%)보다도 낮아짐 — 표본이 늘면서 실제 약점이 더 뚜렷하게 드러남. 클래스별: straight 1/38(2.6%), taper_smooth 1/22(4.5%), taper_step 4/34(11.8%), mug 6/10(60.0%). **straight/taper_smooth/taper_step 거의 전부 mug로 쏠림**(26/38, 18/22, 28/34) — 원인은 기존과 동일(촬영 각도 원근 왜곡으로 키가 눌려 보여 mug 경계(h/d≤1.5)를 넘어버림, docs/troubleshooting.md 9/15 항목). 4트랙 중 유일하게 Test에서 mug가 제일 정확한 클래스(다른 트랙은 mug가 오히려 약하거나 평범함) — 룰베이스의 판정 로직 자체가 "애매하면 mug로 판정"하는 구조적 편향을 갖고 있음을 시사 |

## Mask R-CNN (`src/deep_learning/dl1_maskrcnn/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `segment.py` 최초 구현(COCO 사전학습 `maskrcnn_resnet50_fpn_v2`, 제로샷·fine-tuning 없음) + `rule_based_raw2_sample.ipynb`와 동일한 40장(seed=42)에 룰베이스와 나란히 비교 — `notebooks/maskrcnn_vs_rule_based.ipynb` | - (raw2, 미검증 라벨 기준 참고용) | - | **전체 75%(룰베이스) → 85%(Mask R-CNN)**. 클래스별: straight 90→100%, taper_smooth 50→70%, **taper_step 60→80%**(목표했던 개선), mug 100→90%(소폭 회귀). 추론 속도는 GPU(RTX 4050)에서 평균 150ms/장으로 룰베이스(CPU, 503ms/장, GrabCut이 병목)보다도 빠름. cup/bottle/vase/wine glass/bowl 중 점수 높은 COCO 검출을 사용 — 텀블러가 COCO 클래스에 없어서 근접 카테고리로 대체 |
| 2026-09-15 | Claude | 룰베이스 마스크를 pseudo-label 삼아 fine-tuning 시도 (배경/텀블러 2클래스, 평가용 40장은 학습에서 제외, `classify_shape()`가 폴더 라벨과 맞은 것만 채택) — `notebooks/maskrcnn_finetune_rulebase_pseudolabels.ipynb`. 학습 266장(3 epoch, 1197s, loss 0.376→0.140), GPU RTX 4050 | 동일 40장(held-out) 재평가: 78%(31/40) | - | **제로샷(85%)보다 오히려 나빠짐(78%) — 우려했던 리스크가 실제로 발생.** straight 100→80%, taper_smooth 70→60%, taper_step 80→70% 전부 회귀, mug만 90→100%(가장 많이 채택된 클래스라 치우침 발생). 원인: (1) 클래스별 pseudo-label 채택 수가 심하게 불균형(mug 88 / straight 81 / taper_smooth 64 / **taper_step 33** — 룰베이스가 약한 클래스일수록 채택 수도 적어져서 학습 데이터가 mug 쪽으로 편향), (2) 룰베이스 마스크의 거친 경계가 COCO 학습으로 얻은 정밀한 마스크 품질을 깎아먹은 것으로 추정. **결론: 이 체크포인트는 폐기, dl1 트랙은 제로샷 버전(`segment.py`의 기본 `load_model()`)을 계속 사용.** 체크포인트는 `checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth`에 남겨두되 실사용 안 함(참고 기록용) |
| 2026-09-15 | Claude | 위 파인튜닝 체크포인트를 폐기하지 않고 `data/test`(실제 촬영 104장)에서 제로샷과 직접 비교 — `src/deep_learning/dl1_maskrcnn/evaluate_test.py` | - | **제로샷 46.2%(48/104) vs 파인튜닝 15.4%(16/104)** | **raw2(85%→78%)보다 실제 Test에서 훨씬 더 크게 무너짐(3배 차이).** Confusion matrix 확인: 파인튜닝 모델이 거의 전부 "mug"로만 예측(straight 24/38, taper_smooth 19/22, taper_step 26/34가 mug로 쏠림) — 학습 데이터의 mug 편향(위 항목 참고)이 domain shift 상황에서 더 극단적으로 드러남(class collapse). **파인튜닝을 다시 시도해볼 이유가 없다는 결론을 재확인** — 진짜 마스크 정답 없이는 근본적으로 개선 안 됨. 그래프: `reports/figures/maskrcnn/{zeroshot,finetuned}_test_confusion_matrix.png` |

## ResNet-18/50 (`src/deep_learning/dl2_resnet/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `train.py` 최초 구현 — ResNet-18 ImageNet 사전학습, 백본 freeze(FC layer만 학습), `data/preprocess`(642장, 강한 신호 제외) 층화 80/20 분할, 증강에 `RandomPerspective` 포함(촬영 각도 왜곡을 직접 겨냥), 20 epoch, GPU RTX 4050(145s) | 91.3%(best epoch 12) | 58.9%(33/56, data/test 초기 56장) | **압도적 개선**: 같은 Test 세트에서 룰베이스 18%, Mask R-CNN 39% 대비 두 배 가까이 높음 — 기하학적 규칙(`classify_shape`) 대신 학습된 시각 패턴을 쓰는 게 촬영 각도 왜곡에 훨씬 강하다는 가설이 실측으로 확인됨. Test에서 mug만 1/6(17%)로 급락 — 표본이 6장뿐이라 노이즈 의심. **주의: 이 체크포인트는 이후 사고로 유실됨(아래 항목 참고), 수치는 참고용** |
| 2026-09-15 | Claude | `data/test`에 `data/test2` 48장 병합(104장, mug 10장으로 확대) 후 재평가하려다, **재학습 프로세스를 도중에 강제종료했는데 1 epoch째 체크포인트 저장이 먼저 끝나버려 기존 91.3% 체크포인트가 그걸로 덮어써짐** — `--eval-only`로 불러온 val_acc가 80.3%로 나와서 발견. `train.py`에 안전장치 추가(학습 중에는 `.tmp` 파일에만 저장, 전체 epoch가 끝까지 성공했을 때만 `os.replace`로 최종 반영 — 중간에 죽어도 기존 체크포인트 안전) 후 처음부터 재학습 | **89.8%**(best epoch 18) | **64.4%**(67/104, data/test 104장 — mug 포함 확대판) | mug 정상화 확인: Test에서 5/10(50%)로 회복 — 이전 1/6(17%)이 정말 표본 노이즈였음이 확인됨. 클래스별 Test 정확도: straight 21/38(55%), taper_smooth 15/22(68%), **taper_step 26/34(76%, 4트랙 중 처음으로 taper_step이 제일 강한 클래스)**, mug 5/10(50%). Val→Test 하락폭 25.3%p로 이전 56장 기준(32.4%p)보다도 줄어듦 — 표본이 커지니 추정치가 더 안정적. 그래프/체크포인트는 이 실행 결과로 덮어써짐(위 항목의 수치는 재현 불가) |
| 2026-09-15 | Claude | `--no-freeze-backbone`(백본까지 fine-tuning) 실험. 백본은 낮은 LR(`args.lr*0.1`), 새 FC layer는 원래 LR을 쓰는 차등 학습률 적용(우선 `checkpoints_experiment/`에 저장해 기존 체크포인트 보호 후 비교, 결과가 확실히 나아서 공식 체크포인트로 승격) | **96.9%**(best epoch 12/14) | **79.8%**(83/104) | **4트랙·모든 실험 통틀어 최고 기록.** 프리즈 버전(Val 89.8%→Test 64.4%) 대비 Val·Test 둘 다 큰 폭 상승, Val→Test 하락폭도 25.3%p→**17.0%p**로 축소 — domain shift 자체에 더 강해짐. 학습 시간은 145s로 프리즈 버전과 거의 동일(이 작은 데이터셋에서는 역전파 비용이 병목이 아니었음). Test confusion matrix: straight 29/38(76%), taper_smooth 13/22(59%), taper_step 32/34(94%), mug 9/10(90%) — 전 클래스 고르게 개선. train_acc가 97~99%로 val보다 훨씬 높아 과적합 신호는 있으나(train_loss↓ 지속, val_loss 진동), best-epoch 체크포인트 저장 덕에 실사용엔 문제 없음. **이후 dl2_resnet 실험은 기본적으로 이 옵션을 쓸 것.** |
| 2026-09-15 | Claude | Grad-CAM으로 판단 근거 시각화 — `src/deep_learning/dl2_resnet/gradcam.py`(`layer4` 활성화·그래디언트 기반), data/test에서 클래스당 4장씩 원본+히트맵 그리드 저장 | - | - | 정성 분석(정확도 지표 아님). **배경이 아니라 물체 본체·손잡이에 정확히 집중** — 배경을 몰래 학습한 징후 없음. taper_step·mug 둘 다 손잡이 영역에 강하게 반응(둘 다 손잡이 있는 디자인이 많아 학습된 유효한 단서로 보임). 오분류 사례에서 원인 짐작 가능: straight가 mug로 오분류된 건(0.96 확신) 주목 영역이 몸통 중간 가로 띠에만 집중돼 전체 높이 비율을 못 본 것으로 보임 — 확신도 자체가 낮았던 오분류(taper_smooth→straight, 0.61)는 모델도 헷갈렸다는 신호로 해석 가능(주의가 입구·아래쪽 두 군데로 분산). 그래프: `reports/figures/resnet18/gradcam.png` |

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
| 2026-09-15 | (재측정) | `data/test`가 56장→104장으로 확대(test2 병합, mug 6→10장)된 뒤 같은 세트로 재측정 | - | Mask R-CNN 제로샷 39%→**46.2%**(48/104), ResNet-18(freeze) 58.9%→**64.4%**(67/104, 재학습 포함) | 표본이 커지니 두 트랙 다 정확도가 올라가고 순위는 그대로 유지(ResNet > Mask R-CNN 제로샷). mug 표본이 늘면서(6→10) 이전 "mug만 급락" 노이즈가 해소됨. **참고**: 룰베이스 pseudo-label로 fine-tuning한 Mask R-CNN 체크포인트도 같은 104장으로 평가해봤는데 **15.4%(16/104)로 제로샷의 1/3 수준** — confusion matrix상 거의 전부 mug로만 예측(class collapse). raw2 기준(85%→78%)보다 실제 Test에서 훨씬 크게 무너짐 — fine-tuning 재시도 안 하는 게 맞다는 결론 재확인 |
| 2026-09-15 | ResNet-18(백본 unfreeze) | preprocess val 96.9% → **Test 79.8%**(83/104) (17.0%p 하락) | 상대적으로 taper_smooth가 제일 약함(13/22, 59%), 나머지는 76~94% | **현재까지 전 실험 통틀어 최고 기록이자 최소 하락폭.** 백본을 프리즈한 버전(Val 89.8%→Test 64.4%, 25.3%p 하락) 대비 절대 정확도·하락폭 둘 다 개선 — "학습된 시각 패턴이 각도 왜곡에 강하다"는 가설이 백본까지 풀었을 때 더 강하게 확인됨. 학습 시간은 프리즈 버전과 거의 동일(145s, 이 데이터 규모에서는 역전파 비용이 병목 아님) — 시간 비용 없이 큰 이득. 이후 dl2_resnet 실험 기본값으로 채택 |
