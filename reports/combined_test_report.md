# 리비전 실촬영(126장) + 웹 밸리데이션(89장) 통합 테스트 종합 평가 보고서

- **평가 모델**: `exp_b_seed43` (`deit_small_patch16_224.fb_in1k`)
- **총 평가 사진 수**: **215 장** (리비전 실촬영 126장 + 웹 밸리데이션 89장)

## 1. 3가지 평가 데이터셋 요약 성과 비교

| 평가 데이터셋 | 사진 수 | Macro F1 | Accuracy | 오분류 수 | 비고 |
|---|---|---|---|---|---|
| **① 리비전 실촬영 (`test1_orientation_backup`)** | 126장 | **`0.8161` (81.61%)** | `0.8095` (80.95%) | 24장 | 정제된 실촬영 각도 데이터 |
| **② 웹 밸리데이션 (`web_validation`)** | 89장 | **`0.9678` (96.78%)** | `0.9663` (96.63%) | 3장 | 미학습 368그룹 분리 검증셋 |
| **③ 통합 테스트셋 (① + ②)** | **215장** | **`0.8784` (87.84%)** | **`0.8744` (87.44%)** | **27장** | 전체 비학습 데이터 총결합 |

## 2. 통합 테스트셋(215장) 클래스별 세부 성능

| 클래스 | Support (사진 수) | Precision | Recall | F1-Score | 분석 특징 |
|---|---|---|---|---|---|
| `straight` | 64장 | 0.8525 | 0.8125 | **`0.8320`** | 실촬영 부감 각도에서 taper와 일부 혼동 |
| `taper_smooth` | 52장 | 0.7869 | 0.9231 | **`0.8496`** | 완만한 경사면 인식, 높은 재현율(Recall) |
| `taper_step` | 55장 | 0.9796 | 0.8727 | **`0.9231`** | 단차 꺾임선 포착으로 최고 성능 달성 |
| `mug` | 44장 | 0.9091 | 0.9091 | **`0.9091`** | 낮은 높이-너비 비율 정확 식별 |

## 3. 통합 테스트셋 오분류 분석 (총 27건)

| 출처 | 파일명 | 정답 라벨 | 예측 라벨 | 예측 확신도 |
|---|---|---|---|---|
| 실촬영 | `KakaoTalk_20260915_114237129_03.jpg` | `straight` | `taper_smooth` | 0.6498 |
| 실촬영 | `KakaoTalk_20260915_114237129_05.jpg` | `straight` | `taper_smooth` | 0.7771 |
| 실촬영 | `KakaoTalk_20260915_154410128_16.jpg` | `straight` | `taper_smooth` | 0.6150 |
| 실촬영 | `KakaoTalk_20260915_154410128_18.jpg` | `straight` | `taper_smooth` | 0.5075 |
| 실촬영 | `KakaoTalk_20260916_091413554_09.jpg` | `straight` | `taper_smooth` | 0.5439 |
| 실촬영 | `KakaoTalk_20260916_091413554_14.jpg` | `straight` | `taper_smooth` | 0.9888 |
| 실촬영 | `KakaoTalk_20260916_091413554_15.jpg` | `straight` | `taper_smooth` | 0.9912 |
| 실촬영 | `KakaoTalk_20260916_091413554_16.jpg` | `straight` | `taper_smooth` | 0.9893 |
| 실촬영 | `KakaoTalk_20260916_091413554_21.jpg` | `straight` | `mug` | 0.6990 |
| 실촬영 | `KakaoTalk_20260916_091413554_23.jpg` | `straight` | `mug` | 0.4418 |
| 실촬영 | `KakaoTalk_20260916_091413554_24.jpg` | `straight` | `taper_smooth` | 0.6148 |
| 실촬영 | `KakaoTalk_20260916_093646477.jpg` | `taper_smooth` | `straight` | 0.9791 |
| 실촬영 | `KakaoTalk_20260916_093646477_01.jpg` | `taper_smooth` | `straight` | 0.8440 |
| 실촬영 | `KakaoTalk_20260916_093646477_02.jpg` | `taper_smooth` | `straight` | 0.9490 |
| 실촬영 | `KakaoTalk_20260916_093646477_03.jpg` | `taper_smooth` | `straight` | 0.7800 |
| 실촬영 | `KakaoTalk_20260915_114226226_03.jpg` | `taper_step` | `taper_smooth` | 0.9383 |
| 실촬영 | `KakaoTalk_20260915_114226226_18.jpg` | `taper_step` | `taper_smooth` | 0.9764 |
| 실촬영 | `KakaoTalk_20260915_114226226_20.jpg` | `taper_step` | `straight` | 0.7180 |
| 실촬영 | `KakaoTalk_20260915_154417925_08.jpg` | `taper_step` | `straight` | 0.5878 |
| 실촬영 | `KakaoTalk_20260916_091505020_14.jpg` | `taper_step` | `mug` | 0.5604 |
| 실촬영 | `KakaoTalk_20260915_114237129_07.jpg` | `mug` | `taper_smooth` | 0.9768 |
| 실촬영 | `KakaoTalk_20260915_114237129_08.jpg` | `mug` | `taper_step` | 0.7227 |
| 실촬영 | `KakaoTalk_20260915_114237129_09.jpg` | `mug` | `taper_smooth` | 0.7360 |
| 실촬영 | `KakaoTalk_20260915_154410128_08.jpg` | `mug` | `straight` | 0.7180 |
| 웹Val | `web_straight_75.jpg` | `straight` | `mug` | 0.9789 |
| 웹Val | `web_taper_step_95.jpg` | `taper_step` | `straight` | 0.9473 |
| 웹Val | `web_taper_step_99.jpg` | `taper_step` | `straight` | 0.5548 |
