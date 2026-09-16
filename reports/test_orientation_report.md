# 실촬영 테스트셋(`test_orientation`) 4종 형태 분류 최종 평가 보고서

- **테스트 데이터 위치**: `C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\test_orientation`
- **총 테스트 사진 수**: 135 장
- **평가 모델**: `exp_b_seed44` (`deit_small_patch16_224.fb_in1k`)
- **주요 평가 지표 (Macro F1)**: **`0.8000` (80.00%)**
- **보조 평가 지표 (Accuracy)**: **`0.7926` (79.26%)**

## 1. 모델별 성능 비교 (다중 시드 및 앙상블)

| 모델 / 가중치 | Macro F1 | Accuracy | 오분류 수 |
|---|---|---|---|
| `exp_b_seed44` | **`0.8000`** (80.00%) | `0.7926` (79.26%) | 28장 / 135장 |
| `exp_b_seed42` | **`0.8021`** (80.21%) | `0.7926` (79.26%) | 28장 / 135장 |
| `exp_b_seed43` | **`0.8157`** (81.57%) | `0.8074` (80.74%) | 26장 / 135장 |
| `ensemble` | **`0.8000`** (80.00%) | `0.7926` (79.26%) | 28장 / 135장 |

## 2. 클래스별 상세 성능 (`exp_b_seed44`)

| 클래스 | Precision | Recall | F1-Score | Support (사진 수) |
|---|---|---|---|---|
| `straight` | 0.7805 | 0.7273 | **0.7529** | 44 |
| `taper_smooth` | 0.6667 | 0.8333 | **0.7407** | 36 |
| `taper_step` | 0.9355 | 0.8788 | **0.9062** | 33 |
| `mug` | 0.8889 | 0.7273 | **0.8000** | 22 |

## 3. 오분류 사례 상세 분석 (총 28건)

| 파일명 | 정답 라벨 | 예측 라벨 | 예측 확신도 | 원본 라벨 확률 |
|---|---|---|---|---|
| `KakaoTalk_20260915_114237129_03.jpg` | `straight` | `taper_smooth` | 0.6988 | 0.2648 |
| `KakaoTalk_20260915_114237129_04.jpg` | `straight` | `taper_smooth` | 0.4195 | 0.4005 |
| `KakaoTalk_20260915_114237129_05.jpg` | `straight` | `taper_smooth` | 0.8060 | 0.1681 |
| `KakaoTalk_20260915_114237129_23.jpg` | `straight` | `taper_smooth` | 0.5643 | 0.1672 |
| `KakaoTalk_20260915_154410128_16.jpg` | `straight` | `taper_smooth` | 0.5920 | 0.1877 |
| `KakaoTalk_20260915_154410128_18.jpg` | `straight` | `taper_smooth` | 0.4890 | 0.2424 |
| `KakaoTalk_20260916_091413554_14.jpg` | `straight` | `taper_smooth` | 0.9837 | 0.0136 |
| `KakaoTalk_20260916_091413554_15.jpg` | `straight` | `taper_smooth` | 0.9891 | 0.0061 |
| `KakaoTalk_20260916_091413554_16.jpg` | `straight` | `taper_smooth` | 0.9836 | 0.0119 |
| `KakaoTalk_20260916_091413554_21.jpg` | `straight` | `mug` | 0.7271 | 0.2193 |
| `KakaoTalk_20260916_091413554_23.jpg` | `straight` | `mug` | 0.4615 | 0.2769 |
| `KakaoTalk_20260916_091413554_24.jpg` | `straight` | `taper_smooth` | 0.5967 | 0.3704 |
| `KakaoTalk_20260915_154410128_01.jpg` | `taper_smooth` | `straight` | 0.5651 | 0.4252 |
| `KakaoTalk_20260916_091505020_07.jpg` | `taper_smooth` | `taper_step` | 0.6880 | 0.2353 |
| `KakaoTalk_20260916_093646477.jpg` | `taper_smooth` | `straight` | 0.9866 | 0.0083 |
| `KakaoTalk_20260916_093646477_01.jpg` | `taper_smooth` | `straight` | 0.8557 | 0.1022 |
| `KakaoTalk_20260916_093646477_02.jpg` | `taper_smooth` | `straight` | 0.9665 | 0.0045 |
| `KakaoTalk_20260916_093646477_03.jpg` | `taper_smooth` | `straight` | 0.8164 | 0.1695 |
| `KakaoTalk_20260915_114226226_03.jpg` | `taper_step` | `taper_smooth` | 0.9487 | 0.0506 |
| `KakaoTalk_20260915_114226226_18.jpg` | `taper_step` | `taper_smooth` | 0.9830 | 0.0166 |
| `KakaoTalk_20260915_114226226_20.jpg` | `taper_step` | `straight` | 0.7533 | 0.2466 |
| `KakaoTalk_20260915_154417925_08.jpg` | `taper_step` | `straight` | 0.6540 | 0.3455 |
| `KakaoTalk_20260915_114237129_07.jpg` | `mug` | `taper_smooth` | 0.9832 | 0.0088 |
| `KakaoTalk_20260915_114237129_08.jpg` | `mug` | `taper_step` | 0.7554 | 0.1615 |
| `KakaoTalk_20260915_114237129_09.jpg` | `mug` | `taper_smooth` | 0.6912 | 0.0027 |
| `KakaoTalk_20260915_154410128_08.jpg` | `mug` | `straight` | 0.7524 | 0.0514 |
| `KakaoTalk_20260916_091413554.jpg` | `mug` | `taper_smooth` | 0.5913 | 0.4050 |
| `KakaoTalk_20260916_091505020_01.jpg` | `mug` | `straight` | 0.5488 | 0.3581 |
