# 실촬영 텀블러 4종 형태 분류 최종 평가 보고서

- **총 평가 사진 수**: 12 장
- **고유 실제 텀블러 제품 수**: 8 개
- **주요 평가 지표 (Macro F1)**: **`1.0000`**
- **보조 평가 지표 (Accuracy)**: `1.0000`

## 1. 클래스별 상세 성능

| 클래스 | Precision | Recall | F1-Score | 사진 수(Support) |
|---|---|---|---|---|
| `straight` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `taper_smooth` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `taper_step` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `mug` | 1.0000 | 1.0000 | 1.0000 | 3 |

## 2. 6조건별 성능 분포

| 촬영 조건 | 사진 수 | 포함된 클래스 | Macro F1 | Accuracy |
|---|---|---|---|---|
| `plain_normal_front` | 2 | straight, taper_step | 0.5000 | 1.0000 |
| `plain_normal_oblique_left` | 2 | straight, taper_step | 0.5000 | 1.0000 |
| `plain_normal_oblique_right` | 2 | straight, taper_step | 0.5000 | 1.0000 |
| `daily_front` | 2 | taper_smooth, mug | 0.5000 | 1.0000 |
| `light_front` | 2 | taper_smooth, mug | 0.5000 | 1.0000 |
| `handheld` | 2 | taper_smooth, mug | 0.5000 | 1.0000 |

## 3. 오분류 사례 분석

총 0건 오분류 발생:

| 파일명 | 제품 ID | 촬영 조건 | 정답 라벨 | 예측 라벨 | 예측 확신도 |
|---|---|---|---|---|---|