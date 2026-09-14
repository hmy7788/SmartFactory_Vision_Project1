## Claude Code 프롬프트
```
「텀블러 형태·재질 분류 및 설명 시스템」 프로젝트를 시작하려고 해.
아래 설계를 기반으로 프로젝트 스캐폴딩(디렉토리, 코드 스켈레톤, 환경설정)을 먼저 구축해줘.
실제 데이터(크롤링·촬영)는 아직 없으니, 데이터 없이도 진행 가능한 부분부터 진행해줘.

## 프로젝트 배경
최종 목표는 "건축 양식 판별"이지만, 건축물 데이터 직접 수집이 어려워 동형 구조의 대리 도메인인
"텀블러 형태 판별"로 치환해 파이프라인을 먼저 구현·검증한다. 검증 대상은 모델이 아니라 시스템이다.

## 분류 체계
- 형태(4종): 직선 원통형(straight) / 연속 테이퍼형(taper_smooth) / 단차 테이퍼형(taper_step) / 머그형(mug)
- 재질(2종): 플라스틱 / 스테인리스
- 판정 규칙(형태):
  1) 높이가 지름의 1.5배 이하면 머그형 (손잡이 유무는 무시)
  2) 아래 지름이 위 지름의 95% 이상, 윤곽 거의 수직이면 직선 원통형
  3) 꺾임점 없이 매끄럽게 좁아지면 연속 테이퍼형
  4) 윤곽에 뚜렷한 꺾임이 1회 이상 있으면 단차 테이퍼형
  5) 우선순위: 실루엣이 손잡이보다 우선 (손잡이 달린 테이퍼는 머그형이 아니라 연속 테이퍼형)

## 1. 디렉토리 구조 생성
아래 구조로 폴더를 생성해줘 (README.md의 "디렉토리 구조 추천" 섹션과 동일):
tumbler-classification/
├── data/{raw, train/{straight,taper_smooth,taper_step,mug}, val/{...}, test/{...}, _hold}/
├── src/{data_collection, rule_based, models/{dl1_maskrcnn, dl2_resnet, efficientnet},
│        reconstruction_3d, explanation}/
├── app/
├── notebooks/
├── checkpoints/
└── reports/figures/

## 2. requirements.txt 작성
Python 3.10 기준: opencv-python, torch, torchvision, timm(EfficientNet/ViT용),
scikit-learn(HOG/SVM baseline), pandas, numpy, matplotlib, seaborn, gradio,
naver-shopping API 호출용 requests, Pillow

## 3. src/rule_based/shape_classifier.py 구현
- 텀블러 실루엣 마스크를 입력받아, 세로 방향 N등분(기본 10구간) 폭 프로파일을 계산하는 함수
- cv2.convexHull + cv2.convexityDefects로 손잡이(비대칭 돌출부) 검출 함수
- 위 판정 규칙(1~5)을 그대로 구현한 classify_shape() 함수
- 머그형 판별을 "손잡이 유무"가 아니라 "높이/지름 비율 1.5 이하"로 우선 판정하도록 주의해서 구현

## 4. src/data_collection/ 스켈레톤
- naver_shopping_api.py: 네이버 쇼핑 API로 클래스별 키워드 검색 후 이미지 다운로드하는 함수 (API 키는 환경변수로)
- augmentation.py: 회전·플립·색상 변환 증강 함수 (torchvision.transforms 기반)

## 5. src/models/ 스켈레톤
- dl2_resnet/train.py: torchvision의 사전학습 ResNet-18/50을 불러와 마지막 FC layer만
  우선 fine-tuning하는 학습 스크립트 (백본 freeze 옵션 포함)
- efficientnet/train.py: timm 라이브러리로 EfficientNet-B0 fine-tuning 스크립트
- dl1_maskrcnn/: torchvision의 Mask R-CNN(ResNet50+FPN)을 로드하고, 예측된 Mask를
  이진화해서 src/rule_based/shape_classifier.py의 width profile 함수를 재사용하도록 연결하는 구조
  (형태: Mask 기반 기하 분석, 재질: 별도 ResNet50 Material Classification head)

## 6. src/pipeline.py 구현
이미지 업로드 → 전처리(Resize, Normalize) → 모델 추론 → Confidence 포함 결과 반환 →
형태·재질별 설명 텍스트(src/explanation/에서 로드) 반환하는 통합 함수 인터페이스 정의
(내부 모델은 아직 학습 전이므로 더미 응답으로 동작 검증)

## 7. app/gradio_app.py
이미지 업로드 → pipeline.py 호출 → 형태/재질/Confidence/설명을 보여주는 간단한 Gradio UI

## 8. README.md
이미 작성된 프로젝트 개요·분류 체계·데이터 구성·촬영 프로토콜·일정을 그대로 유지하고,
실행 방법(설치, 데이터 배치, 학습, 데모 실행 순서) 섹션만 추가해줘.

데이터 의존적인 부분은 더미/샘플 이미지로 동작 검증까지 해줘.
```