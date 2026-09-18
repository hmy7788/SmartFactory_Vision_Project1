# K-Trip iOS demo

텀블러 모양의 건축물로 이루어진 대한민국을 여행하며 네 가지 건축 양식을 모으는 도감입니다. 아이폰 외형과 Dynamic Island 내부에 Discover / Camera / My atlas를 구성했습니다. 영어·한국어를 지원합니다.

## 설치 및 실행

저장소 루트에서 실행하세요. Python 3.10 이상이 필요합니다.

```powershell
cd k_trip_ios
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

팀에서 전달받은 **best_convnext_v5.pth**를 `k_trip_ios/checkpoints/best_convnext_v5.pth`에 놓으세요. 모델 가중치는 저장소의 기존 규칙에 따라 Git에 포함하지 않습니다. `checkpoints/meta.json`은 함께 제공됩니다. 가중치가 없으면 서버가 시작되지 않습니다.

```powershell
.\run.ps1
```

[http://localhost:8765](http://localhost:8765)를 열고 Camera → Enable camera에서 카메라 권한을 허용하세요. 가상환경을 활성화한 터미널에서는 `run.bat`도 사용할 수 있습니다. macOS/Linux에서는 가상환경을 활성화하고 `python -m uvicorn server:app --host 127.0.0.1 --port 8765 --ws-max-size 6291456`으로 실행합니다.

카메라는 localhost 또는 HTTPS에서 사용할 수 있습니다. 내장 브라우저가 카메라를 거부하면 Chrome/Edge에서 여세요. 사진 업로드도 지원합니다. 실제 iPhone에서 접속하려면 별도 HTTPS와 네트워크 바인딩 설정이 필요합니다. 기본 서버는 로컬 PC 전용입니다.

## 기능

- 브라우저 카메라 30fps 요청, 실제 재생 FPS와 추론 갱신 속도 분리 표시
- WebSocket으로 한 프레임씩 전송하여 대기열 누적 방지
- 별도 촬영 버튼과 사진 업로드, 클래스별 확률, Grad-CAM
- 기존 팀 프로젝트의 양식 명칭·주요 특징·구조적 이유·건축 사조·대표 건축물
- 도감 양식 아이콘 선택, 발견 여부와 수집 수, 사진 필터와 상세 설명 연동
- IndexedDB에 사진과 분석 저장, 새로고침 유지, 삭제 및 분석 다시 보기

## 모델과 추론

- torchvision ConvNeXt-Tiny, `best_convnext_v5.pth` (약 111.4 MB)
- 클래스 순서: `mug`, `straight`, `taper_smooth`, `taper_step`
- 전체 이미지 Resize((224,224), bilinear) → ImageNet Normalize
- GPU 지원 PyTorch와 GPU가 있으면 CUDA, 그 외에는 CPU 자동 선택
- 오른쪽 모델명은 `/api/health`의 실제 모델 정보에서 표시
- 새 캡처에는 모델 아키텍처와 리사이즈 방식도 저장. 기존 도감 사진은 자동 재분류하지 않음
- 다른 체크포인트 폴더는 `KTRIP_CHECKPOINT` 환경변수로 지정

카메라 프레임은 최대 640px JPEG로 전송하고 추론 응답 80ms 후 다음 프레임을 보냅니다. 개발 PC의 CPU에서 추론 약 142ms, 확률 갱신 약 4Hz 전후가 예상되며 기기와 부하에 따라 달라집니다. 카메라 재생과 추론 갱신 속도는 별도입니다. 캡처는 최대 1280px 사진에 Grad-CAM을 계산합니다. 카메라 정지·탭 이동·백그라운드 전환 시 카메라 트랙을 종료합니다.

70% 미만 결과는 잠정 분류로 저장하고 4종 수집 달성에서 제외합니다. 이 모델은 사물 부재나 미학습 사물을 검출하지 않습니다. Grad-CAM은 모델의 주목 영역이며 정답의 증거가 아닙니다. 형태 해설은 세계관의 양식 안내이며 실제 윤곽을 측정한 결과와 구분합니다.

## 코드 구성과 출처

- `server.py`: FastAPI 캡처 API 및 WebSocket 추론. 모델 접근을 직렬화하여 Grad-CAM과 실시간 추론 충돌 방지
- `static/`: 빌드 과정 없는 HTML/CSS/JavaScript UI. `profiles.js`는 기존 `src/explanation/class_profiles.yaml`의 한국어 내용을 옮기고 영어 번역을 추가
- `src/pipeline.py`: 팀 웹 데모의 메타데이터 기반 로더를 재사용. 히트맵은 실제 모델 입력 이미지에서 복원
- `src/gradcam.py`: 팀 Grad-CAM 수식을 사용하고 임시 hook / autograd.grad로 추론 후 훅이 남지 않도록 수정
- `checkpoints/meta.json`: 팀 ConvNeXt 추론 코드의 모델 구성·전처리·클래스 순서

사진은 서버에 영구 저장하지 않으며 같은 브라우저·origin의 IndexedDB에 보관됩니다. 브라우저 데이터 삭제 시 도감도 사라집니다.

## 검증

모델 파일 배치와 의존성 설치 후 `k_trip_ios`에서 실행하세요.

```powershell
python -m unittest test_demo -v
```

실제 모델로 API, 확률 합, Grad-CAM 이후 반복 추론, 잘못된/과대 입력 거부, 학습 전처리 일치를 검증합니다. 브라우저에서는 사진 분석·히트맵·도감 영구 저장·양식 필터·상세 설명·한영 전환을 확인했습니다. 개발 중 내장 브라우저의 카메라 권한이 거부되어 실제 카메라 FPS는 측정하지 못했습니다.
