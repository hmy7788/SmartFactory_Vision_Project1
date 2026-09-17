# app/ — 판정 데모 화면

저장소 루트에서:

    app\run_demo.bat            (윈도우 — 더블클릭. torch+streamlit이 깔린 python을 알아서 찾는다)
    streamlit run app/demo.py    (직접, 활성화된 env에서)

## 모델 넣는 법 (checkpoints/model/)
1. `meta.json` — 지금 값은 팀 ConvNeXt-Tiny(`src/models/convnext/inference.py`) 기준으로 채워져 있다.
   모델을 바꾸면 `meta.example.json`을 보고 library/arch · img_size/resize/mean/std · classes 순서 · weights 파일명을 고친다.
2. 가중치 파일(`best_convnext_v5.pth`, 111MB)을 같은 폴더에 넣는다. git에는 올라가지 않는다(.gitignore) — 학습한 팀원한테 직접 받는다.
3. `check_model.bat` — 라벨된 사진 폴더(`<폴더>/<클래스>/*.jpg`, 저장소 옆에 두면 알아서 찾음)로 정확도를 재고,
   `--compare`로 팀원 `inference.py`와 같은 답을 내는지 대조한다. 라벨 일치 100%·확률 차이 ≈0 이면 붙은 것.
   값이 틀리면(가중치 키 불일치·클래스 불일치) 로드 시점에 에러가 뜬다. 전처리(resize)만은 틀려도 에러가 안 나고
   점수만 떨어지므로 이 검사가 필요하다.
4. `app\run_demo.bat`.

## 실시간 탭 — 카메라가 어디서 열리는가
- `streamlit-webrtc`가 깔려 있으면 **보는 사람의 브라우저** 카메라를 쓴다. 팀원이 내 주소로 접속하면
  팀원 기기의 카메라로 판정된다. 설치: `install_webrtc.bat`.
- 없으면 **streamlit을 켠 컴퓨터**의 카메라로 대체된다. 이때 팀원이 접속하면 보이는 건 내 카메라다.
- 브라우저는 주소가 **https이거나 localhost**일 때만 카메라를 내준다(보안 컨텍스트). 같은 와이파이의
  `http://192.168.x.x:8501`로 들어오면 권한 창이 아예 안 뜬다 — 코드 문제가 아니다. 다른 기기에서
  쓰려면 터널로 https 주소를 만들어야 한다: `cloudflared tunnel --url http://localhost:8501`.
- 영상과 판정은 분리돼 있다. 영상은 카메라 속도대로 흐르고, 판정은 되는 만큼만(CPU에서 초당 3회쯤)
  최신 프레임에 대해 돈다. 밀린 프레임은 버린다 — 몇 초 전 장면을 판정해봐야 의미가 없다.

## 구조
- `app/demo.py` — Streamlit 화면. 사진 업로드 / 실시간. 판정 아래 양식 해설.
- `src/pipeline.py` — meta.json + 가중치 → `Pipeline.predict(image_bgr)`. GPU가 있으면 자동으로 쓴다. 모델 바꿀 때 이 파일은 안 건드린다.
- `src/live.py` — 브라우저 카메라(webrtc) 실시간 판정. 프레임 스레드와 판정 스레드 분리.
- `src/models/convnext/` — 팀원이 준 ConvNeXt 추론 코드 원본(GradCAM 포함). 대시보드는 직접 쓰지 않고 meta.json으로 같은 모델을 복원한다.
- `src/explanation/class_profiles.yaml` — 형태별 해설(디자인·사조·대표 제품). 문구는 여기서만 고친다.
- `src/explanation/profiles.py` — 해설 로더 + 판정 규칙(확신도 기준).
- `src/sources.py` — 이 컴퓨터 카메라 프레임(대비책) + 이미지 읽기.
- `tools/check_model.py` — 정확도·동일성 검사. `tools/pyfind.ps1` — bat이 쓰는 python 탐색기.
