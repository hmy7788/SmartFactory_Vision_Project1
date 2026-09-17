# app/ — 판정 데모 화면

저장소 루트에서:

    app\run_demo.bat            (윈도우)
    streamlit run app/demo.py    (직접)

## 모델 넣는 법 (checkpoints/model/)
1. `meta.example.json`을 복사해 `meta.json`을 만들고, 학습한 사람한테 받은 값을 채운다
   — library/arch · img_size/resize/mean/std · classes 순서 · weights 파일명.
2. 가중치 파일(`best.pt`)을 같은 폴더에 넣는다. git에는 올라가지 않는다(.gitignore).
3. 실행. 값이 틀리면 첫 화면에서 원인이 뜬다(가중치 키 불일치·클래스 불일치). 
   전처리(resize)만은 틀려도 에러가 안 나고 점수가 떨어지니, 붙인 뒤 아는 사진 몇 장으로 확인한다.

## 구조
- `app/demo.py` — Streamlit 화면. 사진 업로드 / 카메라·폴더 실시간. 판정 아래 양식 해설.
- `src/pipeline.py` — meta.json + 가중치 → `Pipeline.predict(image_bgr)`. 모델 바꿀 때 이 파일은 안 건드린다.
- `src/explanation/class_profiles.yaml` — 형태별 해설(디자인·사조·대표 제품). 문구는 여기서만 고친다.
- `src/explanation/profiles.py` — 해설 로더 + 판정 규칙(확신도 기준).
- `src/sources.py` — 웹캠·폴더 프레임.
