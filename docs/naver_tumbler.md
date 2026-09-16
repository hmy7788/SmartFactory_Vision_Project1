# 네이버 텀블러 이미지 수집기 (NAVER API HUB)

네이버 클라우드 플랫폼의 NAVER API HUB 이미지 검색 API(`search/v1/image`)를 통해
후보 이미지를 수집하고, 사람이 작성한 검수표를 재현 가능하게 적용해 미분류 데이터셋을 만드는 도구다.
검색어는 데이터 출처일 뿐 형태 라벨로 사용하지 않는다.

## 사전 준비

네이버 클라우드 플랫폼(NCP)의 **NAVER API HUB**에서 애플리케이션을 생성하고 **이미지(Search Image API)**를
선택한 후, 발급된 API 인증키를 현재 PowerShell 세션의 환경변수로 지정한다. 키를 소스 코드나 Git에 저장하지 않는다.

```powershell
$env:NAVER_CLIENT_ID = "발급받은 Client ID (또는 Access Key)"
$env:NAVER_CLIENT_SECRET = "발급받은 Client Secret"
```

## 1. 후보 수집

먼저 100장으로 파일럿을 수행한다. 명령을 다시 실행하면 SQLite 체크포인트를 이용해 이미
완료된 검색 구간과 같은 상품을 건너뛴다.

```powershell
.\.venv\Scripts\python.exe -m src.data_collection.naver_tumbler collect --candidate-limit 100
```

파일럿 확인 후 후보 상한을 늘린다. 기본값은 3,000장이고 다운로드 동시성은 코드에서 2로
고정되어 있다.

```powershell
.\.venv\Scripts\python.exe -m src.data_collection.naver_tumbler collect --candidate-limit 3000
```

결과는 `data/naver_tumbler/` 아래에 저장된다.

- `raw/`: EXIF 방향을 적용해 저장한 후보 원본
- `crawl.sqlite3`: 검색·다운로드·실패·중복 체크포인트
- `review/review.csv`: 검수 입력 파일
- `summary.json`: 후보·중복·실패·검수 상태 수량

API는 대표 썸네일을 제공하므로 원본이 크기 기준에 못 미치면 실패로 기록한다. 도구는 URL의
크기 매개변수를 변경하거나 존재하지 않는 고해상도 주소를 추측하지 않는다.

## 2. 검수

검수용 연락판은 후보 25장당 한 장으로 생성한다.

```powershell
.\.venv\Scripts\python.exe -m src.data_collection.naver_tumbler contact-sheet
```

`review/review.csv`에서 다음 열만 사람이 수정한다.

- `decision`: `accept`, `reject`, `pending` 중 하나
- `crop_x`, `crop_y`, `crop_width`, `crop_height`: 원본 픽셀 좌표. 크롭하지 않으면 모두 빈 값
- `reason`: 판단 근거

`label`은 이 수집 단계에서 항상 빈 값으로 유지된다. 텀블러 전체 윤곽과 바닥이 보여야 하며,
약한 손 가림은 몸통의 기울기와 단차가 판독될 때만 허용한다.

```powershell
.\.venv\Scripts\python.exe -m src.data_collection.naver_tumbler apply-review
```

크롭 좌표가 일부만 입력되었거나 범위를 벗어난 경우, 크롭 결과가 128×256px 기준에 미달한
경우, 승인 결과가 픽셀 중복인 경우에는 실패한다. 오류는
`review/apply_errors.json`에서 확인한다.

## 3. 최종 500장 내보내기

```powershell
.\.venv\Scripts\python.exe -m src.data_collection.naver_tumbler export --target 500
```

승인 이미지가 500장보다 적으면 품질 기준을 낮추거나 부분 결과를 완료로 표시하지 않고
명령이 실패한다. 성공 결과는 `final/unlabeled/tumbler_0001.jpg`부터 저장되고,
`final/manifest.csv`가 원본 URL·상품 그룹·크롭 좌표를 연결한다.

## 주의 사항

- 접근 제한이나 HTTP 401/403/429가 발생하면 우회하지 않고 중단한다.
- API 키는 로그와 DB에 기록되지 않는다.
- 같은 상품 ID, 동일 픽셀 이미지는 후보 수를 늘리지 않는다. 지각 해시는 메타데이터에 남겨
  육안으로 근접 중복을 검수할 수 있게 한다.
- 웹 상품 이미지는 출처·이용 조건을 확인하고 비전 프로젝트 실습 범위에서 사용한다.

