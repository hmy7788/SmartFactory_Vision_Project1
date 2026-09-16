# SSG 텀블러 이미지 수집 봇

대상: <https://www.ssg.com/search.ssg?target=all&query=%ED%85%80%EB%B8%94%EB%9F%AC>

## 수집 범위와 확인된 제약

- 검색 결과의 **모든 페이지에 노출되는 상품 카드 이미지**를 수집한다. 상품 상세페이지·리뷰 이미지·상세 설명의 긴 이미지는 범위에 포함하지 않는다.
- 2026-09-15 브라우저 조사 당시 페이지당 40개 상품, 마지막 이동 페이지는 250이었다. 검색 순위와 상품 수는 변하므로 250을 코드에 고정하지 않는다. “전부”는 사이트가 검색 결과로 공개하는 범위이며, SSG 전체 상품 목록을 뜻하지 않는다.
- 카드의 `srcset`에 공개된 가장 큰 이미지를 사용한다. 확인한 카드에서는 232px와 464px가 제공됐다. 제공되지 않은 고해상도 주소를 추측하지 않는다.
- 일반 HTTP 검색 요청은 실제로 자동화 접근 차단 화면을 반환했다. 봇은 Playwright 브라우저로 공개 페이지를 읽는다. 브라우저 사용도 접속 성공을 보장하지 않는다.
- **현재 `robots.txt`는 일반 봇에 `Disallow: /`를 적용한다.** 기본 실행은 이를 확인하고 중단한다. 사이트로부터 자동 수집 허가를 확보한 경우에만 `--site-permission`을 지정한다. 이 옵션은 접근제어를 해제하지 않는다. HTTP 401/403/429 또는 CAPTCHA는 즉시 중단하며, 프록시 회전·허용 봇 사칭·CAPTCHA 우회 기능은 없다.
- 코드 구현 및 로컬 검증이 완료된 상태이며, SSG 전체 이미지를 수집 완료한 상태는 아니다. 원본 이미지의 이용 조건은 수집·학습·재배포 목적에 맞게 별도로 확인한다.

## 설치 (Windows PowerShell, Python 3.10 이상)

아래 명령은 `SmartFactory_Vision_Project1` 폴더에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-crawler.txt
```

기본 브라우저는 Windows에 설치된 Microsoft Edge다. 별도 브라우저 다운로드가 필요 없다. Chrome은 `--browser chrome`, Playwright Chromium은 아래 설치 후 `--browser chromium`을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

이 작업 환경에는 `.venv`와 필요한 라이브러리를 설치해 두었다. `python`이 PATH에 없어도 `.\.venv\Scripts\python.exe`로 실행할 수 있다. 다른 PC로 `.venv`를 복사하지 말고 위 설치 명령을 사용한다.

## 실행

현재 수집 정책 확인 (상품/이미지 수집 없음):

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --check-policy
```

**SSG 자동 수집 허가를 확보한 경우** 먼저 1페이지의 5개 이미지로 확인한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --site-permission --max-pages 1 --max-images 5
```

검색 결과 마지막 페이지까지 수집/재개:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --site-permission
```

선택 옵션:

```powershell
# 수집 화면을 직접 확인
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --site-permission --headed --max-pages 1

# 이미지 다운로드 없이 공개 이미지 주소 목록만 기록
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --site-permission --list-only --max-pages 2

# 특정 페이지부터 다른 출력 폴더에 저장
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --site-permission --start-page 20 --output data/ssg_second_run
```

`--max-pages 0`, `--max-images 0`은 제한 없음이다. 이미지 한도는 **이번 실행에서 새로 처리한 상품 이미지 건수(중복 이미지 포함)**이며, 이미 저장된 항목은 차감하지 않는다. 페이지 한도는 이번 실행에서 방문한 페이지 수다. 기본 대기는 이미지 요청 간 1초, 페이지 사이 3초다. 10,000개 이미지라면 대기 시간만 약 3시간이므로 처음에는 한도를 지정한다.

`Ctrl+C`로 멈춘 뒤 같은 명령을 실행하면 미완료 페이지부터 재개한다. 다운로드 성공 항목은 건너뛰며, 실패·삭제된 파일은 다시 시도한다. 상품 순위 변동 때문에 이전 페이지에 새 상품이 추가되었을 수 있으므로 전체 목록을 새로 확인하려면 `--start-page 1`을 사용한다. 두 프로세스가 같은 출력 폴더를 동시에 사용하지 않도록 한다.

권한 확보 전에도 저장된 검색 HTML의 상품 주소를 오프라인으로 추출할 수 있다. 파일은 UTF-8이며 원래 상품 링크와 `src`/`srcset` URL이 있어야 한다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.data_collection.ssg_crawler --html-dir reports/saved_ssg_pages --output data/ssg_tumbler
```

오프라인 모드는 목록만 저장한다. 파일명 순서로 페이지를 부여하므로 `001.html`, `002.html`처럼 이름을 지정한다. 실제 다운로드는 이후 허가된 온라인 실행으로 진행한다.

## 산출물

기본 위치: `data/ssg_tumbler/` (Git 제외)

```text
data/ssg_tumbler/
├── raw/s01_MMDD_ssg/<상품ID>_<해시>.jpg   # 실제 형식에 따라 png/webp/gif 가능
├── manifest.csv                         # 상품명·상품ID·출처·원본 URL·크기·성공/실패
├── meta.csv                             # vision-harness용, 형태 라벨은 빈칸
├── crawl.sqlite3                        # 재개 체크포인트와 이미지 중복 인덱스
└── summary.json                         # 종료 이유·건수·정책 확인 결과
```

- 배송 채널 때문에 같은 상품이 여러 번 나와도 같은 상품/이미지 조합은 하나로 기록한다.
- 다른 상품 ID라도 픽셀 내용이 같은 이미지는 한 파일만 저장하고 출처는 `manifest.csv`에 모두 남긴다. 비슷하지만 다른 크롭·해상도 이미지는 별도다.
- `manifest.csv`의 `status`: `pending`, `downloaded`, `duplicate`, `failed`. 오류는 `error`에 기록한다.
- `summary.json`의 `search_complete: true`는 1페이지부터 마지막까지 수집하고, 미처리·실패가 없을 때만 표시한다. 한도 도달이나 중간 페이지부터 수집한 결과는 전체 완료가 아니다.
- CSV는 UTF-8 BOM 형식이다. Excel에서 열 수 있다. 재실행 시 기존 `meta.csv`에 사람이 입력한 라벨·그룹·조건·메모를 보존한다. 실행 중 CSV를 Excel에서 열어 파일을 잠그지 않는다.
- 종료 코드: `0` 정상/설정한 한도 도달, `1` 수집 또는 파일 오류, `2` 정책 확인/접근 허가 필요, `130` 사용자 중단. 전체 완료 여부는 반드시 `search_complete`로 확인한다.

## 텀블러 프로젝트 / vision-harness 연결

`meta.csv`는 기존 하네스의 `filename, session, group, label, count, value, cond_light, cond_bg, cond_angle, note` 규약을 따른다. 기본 `group=ssg_<상품ID>`, 촬영 조건은 관측할 수 없으므로 `unknown`, `label`은 빈칸이다.

검수 후 `straight`, `taper_smooth`, `taper_step`, `mug` 중 하나로 라벨을 붙인다. 검색 결과에는 세트·뚜껑·파우치·관련 상품도 포함될 수 있어 검색어만으로 텀블러/형태 정답을 자동 확정하지 않는다. 같은 실제 제품이 다른 판매자 ID로 올라온 경우 group을 합쳐 데이터 누수를 막는다. 같은 상품 ID도 여러 제품 옵션을 묶을 수 있으므로 그룹 기준을 사람이 확인한다.

`vision-harness/config.yaml`의 클래스를 위 네 가지로 맞춘 뒤, 하네스에서 `--data-dir ../SmartFactory_Vision_Project1/data/ssg_tumbler`를 사용하면 된다. 원 프로젝트의 **웹 데이터=train/val, 직접 촬영=test** 설계를 유지한다. 이 웹 수집분 전체에 무작위 train/val/test 분할을 적용하면 원래 의도한 도메인 이동 평가가 달라진다.

## 검증

외부 사이트에 요청하지 않는 자동 테스트:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_ssg_crawler.py -v
.\.venv\Scripts\python.exe -X utf8 tests/check_ssg_browser.py
```

첫 명령은 실제 관측 DOM을 축약한 fixture로 이미지 선택·아이콘 제외·페이지 경계·중복·체크포인트·라벨 보존·차단 중단을 검증한다. 두 번째는 로컬 HTTP 서버와 실제 Edge를 사용해 2페이지 순회, 이미지 저장, HTTP 503 재시도, 한도 중단, 재개, 중복 제거를 끝까지 확인한다. 검증용 이미지와 임시 폴더는 테스트 종료 시 정리된다.
