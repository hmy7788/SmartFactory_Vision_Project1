from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# 클래스별 검색 키워드
CLASS_QUERIES = {
    "straight":     ["원통형 텀블러", "직선 텀블러", "심플 텀블러 원통"],
    "taper_smooth": ["슬림 텀블러", "테이퍼 텀블러", "좁아지는 텀블러"],
    "taper_step":   ["스탠리 텀블러", "단차 텀블러", "숄더 텀블러"],
    "mug":          ["머그컵 텀블러", "손잡이 텀블러", "머그형 보온컵"],
}

TARGET = 300  # 클래스당 목표 장수
SAVE_ROOT = "data/train"

def search_naver(query, display=100, start=1):
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "start": start, "sort": "sim"}
    res = requests.get(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()
    return res.json()

def download_image(img_url, save_path):
    try:
        res = requests.get(img_url, timeout=5)
        if res.status_code == 200 and "image" in res.headers.get("Content-Type", ""):
            with open(save_path, "wb") as f:
                f.write(res.content)
            return True
    except Exception:
        pass
    return False

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# 폴더 생성
for cls in CLASS_QUERIES:
    os.makedirs(f"{SAVE_ROOT}/{cls}", exist_ok=True)

# 수집 시작
for cls, queries in CLASS_QUERIES.items():
    print(f"\n[{cls}] 수집 시작")
    save_dir = f"{SAVE_ROOT}/{cls}"
    seen_hashes = set()
    count = 0

    for query in queries:
        if count >= TARGET:
            break
        for start in range(1, 1001, 100):
            if count >= TARGET:
                break
            try:
                data = search_naver(query, display=100, start=start)
            except Exception as e:
                print(f"  API 오류: {e}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in tqdm(items, desc=f"  {query} (start={start})"):
                if count >= TARGET:
                    break
                img_url = item.get("image", "")
                if not img_url:
                    continue

                fname = f"{cls}_{count:04d}.jpg"
                fpath = f"{save_dir}/{fname}"

                if download_image(img_url, fpath):
                    h = get_hash(fpath)
                    if h in seen_hashes:
                        os.remove(fpath)  # 중복 제거
                    else:
                        seen_hashes.add(h)
                        count += 1

            time.sleep(0.2)

    print(f"[{cls}] 완료: {count}장")

print("\n✅ 전체 수집 완료!")