"""붙인 모델이 제대로 붙었는지 확인한다 — 라벨된 사진 폴더로 정확도를 재고, 팀원 inference.py와 같은 답을 내는지 대조한다.

    python tools\\check_model.py                         # 저장소 옆에서 라벨 폴더를 찾아 씀 (예: ..\\검수_test)
    python tools\\check_model.py --folder D:\\photos\\test  # 폴더 지정
    python tools\\check_model.py --compare               # src/models/convnext/inference.py 결과와 대조

라벨 폴더 구조:  <folder>/<클래스이름>/*.jpg   (클래스이름 = meta.json의 classes)

왜 필요한가 — meta.json의 resize/mean/std/클래스 순서가 학습과 다르면 에러 없이 조용히 틀린다.
정확도가 학습한 사람이 말한 값 근처면 붙은 것이고, --compare에서 라벨 일치 100%·확률 차이 ≈0이면
대시보드가 팀원 코드와 수학적으로 같은 모델을 돌리고 있다는 뜻이다.
"""
import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import sources                      # noqa: E402
from src.pipeline import Pipeline            # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")  # cmd 창 코드페이지에 없는 글자가 있어도 죽지 않게
except AttributeError:
    pass


def find_labeled_folder(classes: list[str]) -> Path | None:
    """저장소 옆(부모 폴더)에서 클래스 이름 하위폴더를 전부 가진 폴더를 찾는다.

    후보가 여럿이면 사진이 제일 많은 폴더. 백업본(test1_orientation_backup 같은)은 보통 원본의 부분집합이라
    이름으로 고르면 엉뚱한 걸 집는다 — 장수로 고르면 현재 쓰는 세트가 잡힌다.
    """
    cands = [c for c in sorted(ROOT.parent.iterdir())
             if c.is_dir() and c != ROOT and all((c / k).is_dir() for k in classes)]
    if not cands:
        return None
    return max(cands, key=lambda c: sum(len(sources.list_images(c / k)) for k in classes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model"))
    ap.add_argument("--folder", default=None, help="라벨된 사진 폴더 (클래스별 하위폴더). 없으면 저장소 옆에서 찾음")
    ap.add_argument("--compare", action="store_true", help="src/models/convnext/inference.py와 같은 답인지 대조")
    ap.add_argument("--limit", type=int, default=0, help="클래스당 최대 장수 (0=전부)")
    ap.add_argument("--threshold", type=float, default=0.70, help="대시보드 판정 기준 확신도 (보류 비율 계산용)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    pipe = Pipeline(args.checkpoint)
    info = pipe.info()
    print(f"모델: {info['arch']} · 입력 {info['input']}px · {info['resize']} · {info['weights']} "
          f"({info['size_mb']} MB) · {info['device']} · 로드 {time.perf_counter() - t0:.1f}s")
    print(f"클래스 순서: {pipe.classes}")

    folder = Path(args.folder) if args.folder else find_labeled_folder(pipe.classes)
    if folder is None or not folder.is_dir():
        print("\n라벨 폴더를 못 찾았다. --folder <폴더> 로 지정해라. 구조: <폴더>/<클래스이름>/*.jpg")
        return 2
    print(f"사진 폴더: {folder}")

    extra = [d.name for d in folder.iterdir() if d.is_dir() and d.name not in pipe.classes]
    if extra:
        print(f"  (클래스에 없는 하위폴더는 건너뜀: {extra})")

    clf = None
    if args.compare:
        from src.models.convnext.inference import TumblerClassifier
        clf = TumblerClassifier(str(Path(args.checkpoint) / pipe.meta["weights"]), device=info["device"])

    n_total = n_correct = n_undecided = 0
    per_class = defaultdict(lambda: [0, 0])          # 클래스 → [장수, 맞은 수]
    confusion = Counter()                            # (정답, 예측) → 수
    wrong, ms, agree, max_diff, sum_diff, n_cmp = [], [], 0, 0.0, 0.0, 0

    for true in pipe.classes:
        cdir = folder / true
        if not cdir.is_dir():
            print(f"  {true}: 폴더 없음 — 건너뜀")
            continue
        names = sources.list_images(cdir)
        if args.limit:
            names = names[:args.limit]
        for name in names:
            image = sources.read_image(cdir / name)
            pred = pipe.predict(image)
            n_total += 1
            per_class[true][0] += 1
            confusion[(true, pred["label"])] += 1
            ms.append(pred["infer_ms"])
            if pred["label"] == true:
                n_correct += 1
                per_class[true][1] += 1
            else:
                wrong.append((f"{true}/{name}", pred["label"], pred["score"]))
            if pred["score"] < args.threshold:
                n_undecided += 1
            if clf is not None:
                # 같은 픽셀(BGR→RGB 배열)을 넣어 전처리+모델만 비교한다. 디코더 차이는 여기서 뺀다.
                ref = clf.predict(image[:, :, ::-1].copy())
                n_cmp += 1
                agree += int(ref["class"] == pred["label"])
                for c in pipe.classes:
                    d = abs(ref["probabilities"][c] - pred["probs"][c])
                    max_diff = max(max_diff, d)
                    sum_diff += d
        print(f"  {true}: {per_class[true][1]}/{per_class[true][0]}")

    if n_total == 0:
        print("\n사진이 한 장도 없다.")
        return 2

    print(f"\n정확도 {n_correct}/{n_total} = {n_correct / n_total:.3f}")
    print(f"확신도 < {args.threshold:.2f} (대시보드에서 '판정 보류'): {n_undecided}/{n_total} = {n_undecided / n_total:.3f}")
    print(f"추론 속도: 평균 {sum(ms) / len(ms):.0f} ms/장 ({info['device']})")

    w = max(len(c) for c in pipe.classes) + 2
    print("\n혼동행렬 (행=정답, 열=예측)")
    print(" " * w + "".join(f"{c:>{w}}" for c in pipe.classes))
    for t in pipe.classes:
        print(f"{t:>{w}}" + "".join(f"{confusion[(t, p)]:>{w}}" for p in pipe.classes))

    if wrong:
        print(f"\n틀린 사진 {len(wrong)}장 (최대 20장 표시):")
        for path, label, score in wrong[:20]:
            print(f"  {path}  ->  {label} ({score:.2f})")

    if clf is not None:
        print(f"\n[--compare] 팀원 inference.py 대조: 라벨 일치 {agree}/{n_cmp}, "
              f"확률 차이 최대 {max_diff:.6f} · 평균 {sum_diff / (n_cmp * len(pipe.classes)):.6f}")
        if agree == n_cmp and max_diff < 1e-3:
            print("  => 같은 모델·같은 전처리다. 대시보드 = 팀원 코드.")
        else:
            print("  => 다르다. meta.json의 resize/mean/std/classes를 inference.py의 TRANSFORM·CLASS_NAMES와 다시 맞춰라.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
