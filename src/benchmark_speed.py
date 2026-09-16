"""텀블러 형태 분류 파이프라인 단계별 연산 시간(Latency & FPS) 벤치마크 스크립트."""
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.models.vit_classifier import create_deit_model, load_tumbler_checkpoint, CLASSES
from src.models.transforms import get_val_transforms
from src.cv_utils.segmentation import extract_tumbler_mask, imread_unicode


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "exp_b_seed44_best.pt"
    model = create_deit_model(num_classes=len(CLASSES), pretrained=False)
    load_tumbler_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)
    model.eval()

    transforms = get_val_transforms(224)

    # 1. 순수 GPU 전방 추론 시간 (Pure GPU Forward Pass)
    dummy_tensor = torch.randn(1, 3, 224, 224, device=device)
    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_gpu = []
    for _ in range(200):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy_tensor)
            _ = torch.softmax(out, dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_gpu.append((time.perf_counter() - t0) * 1000)

    gpu_mean = float(np.mean(times_gpu))
    gpu_std = float(np.std(times_gpu))
    gpu_fps = 1000.0 / gpu_mean

    # 2. 웹 수집 이미지 (~500x500) End-to-End 시간
    web_img_path = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\raw\straight\1.jpg")
    times_web_prep = []
    times_web_e2e = []
    for _ in range(50):
        t0 = time.perf_counter()
        with Image.open(web_img_path) as im:
            tensor = transforms(im).unsqueeze(0).to(device)
        t1 = time.perf_counter()
        times_web_prep.append((t1 - t0) * 1000)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            out = model(tensor)
            _ = torch.softmax(out, dim=1)[0].cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        times_web_e2e.append((t2 - t0) * 1000)

    web_prep_mean = float(np.mean(times_web_prep))
    web_e2e_mean = float(np.mean(times_web_e2e))
    web_fps = 1000.0 / web_e2e_mean

    # 3. 실촬영 스마트폰 고해상도(4000x3000, 1200만 화소) 사진 End-to-End 시간
    real_img_path = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\test_orientation\straight\KakaoTalk_20260915_114226226_27.jpg")
    times_real_io_prep = []
    times_real_e2e = []
    for _ in range(30):
        t0 = time.perf_counter()
        with Image.open(real_img_path) as im:
            tensor = transforms(im).unsqueeze(0).to(device)
        t1 = time.perf_counter()
        times_real_io_prep.append((t1 - t0) * 1000)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            out = model(tensor)
            _ = torch.softmax(out, dim=1)[0].cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        times_real_e2e.append((t2 - t0) * 1000)

    real_prep_mean = float(np.mean(times_real_io_prep))
    real_e2e_mean = float(np.mean(times_real_e2e))
    real_fps = 1000.0 / real_e2e_mean

    # 4. 배경 분리 (누끼 / GrabCut 448px) 연산 시간
    bgr = imread_unicode(str(real_img_path))
    times_nukki = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = extract_tumbler_mask(bgr)
        times_nukki.append((time.perf_counter() - t0) * 1000)

    nukki_mean = float(np.mean(times_nukki))

    # 결과 출력
    print("====================================================================")
    print(f" 연산 시간 벤치마크 결과 (디바이스: {gpu_name})")
    print("====================================================================")
    print(f"1. 순수 ViT 모델 GPU 추론 (1장): {gpu_mean:.2f} ms ± {gpu_std:.2f} ms (처리량: {gpu_fps:.1f} FPS)")
    print("--------------------------------------------------------------------")
    print(f"2. 웹 이미지(~500px) End-to-End: {web_e2e_mean:.2f} ms (처리량: {web_fps:.1f} FPS)")
    print(f"   - 디스크 로드 + 224패딩 전처리: {web_prep_mean:.2f} ms")
    print(f"   - GPU 모델 추론 + Softmax: {web_e2e_mean - web_prep_mean:.2f} ms")
    print("--------------------------------------------------------------------")
    print(f"3. 실촬영 1200만화소(4000x3000) End-to-End: {real_e2e_mean:.2f} ms (처리량: {real_fps:.1f} FPS)")
    print(f"   - 1200만화소 디코딩 + EXIF회전 + 고화질 리사이즈: {real_prep_mean:.2f} ms")
    print(f"   - GPU 모델 추론 + Softmax: {real_e2e_mean - real_prep_mean:.2f} ms")
    print("--------------------------------------------------------------------")
    print(f"4. 실시간 누끼따기 (GrabCut 448px 다운스케일): {nukki_mean:.2f} ms ({nukki_mean/1000:.2f} 초)")
    print("--------------------------------------------------------------------")
    print(f"5. 실촬영 135장 전체 일괄 평가 소요 시간 (3개 모델 앙상블 405회 추론):")
    print(f"   - 135장 기준 총 소요 시간: 약 {real_prep_mean * 135 / 1000 + (gpu_mean * 135 * 3) / 1000:.1f} 초")
    print("====================================================================")

    # JSON 리포트 저장
    bench_data = {
        "device": gpu_name,
        "pure_gpu_forward_ms": gpu_mean,
        "pure_gpu_fps": gpu_fps,
        "web_e2e_ms": web_e2e_mean,
        "web_prep_ms": web_prep_mean,
        "web_fps": web_fps,
        "real_4000x3000_e2e_ms": real_e2e_mean,
        "real_4000x3000_prep_ms": real_prep_mean,
        "real_fps": real_fps,
        "nukki_grabcut_ms": nukki_mean,
    }
    with open(Path(__file__).resolve().parent.parent / "reports" / "latency_benchmark.json", "w", encoding="utf-8") as f:
        import json
        json.dump(bench_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
