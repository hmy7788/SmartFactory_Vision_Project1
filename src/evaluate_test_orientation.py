"""실촬영 test_orientation 135장 테스트셋에 대한 텀블러 4종 형태 분류 평가 스크립트."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.models.vit_classifier import CLASSES, create_deit_model, load_tumbler_checkpoint
from src.models.transforms import get_val_transforms

DATA_DIR = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\test_orientation")
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 테스트셋 이미지 수집
    records = []
    for label in CLASSES:
        cls_dir = DATA_DIR / label
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.glob("*.*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                records.append({
                    "filename": img_path.name,
                    "rel_path": f"{label}/{img_path.name}",
                    "full_path": str(img_path),
                    "label": label,
                })

    df = pd.DataFrame(records)
    print(f"Total test images found: {len(df)}")
    print(df["label"].value_counts())

    # 2. 전처리 파이프라인
    val_transforms = get_val_transforms(target_size=224)

    # 3. 모델 로드 (Seed 44, Seed 42, Seed 43)
    ckpt_paths = {
        "exp_b_seed44": CHECKPOINTS_DIR / "exp_b_seed44_best.pt",
        "exp_b_seed42": CHECKPOINTS_DIR / "exp_b_seed42_best.pt",
        "exp_b_seed43": CHECKPOINTS_DIR / "exp_b_seed43_best.pt",
    }

    models = {}
    for key, path in ckpt_paths.items():
        if path.is_file():
            model = create_deit_model(num_classes=len(CLASSES), pretrained=False)
            load_tumbler_checkpoint(path, model, device=device)
            model = model.to(device)
            model.eval()
            models[key] = model
            print(f"Loaded {key} from {path.name}")

    if not models:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINTS_DIR}")

    # 4. 추론 수행
    all_preds = {k: [] for k in models}
    all_probs = {k: [] for k in models}

    for idx, row in df.iterrows():
        img_path = Path(row["full_path"])
        with Image.open(img_path) as im:
            tensor = val_transforms(im).unsqueeze(0).to(device)

        with torch.no_grad():
            for k, model in models.items():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                all_preds[k].append(CLASSES[pred_idx])
                all_probs[k].append(probs)

    # DataFrame에 결과 추가
    for k in models:
        df[f"pred_{k}"] = all_preds[k]
        df[f"score_{k}"] = [float(p[CLASSES.index(pred)]) for p, pred in zip(all_probs[k], all_preds[k])]
        for i, c in enumerate(CLASSES):
            df[f"prob_{k}_{c}"] = [float(p[i]) for p in all_probs[k]]

    # 앙상블 (소프트 보팅)
    ensemble_probs = np.mean([np.array(all_probs[k]) for k in models], axis=0)
    df["pred_ensemble"] = [CLASSES[int(np.argmax(p))] for p in ensemble_probs]
    df["score_ensemble"] = [float(np.max(p)) for p in ensemble_probs]

    # 5. 성능 평가 및 리포트 작성
    y_true = df["label"].tolist()

    eval_results = {}
    for k in list(models.keys()) + ["ensemble"]:
        y_pred = df[f"pred_{k}"].tolist()
        macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        eval_results[k] = {
            "macro_f1": float(macro_f1),
            "accuracy": float(acc),
            "report": report,
            "cm": cm.tolist(),
        }
        print(f"\n[{k}] Macro F1: {macro_f1:.4f} ({macro_f1*100:.2f}%), Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # 6. 혼동 행렬 시각화 (Primary: Seed 44 및 Ensemble)
    primary_key = "exp_b_seed44" if "exp_b_seed44" in models else list(models.keys())[0]
    cm_primary = np.array(eval_results[primary_key]["cm"])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_primary,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
    )
    plt.title(f"Test Orientation Confusion Matrix ({primary_key})\nMacro F1: {eval_results[primary_key]['macro_f1']:.4f} (Acc: {eval_results[primary_key]['accuracy']:.4f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "test_orientation_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # 7. CSV 및 상세 Markdown 리포트 저장
    pred_csv_path = REPORTS_DIR / "test_orientation_predictions.csv"
    df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions CSV: {pred_csv_path}")

    # 오분류 목록
    mis_df = df[df["label"] != df[f"pred_{primary_key}"]]

    report_md = f"""# 실촬영 테스트셋(`test_orientation`) 4종 형태 분류 최종 평가 보고서

- **테스트 데이터 위치**: `{DATA_DIR}`
- **총 테스트 사진 수**: {len(df)} 장
- **평가 모델**: `{primary_key}` (`deit_small_patch16_224.fb_in1k`)
- **주요 평가 지표 (Macro F1)**: **`{eval_results[primary_key]['macro_f1']:.4f}` ({eval_results[primary_key]['macro_f1']*100:.2f}%)**
- **보조 평가 지표 (Accuracy)**: **`{eval_results[primary_key]['accuracy']:.4f}` ({eval_results[primary_key]['accuracy']*100:.2f}%)**

## 1. 모델별 성능 비교 (다중 시드 및 앙상블)

| 모델 / 가중치 | Macro F1 | Accuracy | 오분류 수 |
|---|---|---|---|
"""
    for k, res in eval_results.items():
        n_err = int(np.sum(df["label"] != df[f"pred_{k}"]))
        report_md += f"| `{k}` | **`{res['macro_f1']:.4f}`** ({res['macro_f1']*100:.2f}%) | `{res['accuracy']:.4f}` ({res['accuracy']*100:.2f}%) | {n_err}장 / {len(df)}장 |\n"

    report_md += f"""
## 2. 클래스별 상세 성능 (`{primary_key}`)

| 클래스 | Precision | Recall | F1-Score | Support (사진 수) |
|---|---|---|---|---|
"""
    for cls_name in CLASSES:
        c_rep = eval_results[primary_key]["report"][cls_name]
        report_md += f"| `{cls_name}` | {c_rep['precision']:.4f} | {c_rep['recall']:.4f} | **{c_rep['f1-score']:.4f}** | {int(c_rep['support'])} |\n"

    report_md += f"""
## 3. 오분류 사례 상세 분석 (총 {len(mis_df)}건)

| 파일명 | 정답 라벨 | 예측 라벨 | 예측 확신도 | 원본 라벨 확률 |
|---|---|---|---|---|
"""
    for _, r in mis_df.iterrows():
        p_score = r[f"score_{primary_key}"]
        true_prob = r[f"prob_{primary_key}_{r['label']}"]
        report_md += f"| `{r['filename']}` | `{r['label']}` | `{r[f'pred_{primary_key}']}` | {p_score:.4f} | {true_prob:.4f} |\n"

    report_md_path = REPORTS_DIR / "test_orientation_report.md"
    report_md_path.write_text(report_md, encoding="utf-8")
    print(f"Saved evaluation report: {report_md_path}")

    # JSON 저장
    with open(REPORTS_DIR / "test_orientation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
