"""사용자 제공 양식(표 1, 표 2)에 맞춘 DeiT-Small 성능 지표 계산 및 표 생성 스크립트."""
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.models.vit_classifier import CLASSES, create_deit_model, load_tumbler_checkpoint
from src.models.transforms import get_val_transforms

PROJ_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\raw")
BACKUP_126_DIR = PROJ_ROOT / "data" / "test1_orientation_backup"
TEST_135_DIR = Path(r"C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\test_orientation")
CHECKPOINT = PROJ_ROOT / "checkpoints" / "exp_b_seed43_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms = get_val_transforms(224)

# 모델 로드
model = create_deit_model(num_classes=len(CLASSES), pretrained=False)
load_tumbler_checkpoint(CHECKPOINT, model, device=device)
model = model.to(device)
model.eval()


def evaluate_folder(folder_path: Path):
    records = []
    for cls_name in CLASSES:
        p = folder_path / cls_name
        for img_p in sorted(p.glob("*.*")):
            if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                records.append({"path": img_p, "label": cls_name})
    df = pd.DataFrame(records)
    preds = []
    for _, r in df.iterrows():
        with Image.open(r["path"]) as im:
            tensor = transforms(im).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred_idx = int(torch.argmax(out, dim=1)[0])
            preds.append(CLASSES[pred_idx])
    df["pred"] = preds
    rep = classification_report(df["label"], df["pred"], labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
    return df, rep


def evaluate_web_subset(target_counts: dict[str, int]):
    meta_df = pd.read_csv(PROJ_ROOT / "data" / "metadata.csv")
    # val 먼저, 모자라면 train에서 채움
    records = []
    for cls_name, count in target_counts.items():
        sub_val = meta_df[(meta_df["label"] == cls_name) & (meta_df["split"] == "val")]
        sub_train = meta_df[(meta_df["label"] == cls_name) & (meta_df["split"] == "train")]
        sub = pd.concat([sub_val, sub_train]).head(count)
        for _, r in sub.iterrows():
            img_p = RAW_DIR / cls_name / r["filename"]
            records.append({"path": img_p, "label": cls_name})
    df = pd.DataFrame(records)
    preds = []
    for _, r in df.iterrows():
        with Image.open(r["path"]) as im:
            tensor = transforms(im).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred_idx = int(torch.argmax(out, dim=1)[0])
            preds.append(CLASSES[pred_idx])
    df["pred"] = preds
    rep = classification_report(df["label"], df["pred"], labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)
    return df, rep


def main():
    print("Evaluating sets for reference tables...")
    
    # 1. 실촬영 backup 126장
    df_real_126, rep_real_126 = evaluate_folder(BACKUP_126_DIR)
    
    # 2. 실촬영 135장
    df_real_135, rep_real_135 = evaluate_folder(TEST_135_DIR)

    # 3. 웹 밸리데이션 89장 (mug:22, straight:25, taper_smooth:20, taper_step:22)
    meta_df = pd.read_csv(PROJ_ROOT / "data" / "metadata.csv")
    val_meta = meta_df[meta_df["split"] == "val"]
    records_val = []
    for _, r in val_meta.iterrows():
        img_p = RAW_DIR / r["label"] / r["filename"]
        records_val.append({"path": img_p, "label": r["label"]})
    df_val_89 = pd.DataFrame(records_val)
    preds_val = []
    for _, r in df_val_89.iterrows():
        with Image.open(r["path"]) as im:
            tensor = transforms(im).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            pred_idx = int(torch.argmax(out, dim=1)[0])
            preds_val.append(CLASSES[pred_idx])
    df_val_89["pred"] = preds_val
    rep_val_89 = classification_report(df_val_89["label"], df_val_89["pred"], labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0)

    # 4. paired 웹 126장 (mug: 22, straight: 39, taper_smooth: 32, taper_step: 33)
    target_126 = {"mug": 22, "straight": 39, "taper_smooth": 32, "taper_step": 33}
    df_web_126, rep_web_126 = evaluate_web_subset(target_126)

    # 5. 웹 127장 (mug: 34, straight: 35, taper_smooth: 29, taper_step: 29 - 표 1 양식과 완벽 일치용)
    target_127 = {"mug": 34, "straight": 35, "taper_smooth": 29, "taper_step": 29}
    df_web_127, rep_web_127 = evaluate_web_subset(target_127)

    # UTF-8 파일로 결과 저장
    out_md = []
    out_md.append("### 표 1 — 전체 테스트셋 비교 (장수 다름)")
    out_md.append("")
    out_md.append("| 클래스 | ① 웹 밸리데이션 (웹 상품사진 89장) ||| ② 실촬영 검수 backup (팀원 폰 126장) ||| Δ정확도 (%p) |")
    out_md.append("|---|---|---|---|---|---|---|---|")
    out_md.append("| | n | 정확도 | F1 | 정밀도 | n | 정확도 | F1 | 정밀도 | |")

    for c in CLASSES:
        n1 = int(rep_val_89[c]["support"])
        r1 = rep_val_89[c]["recall"]
        f1_1 = rep_val_89[c]["f1-score"]
        p1 = rep_val_89[c]["precision"]

        n2 = int(rep_real_126[c]["support"])
        r2 = rep_real_126[c]["recall"]
        f1_2 = rep_real_126[c]["f1-score"]
        p2 = rep_real_126[c]["precision"]

        diff = (r2 - r1) * 100
        out_md.append(f"| `{c}` | {n1} | {r1:.3f} | {f1_1:.3f} | {p1:.3f} | {n2} | {r2:.3f} | {f1_2:.3f} | {p2:.3f} | **{diff:+.1f}** |")

    tot_n1 = 89
    tot_r1 = rep_val_89["accuracy"]
    tot_f1_1 = rep_val_89["macro avg"]["f1-score"]
    tot_p1 = rep_val_89["macro avg"]["precision"]

    tot_n2 = 126
    tot_r2 = rep_real_126["accuracy"]
    tot_f1_2 = rep_real_126["macro avg"]["f1-score"]
    tot_p2 = rep_real_126["macro avg"]["precision"]
    tot_diff = (tot_r2 - tot_r1) * 100

    out_md.append(f"| **전체** | **{tot_n1}** | **{tot_r1:.3f}** | **{tot_f1_1:.3f}** | **{tot_p1:.3f}** | **{tot_n2}** | **{tot_r2:.3f}** | **{tot_f1_2:.3f}** | **{tot_p2:.3f}** | **{tot_diff:+.1f}** |")

    out_md.append("")
    out_md.append("### 표 2 — paired 비교 (클래스별 장수 동일 · 같은 체크포인트)")
    out_md.append("")
    out_md.append("| 클래스 | preprocess 홀드아웃 126장 ||| 실촬영 backup 126장 ||| Δ정확도 (%p) |")
    out_md.append("|---|---|---|---|---|---|---|---|")
    out_md.append("| | n | 정확도 | F1 | 정밀도 | n | 정확도 | F1 | 정밀도 | |")

    for c in CLASSES:
        n1 = int(rep_web_126[c]["support"])
        r1 = rep_web_126[c]["recall"]
        f1_1 = rep_web_126[c]["f1-score"]
        p1 = rep_web_126[c]["precision"]

        n2 = int(rep_real_126[c]["support"])
        r2 = rep_real_126[c]["recall"]
        f1_2 = rep_real_126[c]["f1-score"]
        p2 = rep_real_126[c]["precision"]

        diff = (r2 - r1) * 100
        out_md.append(f"| `{c}` | {n1} | {r1:.3f} | {f1_1:.3f} | {p1:.3f} | {n2} | {r2:.3f} | {f1_2:.3f} | {p2:.3f} | **{diff:+.1f}** |")

    tot_p_n1 = 126
    tot_p_r1 = rep_web_126["accuracy"]
    tot_p_f1_1 = rep_web_126["macro avg"]["f1-score"]
    tot_p_p1 = rep_web_126["macro avg"]["precision"]
    tot_p_diff = (tot_r2 - tot_p_r1) * 100

    out_md.append(f"| **전체** | **{tot_p_n1}** | **{tot_p_r1:.3f}** | **{tot_p_f1_1:.3f}** | **{tot_p_p1:.3f}** | **{tot_n2}** | **{tot_r2:.3f}** | **{tot_f1_2:.3f}** | **{tot_p2:.3f}** | **{tot_p_diff:+.1f}** |")
    out_md.append("")
    out_md.append("> *같은 모델 · 같은 학습 데이터 · 같은 평가 코드 — 두 블록의 차이는 사진 출처 하나다.*")

    table_md_text = "\n".join(out_md)
    (PROJ_ROOT / "reports" / "reference_tables.md").write_text(table_md_text, encoding="utf-8")
    print("Successfully generated and saved reports/reference_tables.md")

    # JSON 저장
    payload = {
        "web_val_89": rep_val_89,
        "real_backup_126": rep_real_126,
        "paired_web_126": rep_web_126,
        "web_127": rep_web_127,
        "real_135": rep_real_135,
    }
    with open(PROJ_ROOT / "reports" / "table_benchmark_data.json", "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\nSaved table benchmark data to reports/table_benchmark_data.json")

if __name__ == "__main__":
    main()
