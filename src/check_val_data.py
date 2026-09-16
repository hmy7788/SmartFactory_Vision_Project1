import pandas as pd
from pathlib import Path

df = pd.read_csv('data/metadata.csv')
val_df = df[df['split'] == 'val']
print('val_df rows:', len(val_df))

dataset_raw_ok = 0
for _, r in val_df.iterrows():
    p = Path(r'C:\Users\한국전파진흥협회\Desktop\Vision Proj\data set\raw') / r['label'] / r['filename']
    if p.is_file():
        dataset_raw_ok += 1

print(f'Exists in data set/raw: {dataset_raw_ok} / {len(val_df)}')

harness_ok = 0
for _, r in val_df.iterrows():
    p = Path(r'..\vision-harness\data\raw\web_val') / f"{r['label']}_{r['filename']}"
    if p.is_file():
        harness_ok += 1
print(f'Exists in vision-harness/data/raw/web_val: {harness_ok} / {len(val_df)}')
