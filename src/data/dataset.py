"""Tumbler PyTorch Dataset 및 DataLoader 구축 모듈."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


class TumblerDataset(Dataset):
    """메타데이터 CSV 및 split 필터 기반 텀블러 데이터셋."""

    def __init__(
        self,
        metadata_df: pd.DataFrame | Path | str,
        split: str = "train",
        transform=None,
    ):
        if isinstance(metadata_df, (str, Path)):
            metadata_df = pd.read_csv(metadata_df)

        self.split = split
        self.transform = transform
        self.df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split={split!r}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = Path(row["full_path"])
        label_str = row["label"]
        label_idx = CLASS_TO_IDX[label_str]
        image_id = row["image_id"]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = img

        return img_tensor, label_idx, image_id


def create_dataloaders(
    metadata_csv: Path | str,
    train_transform,
    val_transform,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train 및 Val DataLoader를 생성합니다."""
    df = pd.read_csv(metadata_csv)
    train_ds = TumblerDataset(df, split="train", transform=train_transform)
    val_ds = TumblerDataset(df, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader
