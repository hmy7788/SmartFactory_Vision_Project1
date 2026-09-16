"""DeiT-Small/16 모델 정의, 가중치 동결/해제, 파라미터 그룹화 및 체크포인트 유틸리티."""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import timm

CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]
DEFAULT_MODEL_NAME = "deit_small_patch16_224.fb_in1k"


def create_deit_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = len(CLASSES),
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """timm의 DeiT-Small/16 사전학습 모델을 로드하고 최종 분류 헤드를 4개 클래스로 교체합니다."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def freeze_backbone_for_linear_probe(model: nn.Module) -> None:
    """실험 A: 특징 추출기(Backbone) 전체를 동결하고 최종 분류 헤드(head)만 학습 가능하게 설정합니다."""
    for param in model.parameters():
        param.requires_grad = False

    # DeiT / ViT 헤드만 활성화
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_last_blocks_for_fine_tuning(
    model: nn.Module,
    num_blocks: int = 4,
) -> None:
    """실험 B: 마지막 num_blocks(기본 4개) Transformer 블록, 최종 정규화층(norm), 분류 헤드(head)만 활성화합니다."""
    # 1. 전체 동결
    for param in model.parameters():
        param.requires_grad = False

    # 2. 마지막 4개 블록 활성화 (12개 블록 중 8, 9, 10, 11)
    if hasattr(model, "blocks"):
        for block in model.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    # 3. 최종 LayerNorm 활성화
    if hasattr(model, "norm") and model.norm is not None:
        for param in model.norm.parameters():
            param.requires_grad = True

    # 4. 헤드 활성화
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def get_optimizer_param_groups(
    model: nn.Module,
    lr_blocks: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 0.01,
) -> list[dict]:
    """실험 B용 차등 학습률(Differential Learning Rate) 파라미터 그룹을 생성합니다."""
    head_params = []
    block_params = []
    head_param_ids = set(id(p) for p in model.get_classifier().parameters())

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            block_params.append(p)

    param_groups = []
    if block_params:
        param_groups.append({
            "params": block_params,
            "lr": lr_blocks,
            "weight_decay": weight_decay,
            "name": "blocks_and_norm",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "name": "classifier_head",
        })
    return param_groups


def save_tumbler_checkpoint(
    path: Path | str,
    model: nn.Module,
    epoch: int,
    val_macro_f1: float,
    val_loss: float,
    train_args: dict,
    classes: list[str] = CLASSES,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """재현성과 하네스 연동을 위한 상세 메타데이터가 포함된 체크포인트를 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "classes": classes,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_macro_f1": float(val_macro_f1),
        "val_loss": float(val_loss),
        "train_args": train_args,
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "neutral_pad_color": [128, 128, 128],
    }
    torch.save(payload, str(path))
    print(f"Saved checkpoint: {path} (Epoch {epoch}, Val Macro F1: {val_macro_f1:.4f}, Val Loss: {val_loss:.4f})")


def load_tumbler_checkpoint(path: Path | str, model: nn.Module, device: torch.device | str = "cpu") -> dict:
    """체크포인트를 로드하고 model에 state_dict를 주입합니다."""
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt
