from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import requests
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import inference_model, init_model


def _patch_efficient_attention_for_contiguous() -> None:
    """Wrap EfficientMultiheadAttention.forward so transposed tensors are
    contiguous before they hit nn.MultiheadAttention.

    Symptom: backward() raises
       "RuntimeError: view size is not compatible with input tensor's size
        and stride. Use .reshape(...) instead."
    Cause: x.transpose(0, 1) yields a non-contiguous tensor; PyTorch's
    MultiheadAttention internally calls .view() on it, which fails on
    backward graph reconstruction. Adding .contiguous() after transpose
    is the clean fix without editing the library.
    """
    from mmseg.models.backbones import mit as _mit

    orig = _mit.EfficientMultiheadAttention.forward

    def patched_forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            from mmseg.models.utils import nchw_to_nlc, nlc_to_nchw
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        if self.batch_first:
            x_q = x_q.transpose(0, 1).contiguous()
            x_kv = x_kv.transpose(0, 1).contiguous()

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1).contiguous()

        return identity + self.dropout_layer(self.proj_drop(out))

    _mit.EfficientMultiheadAttention.forward = patched_forward
    del orig


_patch_efficient_attention_for_contiguous()


def _patch_segformer_head_for_contiguous() -> None:
    """torch.cat([upsampled, ...], dim=1) даёт тензор с не-contiguous-strides
    (источники приходят из bilinear-resize). Когда он попадает в BatchNorm
    декода, BN.backward внутри C++ делает .view() и падает на стенке
    'view size is not compatible with input tensor stride'. Добавляем
    .contiguous() после cat.
    """
    from mmseg.models.decode_heads import segformer_head as _shead
    from mmseg.models.utils import resize as _resize

    def patched_forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            # Backbone выдаёт тензор не-contiguous (после nlc_to_nchw +
            # transpose в attention блоках). BatchNorm в conv(x).backward
            # падает на view → делаем contiguous здесь и после resize.
            x = inputs[idx].contiguous()
            conv = self.convs[idx]
            up = _resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            outs.append(up.contiguous())
        out = self.fusion_conv(torch.cat(outs, dim=1).contiguous())
        out = self.cls_seg(out)
        return out

    _shead.SegformerHead.forward = patched_forward


_patch_segformer_head_for_contiguous()


def _patch_accuracy_for_safe_indexing() -> None:
    """В mmseg `accuracy()` делает `correct[:, target != ignore_index]` по
    3D-бул-маске. На CPU/MPS PyTorch 2.11 это падает с
       "index 255 is out of bounds: 0, range 0 to 5"
    Переписываем через flatten + 1D-маску — тот же результат, но без
    advanced bool-indexing по нескольким измерениям.
    """
    from mmseg.models.losses import accuracy as _acc_module

    def safe_accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
        if isinstance(topk, int):
            topk = (topk,)
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.size(0) == 0:
            accu = [pred.new_tensor(0.0) for _ in range(len(topk))]
            return accu[0] if return_single else accu

        # На MPS PyTorch 2.11 любая bool-маска (включая `target != 255`) и
        # последующие операции `.eq() + .expand_as()` ломаются с
        # "index 255 is out of bounds". Acc_seg — просто метрика для
        # логирования, считаем целиком на CPU.
        device_orig = pred.device
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()

        assert pred_cpu.ndim == target_cpu.ndim + 1
        pred_value, pred_label = pred_cpu.topk(maxk, dim=1)
        pred_label = pred_label.transpose(0, 1)
        correct = pred_label.eq(target_cpu.unsqueeze(0).expand_as(pred_label))
        if thresh is not None:
            correct = correct & (pred_value > thresh).t()

        if ignore_index is not None:
            correct_flat = correct.reshape(correct.shape[0], -1)
            valid_mask = (target_cpu != ignore_index).reshape(-1)
            correct = correct_flat[:, valid_mask]
            valid_total = int(valid_mask.sum().item())
        else:
            valid_total = target_cpu.numel()

        eps = torch.finfo(torch.float32).eps
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
            res.append(correct_k.mul_(100.0 / (valid_total + eps)).to(device_orig))
        return res[0] if return_single else res

    _acc_module.accuracy = safe_accuracy

    from mmseg.models.decode_heads import decode_head as _dh
    _dh.accuracy = safe_accuracy


_patch_accuracy_for_safe_indexing()


# Anomaly detection даёт развёрнутый stack trace на ошибки в backward —
# использовалась для отладки view-stride багов. Включать только при отладке,
# в обычной тренировке тормозит в ~5x.
# torch.autograd.set_detect_anomaly(True)


def _patch_cross_entropy_for_mps_class_weights() -> None:
    """`F.cross_entropy(pred, target, weight=cw, ignore_index=255)` падает на
    MPS PyTorch 2.11 с "index 255 out of bounds: range 0 to 5", когда target
    содержит значения >= num_classes (паддинг 255). Workaround:
    1. заменить ignore-позиции в label на валидный класс (0)
    2. вызвать F.cross_entropy без ignore_index
    3. обнулить loss на исходных ignore-позициях
    Проксируется только если class_weight задан И device=mps. На CUDA исходное
    поведение сохраняется.
    """
    import torch.nn.functional as F
    from mmseg.models.losses import cross_entropy_loss as _cel

    orig_cross_entropy = _cel.cross_entropy

    def safe_cross_entropy(pred, label, weight=None, class_weight=None,
                           reduction="mean", avg_factor=None,
                           ignore_index=-100, avg_non_ignore=False):
        if class_weight is None or pred.device.type != "mps":
            return orig_cross_entropy(
                pred, label, weight=weight, class_weight=class_weight,
                reduction=reduction, avg_factor=avg_factor,
                ignore_index=ignore_index, avg_non_ignore=avg_non_ignore,
            )

        ignore_mask = (label == ignore_index)
        label_safe = torch.where(ignore_mask, torch.zeros_like(label), label)
        loss = F.cross_entropy(
            pred, label_safe, weight=class_weight, reduction="none",
        )
        loss = loss.masked_fill(ignore_mask, 0.0)

        if avg_factor is None and reduction == "mean":
            label_weights = class_weight[label_safe]
            label_weights = label_weights.masked_fill(ignore_mask, 0.0)
            avg_factor = label_weights.sum().clamp_min(1.0)

        from mmseg.models.losses.utils import weight_reduce_loss
        return weight_reduce_loss(
            loss, weight=weight.float() if weight is not None else None,
            reduction=reduction, avg_factor=avg_factor,
        )

    _cel.cross_entropy = safe_cross_entropy


_patch_cross_entropy_for_mps_class_weights()


CLASSES = (
    "background",
    "water",
    "vegetation",
    "buildings",
    "roads_runway_taxiway",
)

PALETTE = [
    [0, 0, 0],
    [255, 80, 0],
    [40, 170, 40],
    [220, 220, 220],
    [70, 70, 230],
]

MODEL_VARIANTS = {
    "b0": {
        "embed_dims": 32,
        "num_layers": [2, 2, 2, 2],
        "num_heads": [1, 2, 5, 8],
        "in_channels": [32, 64, 160, 256],
    },
    "b1": {
        "embed_dims": 64,
        "num_layers": [2, 2, 2, 2],
        "num_heads": [1, 2, 5, 8],
        "in_channels": [64, 128, 320, 512],
    },
    "b2": {
        "embed_dims": 64,
        "num_layers": [3, 4, 6, 3],
        "num_heads": [1, 2, 5, 8],
        "in_channels": [64, 128, 320, 512],
    },
    "b5": {
        "embed_dims": 64,
        "num_layers": [3, 6, 40, 3],
        "num_heads": [1, 2, 5, 8],
        "in_channels": [64, 128, 320, 512],
    },
}

IMAGENET_PRETRAIN = {
    "b0": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth",
    "b1": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth",
    "b2": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth",
    "b5": "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth",
}

ADE20K_PRETRAIN = {
    "b0": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
    "b1": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth",
    "b2": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth",
    "b5": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def prepare_mmseg_dataset(dataset_root: Path, out_root: Path, val_ratio: float, seed: int) -> tuple[int, int]:
    meta_path = dataset_root / "meta_patches.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    rows = list(csv.DictReader(meta_path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError("meta_patches.csv is empty")

    pairs: list[tuple[Path, Path, str]] = []
    for row in rows:
        img = dataset_root / row["image_path"]
        msk = dataset_root / row["mask_path"]
        if img.exists() and msk.exists():
            pairs.append((img, msk, row["patch_id"]))

    if len(pairs) < 20:
        raise RuntimeError(f"Too few patch pairs for finetune: {len(pairs)}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    val_count = max(1, int(len(pairs) * val_ratio))
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = out_root / "img_dir" / split_name
        ann_dir = out_root / "ann_dir" / split_name
        ensure_dir(img_dir)
        ensure_dir(ann_dir)

        for img_src, msk_src, patch_id in split_pairs:
            img_dst = img_dir / f"{patch_id}.png"
            msk_dst = ann_dir / f"{patch_id}.png"
            link_or_copy(img_src, img_dst)
            link_or_copy(msk_src, msk_dst)

    return len(train_pairs), len(val_pairs)


def compute_visual_feature(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).reshape(-1).astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_stats = np.array([gray.mean() / 255.0, gray.std() / 255.0], dtype=np.float32)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = np.array([float(edges.mean() / 255.0)], dtype=np.float32)

    feat = np.concatenate([hist, gray_stats, edge_ratio], axis=0)
    norm = float(np.linalg.norm(feat))
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def select_focus_patch_ids(
    data_root: Path,
    start_tile_path: Path,
    top_k: int,
    min_building: int,
    min_roads: int,
) -> list[str]:
    train_img_dir = data_root / "img_dir" / "train"
    train_ann_dir = data_root / "ann_dir" / "train"

    start_img = cv2.imread(str(start_tile_path), cv2.IMREAD_COLOR)
    if start_img is None:
        raise RuntimeError(f"Failed to read start tile: {start_tile_path}")
    start_feat = compute_visual_feature(start_img)

    records: list[tuple[float, str, bool, bool]] = []
    for img_path in train_img_dir.glob("*.png"):
        patch_id = img_path.stem
        mask_path = train_ann_dir / f"{patch_id}.png"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        feat = compute_visual_feature(img)
        dist = float(np.linalg.norm(feat - start_feat))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        has_building = bool(np.any(mask == 3))
        has_roads = bool(np.any(mask == 4))

        records.append((dist, patch_id, has_building, has_roads))

    if not records:
        raise RuntimeError("No train records available for focus subset selection")

    records.sort(key=lambda x: x[0])

    selected: list[str] = []
    used: set[str] = set()
    building_count = 0
    roads_count = 0

    def add_patch(patch_id: str, has_building: bool, has_roads: bool) -> None:
        nonlocal building_count, roads_count
        if patch_id in used or len(selected) >= top_k:
            return
        selected.append(patch_id)
        used.add(patch_id)
        if has_building:
            building_count += 1
        if has_roads:
            roads_count += 1

    for dist, patch_id, has_building, has_roads in records:
        if building_count >= min_building:
            break
        if has_building:
            add_patch(patch_id, has_building, has_roads)

    for dist, patch_id, has_building, has_roads in records:
        if roads_count >= min_roads:
            break
        if has_roads:
            add_patch(patch_id, has_building, has_roads)

    for dist, patch_id, has_building, has_roads in records:
        if len(selected) >= top_k:
            break
        add_patch(patch_id, has_building, has_roads)

    return selected


def create_focus_dataset(data_root: Path, focus_root: Path, selected_patch_ids: list[str]) -> tuple[int, int]:
    train_img_src = data_root / "img_dir" / "train"
    train_ann_src = data_root / "ann_dir" / "train"
    val_img_src = data_root / "img_dir" / "val"
    val_ann_src = data_root / "ann_dir" / "val"

    for patch_id in selected_patch_ids:
        src_img = train_img_src / f"{patch_id}.png"
        src_ann = train_ann_src / f"{patch_id}.png"
        if src_img.exists() and src_ann.exists():
            link_or_copy(src_img, focus_root / "img_dir" / "train" / src_img.name)
            link_or_copy(src_ann, focus_root / "ann_dir" / "train" / src_ann.name)

    val_count = 0
    for src_img in val_img_src.glob("*.png"):
        src_ann = val_ann_src / src_img.name
        if not src_ann.exists():
            continue
        link_or_copy(src_img, focus_root / "img_dir" / "val" / src_img.name)
        link_or_copy(src_ann, focus_root / "ann_dir" / "val" / src_ann.name)
        val_count += 1

    train_count = len(list((focus_root / "img_dir" / "train").glob("*.png")))
    return train_count, val_count


def estimate_class_weights(train_ann_dir: Path, num_classes: int) -> list[float]:
    counts = np.zeros(num_classes, dtype=np.float64)
    masks = list(train_ann_dir.glob("*.png"))
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binc = np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
        counts += binc.astype(np.float64)

    total = float(counts.sum())
    if total <= 0:
        return [1.0] * num_classes

    freq = counts / total
    # Downweight dominant classes and mildly upweight rare ones.
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / max(weights.mean(), 1e-8)
    weights = np.clip(weights, 0.25, 4.0)
    # Background should not dominate optimization.
    weights[0] = min(weights[0], 0.5)
    return [float(round(w, 4)) for w in weights.tolist()]


def cleanup_water_labels(data_root: Path, water_class: int = 1, vegetation_class: int = 2) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    for split in ("train", "val"):
        ann_dir = data_root / "ann_dir" / split
        img_dir = data_root / "img_dir" / split
        converted_total = 0
        water_total = 0

        for mask_path in ann_dir.glob("*.png"):
            img_path = img_dir / mask_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if mask is None or image is None:
                continue

            water = mask == water_class
            water_count = int(water.sum())
            if water_count == 0:
                continue

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]

            blue_like = (h >= 85) & (h <= 140) & (s >= 40) & (v >= 20)
            dark_water_like = (v <= 55) & (s <= 85)
            keep_as_water = blue_like | dark_water_like
            convert_to_vegetation = water & (~keep_as_water)

            converted = int(convert_to_vegetation.sum())
            if converted > 0:
                mask[convert_to_vegetation] = vegetation_class
                cv2.imwrite(str(mask_path), mask)

            converted_total += converted
            water_total += water_count

        stats[split] = (converted_total, water_total)

    return stats


def resolve_pretrain_checkpoint(
    model_size: str,
    pretrain_source: Literal["imagenet", "ade20k"],
    init_checkpoint: str | None,
) -> tuple[str, Literal["backbone", "segmentor"]]:
    if init_checkpoint:
        return init_checkpoint, "segmentor"

    if pretrain_source == "imagenet":
        return IMAGENET_PRETRAIN[model_size], "backbone"

    return ADE20K_PRETRAIN[model_size], "segmentor"


def build_quick_cfg(
    base_cfg_path: Path,
    data_root: Path,
    work_dir: Path,
    model_size: str,
    pretrain_source: Literal["imagenet", "ade20k"],
    init_checkpoint: str | None,
    max_iters: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    class_weights: list[float] | None,
) -> Config:
    cfg = Config.fromfile(str(base_cfg_path))

    cfg.work_dir = str(work_dir)
    cfg.resume = False

    variant = MODEL_VARIANTS[model_size]
    cfg.model.backbone.embed_dims = variant["embed_dims"]
    cfg.model.backbone.num_layers = variant["num_layers"]
    cfg.model.backbone.num_heads = variant["num_heads"]
    cfg.model.decode_head.in_channels = variant["in_channels"]

    ckpt, init_mode = resolve_pretrain_checkpoint(model_size, pretrain_source, init_checkpoint)
    if init_mode == "backbone":
        cfg.load_from = None
        cfg.model.backbone.init_cfg = {"type": "Pretrained", "checkpoint": ckpt}
    else:
        cfg.load_from = ckpt
        cfg.model.backbone.init_cfg = {"type": "Pretrained", "checkpoint": IMAGENET_PRETRAIN[model_size]}

    num_classes = len(CLASSES)
    cfg.model.decode_head.num_classes = num_classes
    cfg.model.decode_head.align_corners = False
    if "auxiliary_head" in cfg.model and cfg.model.auxiliary_head is not None:
        try:
            cfg.model.auxiliary_head.num_classes = num_classes
        except Exception:
            pass

    cfg.data_preprocessor.size = (512, 512)

    metainfo = {"classes": CLASSES, "palette": PALETTE}

    # Расширенный масштаб (0.4..2.0) эмулирует разную высоту полёта над землёй
    # при фиксированном фокусе. Albu-блок добавляет снег/туман/дождь, чтобы
    # модель не падала при смене сезона относительно тренировочных тайлов.
    # Albumentations 2.x: snow_point_range / fog_coef_range tuples (старые
    # snow_point_lower/upper и fog_coef_lower/upper удалены).
    albu_transforms = [
        {"type": "RandomSnow", "snow_point_range": (0.1, 0.3),
         "brightness_coeff": 2.5, "p": 0.15},
        {"type": "RandomFog", "fog_coef_range": (0.1, 0.4),
         "alpha_coef": 0.08, "p": 0.15},
        {"type": "RandomRain", "blur_value": 3, "p": 0.05},
        {"type": "RandomShadow", "p": 0.2},
        {"type": "RGBShift", "r_shift_limit": 15, "g_shift_limit": 15, "b_shift_limit": 15, "p": 0.3},
        {"type": "CLAHE", "clip_limit": 2.0, "p": 0.15},
    ]
    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "reduce_zero_label": False},
        {"type": "RandomResize", "scale": (1024, 1024), "ratio_range": (0.4, 2.0), "keep_ratio": True},
        {"type": "RandomCrop", "crop_size": (512, 512), "cat_max_ratio": 0.9},
        {"type": "RandomFlip", "prob": 0.5, "direction": "horizontal"},
        {"type": "RandomFlip", "prob": 0.2, "direction": "vertical"},
        {"type": "RandomRotate", "prob": 0.5, "degree": 30, "pad_val": 0, "seg_pad_val": 0},
        {"type": "PhotoMetricDistortion",
         "brightness_delta": 32,
         "contrast_range": (0.5, 1.5),
         "saturation_range": (0.5, 1.5),
         "hue_delta": 18},
        {"type": "Albu", "transforms": albu_transforms,
         "keymap": {"img": "image"}},
        {"type": "PackSegInputs"},
    ]

    eval_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": (512, 512), "keep_ratio": False},
        {"type": "LoadAnnotations", "reduce_zero_label": False},
        {"type": "PackSegInputs"},
    ]

    train_dataset = {
        "type": "BaseSegDataset",
        "data_root": str(data_root),
        "data_prefix": {"img_path": "img_dir/train", "seg_map_path": "ann_dir/train"},
        "img_suffix": ".png",
        "seg_map_suffix": ".png",
        "reduce_zero_label": False,
        "metainfo": metainfo,
        "pipeline": train_pipeline,
    }

    val_dataset = {
        "type": "BaseSegDataset",
        "data_root": str(data_root),
        "data_prefix": {"img_path": "img_dir/val", "seg_map_path": "ann_dir/val"},
        "img_suffix": ".png",
        "seg_map_suffix": ".png",
        "reduce_zero_label": False,
        "metainfo": metainfo,
        "pipeline": eval_pipeline,
    }

    cfg.train_dataloader = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "sampler": {"type": "InfiniteSampler", "shuffle": True},
        "dataset": train_dataset,
    }
    cfg.val_dataloader = {
        "batch_size": 1,
        "num_workers": max(1, num_workers // 2),
        "persistent_workers": num_workers > 1,
        "sampler": {"type": "DefaultSampler", "shuffle": False},
        "dataset": val_dataset,
    }
    cfg.test_dataloader = cfg.val_dataloader

    cfg.train_cfg.max_iters = max_iters
    val_interval = max(1, min(max_iters, max(100, max_iters // 6)))
    cfg.train_cfg.val_interval = val_interval

    cfg.val_cfg = {"type": "ValLoop"}
    cfg.test_cfg = {"type": "TestLoop"}
    cfg.val_evaluator = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}
    cfg.test_evaluator = {"type": "IoUMetric", "iou_metrics": ["mIoU"]}

    ckpt_interval = max(1, min(max_iters, max(200, max_iters // 3)))
    cfg.default_hooks.checkpoint.interval = ckpt_interval
    cfg.default_hooks.checkpoint.save_best = "mIoU"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.max_keep_ckpts = 4
    cfg.default_hooks.logger.interval = 10

    cfg.optim_wrapper.optimizer.lr = lr
    cfg.optim_wrapper.optimizer.weight_decay = weight_decay
    warmup_steps = int(max_iters * warmup_ratio)
    warmup_end = max(1, min(max_iters - 1, warmup_steps if warmup_steps >= 1 else 1))
    cfg.param_scheduler = [
        {"type": "LinearLR", "begin": 0, "end": warmup_end, "by_epoch": False, "start_factor": 1e-5},
        {"type": "PolyLR", "begin": warmup_end, "end": max_iters, "by_epoch": False, "power": 1.0, "eta_min": 0.0},
    ]

    if class_weights is not None:
        try:
            cfg.model.decode_head.loss_decode.class_weight = class_weights
        except Exception:
            cfg.model.decode_head.loss_decode["class_weight"] = class_weights

    cfg.launcher = "none"
    cfg.env_cfg.dist_cfg = {"backend": "nccl"}

    return cfg


def latest_checkpoint(work_dir: Path) -> Path:
    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest

    candidates = sorted(
        work_dir.glob("iter_*.pth"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")
    return candidates[-1]


def best_checkpoint(work_dir: Path) -> Path | None:
    candidates = sorted(
        work_dir.glob("best_mIoU*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1]


def select_infer_checkpoint(work_dir: Path, prefer_best: bool) -> Path:
    if prefer_best:
        best = best_checkpoint(work_dir)
        if best is not None:
            return best
    return latest_checkpoint(work_dir)


def download_start_tile_google(lat: float, lon: float, zoom: int, out_path: Path) -> Path:
    x, y = deg2num(lat, lon, zoom)
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    headers = {"User-Agent": "Mozilla/5.0 (Avia-Geo-Fusion)"}

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode start tile image")

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)
    return out_path


def render_overlay(image_path: Path, pred_mask: np.ndarray, out_overlay: Path, out_mask: Path, alpha: float = 0.45) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read {image_path}")

    color = np.zeros_like(image)
    for cls_idx, rgb in enumerate(PALETTE):
        # OpenCV uses BGR, but palette is defined in RGB.
        color[pred_mask == cls_idx] = [rgb[2], rgb[1], rgb[0]]

    overlay = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0)

    ensure_dir(out_overlay.parent)
    cv2.imwrite(str(out_mask), pred_mask.astype(np.uint8))
    cv2.imwrite(str(out_overlay), overlay)


def suppress_green_water(
    image_path: Path,
    pred_mask: np.ndarray,
    blue_margin: int,
    green_red_margin: int,
    green_blue_slack: int,
) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return pred_mask, 0

    b, g, r = cv2.split(image)
    water = pred_mask == 1
    likely_water = (b > (g + blue_margin)) & (b > (r + blue_margin))
    likely_green_land = (g > (r + green_red_margin)) & (g >= (b - green_blue_slack))
    convert_to_vegetation = water & likely_green_land & (~likely_water)
    changed = int(convert_to_vegetation.sum())
    if changed == 0:
        return pred_mask, 0

    updated = pred_mask.copy()
    updated[convert_to_vegetation] = 2
    return updated, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick SegFormer finetune and start-tile inference")
    parser.add_argument("--dataset-root", default="data/overture_ru_dataset_starter", help="Dataset root with meta_patches.csv")
    parser.add_argument("--base-config", default="", help="Optional MMSeg config path; if empty uses SegFormer-B0 ADE20K config from installed MMSeg")
    parser.add_argument("--model-size", default="b0", choices=sorted(MODEL_VARIANTS.keys()))
    parser.add_argument("--pretrain-source", default="ade20k", choices=["imagenet", "ade20k"])
    parser.add_argument("--init-checkpoint", default="", help="Optional explicit checkpoint path/URL; overrides --pretrain-source")
    parser.add_argument("--work-dir", default="results/segformer_overture_quick")
    parser.add_argument("--iters", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--use-class-weights", action="store_true", default=False)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--manual-class-weights", default="",
                        help="Comma-separated class weights override (e.g. '0.5,1.0,0.4,0.6,1.5'). "
                             "If set, takes priority over --use-class-weights auto estimation.")
    parser.add_argument("--water-label-cleanup", action="store_true", default=True)
    parser.add_argument("--no-water-label-cleanup", dest="water_label_cleanup", action="store_false")
    parser.add_argument("--focus-iters", type=int, default=0, help="Second-stage finetune iters on visually similar subset")
    parser.add_argument("--focus-topk", type=int, default=180, help="How many most similar train patches to use in focus stage")
    parser.add_argument("--focus-min-building", type=int, default=60, help="Minimum building-containing patches in focus stage")
    parser.add_argument("--focus-min-roads", type=int, default=80, help="Minimum roads-containing patches in focus stage")
    parser.add_argument("--focus-lr-mult", type=float, default=0.35, help="LR multiplier for second-stage focus finetune")
    parser.add_argument("--stage1-checkpoint", default="", help="Optional existing checkpoint path to skip stage1 training")
    parser.add_argument("--prefer-best-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-prefer-best-checkpoint", dest="prefer_best_checkpoint", action="store_false")
    parser.add_argument("--postprocess-green-water", action="store_true", default=False)
    parser.add_argument("--postprocess-blue-margin", type=int, default=6)
    parser.add_argument("--postprocess-green-red-margin", type=int, default=8)
    parser.add_argument("--postprocess-green-blue-slack", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-lat", type=float, default=55.086025)
    parser.add_argument("--start-lon", type=float, default=38.149033)
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if args.base_config.strip():
        base_cfg = Path(args.base_config).resolve()
    else:
        import mmseg

        base_cfg = (
            Path(mmseg.__file__).resolve().parent
            / ".mim"
            / "configs"
            / "segformer"
            / "segformer_mit-b0_8xb2-160k_ade20k-512x512.py"
        )

    init_ckpt = args.init_checkpoint.strip() or None
    work_dir = Path(args.work_dir).resolve()
    mmseg_data_root = dataset_root / "mmseg_ready_quick"
    mmseg_focus_root = dataset_root / "mmseg_ready_focus"
    focus_work_dir = work_dir / "focus_refine"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    if mmseg_data_root.exists():
        shutil.rmtree(mmseg_data_root)
    if mmseg_focus_root.exists():
        shutil.rmtree(mmseg_focus_root)

    ensure_dir(work_dir)
    train_count, val_count = prepare_mmseg_dataset(dataset_root, mmseg_data_root, args.val_ratio, args.seed)
    print(f"prepared_train={train_count} prepared_val={val_count}")
    print(
        f"train_hparams model={args.model_size} init={args.pretrain_source} "
        f"iters={args.iters} batch={args.batch_size} lr={args.lr} wd={args.weight_decay}"
    )

    if args.water_label_cleanup:
        cleanup_stats = cleanup_water_labels(mmseg_data_root)
        for split_name, (converted, water_total) in cleanup_stats.items():
            ratio = (100.0 * converted / water_total) if water_total > 0 else 0.0
            print(
                f"water_cleanup split={split_name} converted={converted} total_water={water_total} "
                f"ratio={ratio:.2f}%"
            )

    class_weights = None
    if args.manual_class_weights.strip():
        class_weights = [float(x) for x in args.manual_class_weights.split(",")]
        if len(class_weights) != len(CLASSES):
            raise ValueError(f"manual-class-weights expects {len(CLASSES)} values, got {len(class_weights)}")
        print(f"class_weights (manual override) {class_weights}")
    elif args.use_class_weights:
        class_weights = estimate_class_weights(mmseg_data_root / "ann_dir" / "train", len(CLASSES))
        print("class_weights (auto)", class_weights)

    cfg = build_quick_cfg(
        base_cfg_path=base_cfg,
        data_root=mmseg_data_root,
        work_dir=work_dir,
        model_size=args.model_size,
        pretrain_source=args.pretrain_source,
        init_checkpoint=init_ckpt,
        max_iters=args.iters,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        class_weights=class_weights,
    )

    cfg_path = work_dir / "segformer_overture_quick_cfg.py"
    cfg.dump(str(cfg_path))
    print(f"config_saved={cfg_path}")

    start_tile = work_dir / f"start_tile_google_z{args.zoom}.png"
    download_start_tile_google(args.start_lat, args.start_lon, args.zoom, start_tile)

    stage1_ckpt_arg = args.stage1_checkpoint.strip()
    if stage1_ckpt_arg:
        ckpt_path = Path(stage1_ckpt_arg).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
        print(f"checkpoint_stage1_reuse={ckpt_path}")
    else:
        runner = Runner.from_cfg(cfg)
        runner.train()
        ckpt_path = select_infer_checkpoint(work_dir, prefer_best=args.prefer_best_checkpoint)
        print(f"checkpoint_stage1={ckpt_path}")

    if args.focus_iters > 0:
        if focus_work_dir.exists():
            shutil.rmtree(focus_work_dir)

        selected_patch_ids = select_focus_patch_ids(
            data_root=mmseg_data_root,
            start_tile_path=start_tile,
            top_k=max(20, args.focus_topk),
            min_building=max(0, args.focus_min_building),
            min_roads=max(0, args.focus_min_roads),
        )
        focus_train_count, focus_val_count = create_focus_dataset(mmseg_data_root, mmseg_focus_root, selected_patch_ids)
        print(
            f"focus_subset selected={len(selected_patch_ids)} "
            f"prepared_train={focus_train_count} prepared_val={focus_val_count}"
        )

        focus_lr = args.lr * max(0.05, args.focus_lr_mult)
        cfg_focus = build_quick_cfg(
            base_cfg_path=base_cfg,
            data_root=mmseg_focus_root,
            work_dir=focus_work_dir,
            model_size=args.model_size,
            pretrain_source=args.pretrain_source,
            init_checkpoint=init_ckpt,
            max_iters=args.focus_iters,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=focus_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=min(0.2, args.warmup_ratio),
            class_weights=class_weights,
        )
        cfg_focus.load_from = str(ckpt_path)

        focus_cfg_path = focus_work_dir / "segformer_overture_focus_cfg.py"
        ensure_dir(focus_work_dir)
        cfg_focus.dump(str(focus_cfg_path))
        print(f"focus_config_saved={focus_cfg_path}")

        focus_runner = Runner.from_cfg(cfg_focus)
        focus_runner.train()

        ckpt_path = select_infer_checkpoint(focus_work_dir, prefer_best=args.prefer_best_checkpoint)
        print(f"checkpoint_stage2={ckpt_path}")

    model = init_model(str(cfg_path), str(ckpt_path), device=args.device)
    result = inference_model(model, str(start_tile))
    mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    if args.postprocess_green_water:
        mask, changed = suppress_green_water(
            start_tile,
            mask,
            blue_margin=args.postprocess_blue_margin,
            green_red_margin=args.postprocess_green_red_margin,
            green_blue_slack=args.postprocess_green_blue_slack,
        )
        print("postprocess_green_water_changed", changed)

    out_mask = work_dir / "start_tile_pred_mask.png"
    out_overlay = work_dir / "start_tile_pred_overlay.png"
    render_overlay(start_tile, mask, out_overlay, out_mask)

    uniq, cnt = np.unique(mask, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print("start_tile_pred_hist", hist)
    print(f"overlay={out_overlay}")


if __name__ == "__main__":
    main()
