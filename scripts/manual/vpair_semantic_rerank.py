"""VPair semantic re-ranking: top-K из DINOv2 retriever → переранжируем по
mask-to-mask IoU (SegFormer на query + кандидатах).

Baseline R@1@25m = 14%. Re-ranking тестирует может ли семантика поднять R@1
если правильный reference часто есть в top-K (R@10@25m = 61%).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patch_mmseg():
    from mmseg.models.backbones import mit as _mit
    from mmseg.models.decode_heads import segformer_head as _shead
    from mmseg.models.utils import resize as _resize

    def patched_attn(self, x, hw_shape, identity=None):
        from mmseg.models.utils import nchw_to_nlc, nlc_to_nchw
        x_q = x
        if self.sr_ratio > 1:
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

    def patched_head(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx].contiguous()
            up = _resize(input=self.convs[idx](x), size=inputs[0].shape[2:],
                         mode=self.interpolate_mode, align_corners=self.align_corners)
            outs.append(up.contiguous())
        out = self.fusion_conv(torch.cat(outs, dim=1).contiguous())
        return self.cls_seg(out)

    _mit.EfficientMultiheadAttention.forward = patched_attn
    _shead.SegformerHead.forward = patched_head


_patch_mmseg()
_orig_load = torch.load
torch.load = lambda *a, **k: (_orig_load(*a, **{**k, "weights_only": False}) if "weights_only" not in k else _orig_load(*a, **k))


def class_iou_score(mask_a: np.ndarray, mask_b: np.ndarray, class_ids=(1, 2, 3, 4)) -> float:
    if mask_a.shape != mask_b.shape:
        h, w = min(mask_a.shape[0], mask_b.shape[0]), min(mask_a.shape[1], mask_b.shape[1])
        mask_a = cv2.resize(mask_a, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_b = cv2.resize(mask_b, (w, h), interpolation=cv2.INTER_NEAREST)
    ious = []
    for c in class_ids:
        a = mask_a == c
        b = mask_b == c
        union = (a | b).sum()
        if union > 100:
            ious.append((a & b).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/external/vpair_sample")
    p.add_argument("--manifest-dir", default="data/vpair/manifests")
    p.add_argument("--db-path", default="results/vpair_vpr_baseline/vpair_dinov2_db.npz")
    p.add_argument("--seg-config",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py")
    p.add_argument("--seg-checkpoint",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of DINOv2 sim (1−α — weight of semantic IoU)")
    p.add_argument("--output", default="results/vpair_semantic_rerank")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Загрузка SegFormer и DINOv2-поиска...")
    from mmseg.apis import init_model, inference_model
    from src.retriever import ReferenceDatabase, Retriever
    seg = init_model(args.seg_config, args.seg_checkpoint, device=args.device)

    # Загружаем существующую базу DINOv2 через Retriever.
    retr = Retriever(device=args.device)
    retr.load_database(Path(args.db_path).with_suffix(""))

    # Манифесты запросов и галереи.
    queries = list(csv.DictReader(open(f"{args.manifest_dir}/queries.csv")))
    refs = list(csv.DictReader(open(f"{args.manifest_dir}/references.csv")))
    dists_path = Path(f"{args.manifest_dir}/distractors.csv")
    dists = list(csv.DictReader(open(dists_path))) if dists_path.exists() else []
    # Порядок галереи: опорные изображения, затем отвлекающие примеры.
    gallery = refs + dists

    if args.max_queries > 0:
        queries = queries[:args.max_queries]
    print(f"queries={len(queries)} gallery={len(gallery)} top_k={args.top_k}")

    results = []
    for i, q in enumerate(queries):
        gt_lat, gt_lon = float(q["lat"]), float(q["lon"])
        q_img = cv2.imread(f"{args.dataset_root}/{q['path']}")
        if q_img is None:
            continue

        # Первые K кандидатов DINOv2.
        topk_cands = retr.query_image(q_img, top_k=args.top_k)
        if not topk_cands:
            continue

        # SegFormer на запросном изображении.
        r_q = inference_model(seg, q_img)
        q_mask = r_q.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

        # SegFormer и IoU для каждого кандидата.
        sem_ious = []
        cand_latlons = []
        dinov2_scores = []
        for cand in topk_cands:
            g_idx = cand["index"]
            g = gallery[g_idx]
            g_img = cv2.imread(f"{args.dataset_root}/{g['path']}")
            if g_img is None:
                sem_ious.append(0.0)
                cand_latlons.append((float(g["lat"]), float(g["lon"])))
                dinov2_scores.append(cand["score"])
                continue
            r_g = inference_model(seg, g_img)
            g_mask = r_g.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
            sem_ious.append(class_iou_score(q_mask, g_mask))
            cand_latlons.append((float(g["lat"]), float(g["lon"])))
            dinov2_scores.append(cand["score"])

        dinov2_scores = np.array(dinov2_scores)
        sem_ious = np.array(sem_ious)
        # Нормируем оба скоринга в диапазон [0, 1].
        def _norm(x):
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 1e-6 else x * 0
        combined = args.alpha * _norm(dinov2_scores) + (1 - args.alpha) * _norm(sem_ious)
        rerank_order = np.argsort(-combined)

        # Сравниваем первый кандидат до и после переранжирования.
        before_lat, before_lon = cand_latlons[0]  # top-1 от DINOv2
        after_lat, after_lon = cand_latlons[rerank_order[0]]
        d_before = haversine_m(gt_lat, gt_lon, before_lat, before_lon)
        d_after = haversine_m(gt_lat, gt_lon, after_lat, after_lon)

        results.append({
            "qid": q["image_id"],
            "d_dinov2_m": d_before,
            "d_rerank_m": d_after,
            "topk_sem_ious": sem_ious.tolist(),
            "topk_dinov2": dinov2_scores.tolist(),
        })
        if (i + 1) % 20 == 0:
            print(f"  q {i+1}/{len(queries)}: d_dino={d_before:.0f}m, d_rerank={d_after:.0f}m")

    # Сводка.
    print("\n=== Recall@1 comparison ===")
    summary = {"num_queries": len(results), "top_k": args.top_k, "alpha": args.alpha, "recall_R1": {}}
    for radius in [25, 50, 100, 200]:
        r_dino = 100.0 * sum(1 for r in results if r["d_dinov2_m"] <= radius) / max(1, len(results))
        r_re = 100.0 * sum(1 for r in results if r["d_rerank_m"] <= radius) / max(1, len(results))
        delta = r_re - r_dino
        sign = "+" if delta >= 0 else ""
        print(f"  R@1@{radius:3}m:  DINOv2={r_dino:5.1f}%  →  +sem_rerank={r_re:5.1f}%  ({sign}{delta:.1f}pp)")
        summary["recall_R1"][f"R@1_{radius}m"] = {"dinov2": r_dino, "rerank": r_re, "delta_pp": delta}

    # Медианная ошибка.
    if results:
        med_d = float(np.median([r["d_dinov2_m"] for r in results]))
        med_r = float(np.median([r["d_rerank_m"] for r in results]))
        print(f"\n  median top-1 dist: DINOv2={med_d:.0f}m  rerank={med_r:.0f}m  ({med_r-med_d:+.0f}m)")
        summary["median_top1_dist_m"] = {"dinov2": med_d, "rerank": med_r}

    (out / "rerank_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out / 'rerank_summary.json'}")


if __name__ == "__main__":
    main()
