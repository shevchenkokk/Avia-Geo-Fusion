import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mmseg.apis import inference_model, init_model
except ImportError:
    pass

from scripts.manual.seg_bench.run_loveda_benchmark import download_satellite_map


def enhance_contrast_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    out = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def infer_at_scale(model, image_bgr, scale):
    h, w = image_bgr.shape[:2]
    scaled = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    pred = inference_model(model, scaled)
    mask_scaled = pred.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(mask_scaled, (w, h), interpolation=cv2.INTER_NEAREST)


def hybrid_enhanced_mask(model, image_bgr):
    # Branch A: raw image keeps thin structures like roads better.
    raw_mask = infer_at_scale(model, image_bgr, scale=2.5)

    # Branch B: contrast-enhanced image recovers vegetation/buildings better.
    clahe_img = enhance_contrast_bgr(image_bgr)
    clahe_mask = infer_at_scale(model, clahe_img, scale=2.0)

    merged = raw_mask.copy()
    override = np.isin(clahe_mask, [1, 4, 5])
    merged[override] = clahe_mask[override]
    return merged

def visualize_segmentation(img_path, mask, out_path):
    img = cv2.imread(img_path)[:, :, ::-1] # BGR to RGB

    # LoveDA palette (RGB)
    palette_info = {
        0: {"name": "Background", "color": [255, 255, 255]},
        1: {"name": "Building", "color": [255, 0, 0]},
        2: {"name": "Road", "color": [255, 255, 0]},
        3: {"name": "Water", "color": [0, 0, 255]},
        4: {"name": "Barren", "color": [159, 129, 183]},
        5: {"name": "Forest", "color": [0, 255, 0]},
        6: {"name": "Agriculture", "color": [255, 195, 128]}
    }

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, info in palette_info.items():
        color_mask[mask == c] = info["color"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(img)
    ax1.set_title("Original Satellite Map (z=16)")
    ax1.axis('off')

    ax2.imshow(color_mask)
    ax2.set_title("Enhanced Segmentation Map")
    ax2.axis('off')

    # Create legend
    unique_classes = np.unique(mask)
    patches = []
    for c in unique_classes:
        info = palette_info.get(c, {"name": f"Unknown {c}", "color": [0,0,0]})
        c_norm = [x/255.0 for x in info["color"]]
        patches.append(mpatches.Patch(color=c_norm, label=info["name"]))

    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    sat_img_z16 = 'data/eval/sat_z16.jpg'
    download_satellite_map(55.086025, 38.149033, 16, sat_img_z16)

    cfg = "configs/deeplabv3plus_r50-d8_4xb4-80k_loveda-512x512.py"
    ckpt = "configs/deeplabv3plus_r50-d8_512x512_80k_loveda_20211105_080442-f0720392.pth"
    model = init_model(cfg, ckpt, device='cuda:0')

    orig_img = cv2.imread(sat_img_z16)
    mask_restored = hybrid_enhanced_mask(model, orig_img)

    unique, counts = np.unique(mask_restored, return_counts=True)
    print("Unique classes found after hybrid enhancement:", unique)
    print("Class pixel counts:", dict(zip(unique.tolist(), counts.tolist())))
    
    out_img = 'results/loveda_benchmark/DeepLabV3Plus_LoveDA_sat_z16_vis.png'
    visualize_segmentation(sat_img_z16, mask_restored, out_img)
    print(f"Done z16. Saved improved visualization to {out_img}")

if __name__ == '__main__':
    main()
