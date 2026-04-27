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

def visualize_segmentation(img_path, mask_result, out_path):
    img = cv2.imread(img_path)[:, :, ::-1] # BGR to RGB
    mask = mask_result.pred_sem_seg.data[0].cpu().numpy()

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
    ax1.set_title("Original Satellite Map (z=18)")
    ax1.axis('off')

    ax2.imshow(color_mask)
    ax2.set_title("Segmentation Map")
    ax2.axis('off')

    # Create legend
    unique_classes = np.unique(mask)
    patches = []
    for c in unique_classes:
        info = palette_info.get(c, {"name": f"Unknown {c}", "color": [0,0,0]})
        # matplotlib expects colors in 0-1 range
        c_norm = [x/255.0 for x in info["color"]]
        patches.append(mpatches.Patch(color=c_norm, label=info["name"]))

    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    sat_img_z18 = 'data/eval/4193.png'
    download_satellite_map(55.086025, 38.149033, 18, sat_img_z18)

    cfg = "configs/deeplabv3plus_r50-d8_4xb4-80k_loveda-512x512.py"
    ckpt = "configs/deeplabv3plus_r50-d8_512x512_80k_loveda_20211105_080442-f0720392.pth"
    model = init_model(cfg, ckpt, device='cuda:0')
    
    result = inference_model(model, sat_img_z18)
    mask = result.pred_sem_seg.data[0].cpu().numpy()
    
    print("Unique classes found in z18:", np.unique(mask))
    
    out_img = 'results/loveda_benchmark/DeepLabV3Plus_LoveDA_sat_z18_vis.png'
    visualize_segmentation(sat_img_z18, result, out_img)
    print(f"Done z18. Saved visualization to {out_img}")

if __name__ == '__main__':
    main()
