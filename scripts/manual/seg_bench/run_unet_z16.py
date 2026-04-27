import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmseg.apis import init_model, inference_model
from scripts.manual.seg_bench.run_loveda_benchmark import download_satellite_map

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)


def visualize(image_path, mask, out_path):
    image = cv2.imread(image_path)[:, :, ::-1]

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(CITYSCAPES_PALETTE)):
        color_mask[mask == i] = CITYSCAPES_PALETTE[i]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.imshow(image)
    ax1.set_title('Original Satellite (z16)')
    ax1.axis('off')

    ax2.imshow(color_mask)
    ax2.set_title('U-Net Segmentation (Cityscapes classes)')
    ax2.axis('off')

    unique = np.unique(mask)
    legend_patches = []
    for cls in unique:
        if 0 <= cls < len(CITYSCAPES_CLASSES):
            color = CITYSCAPES_PALETTE[cls] / 255.0
            legend_patches.append(mpatches.Patch(color=color, label=CITYSCAPES_CLASSES[cls]))

    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    sat_img = 'data/eval/sat_z17.jpg'
    download_satellite_map(55.086025, 38.149033, 17, sat_img)

    cfg = 'configs/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
    ckpt = 'configs/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'

    model = init_model(cfg, ckpt, device='cuda:0')

    result = inference_model(model, sat_img)
    mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

    unique = np.unique(mask)
    print('U-Net unique classes:', unique)

    out_img = 'results/loveda_benchmark/UNet_cityscapes_sat_z16_vis.jpg'
    visualize(sat_img, mask, out_img)
    print('Saved:', out_img)


if __name__ == '__main__':
    main()
