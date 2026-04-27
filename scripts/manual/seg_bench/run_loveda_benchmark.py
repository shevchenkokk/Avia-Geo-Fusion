import os
import cv2
import urllib.request
import math
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_discovery import pick_default_video

try:
    from mmseg.apis import inference_model, init_model
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False

def extract_frame(video_path, output_path, frame_idx=150):
    assert os.path.exists(video_path), f"Видео не найдено: {video_path}"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()
    return output_path

def download_satellite_map(lat, lon, zoom=17, output_path="map.jpg"):
    if os.path.exists(output_path):
        return output_path
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    x, y = deg2num(lat, lon, zoom)
    img_size = 256
    full_image = np.zeros((img_size * 3, img_size * 3, 3), dtype=np.uint8)
    
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y+dy}/{x+dx}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                full_image[(dy+1)*img_size:(dy+2)*img_size, (dx+1)*img_size:(dx+2)*img_size] = image
            except Exception:
                pass
                
    cv2.imwrite(output_path, full_image)
    return output_path

def blend_mask(image_path, result, out_path, alpha=0.5):
    image = cv2.imread(image_path)
    mask = result.pred_sem_seg.data[0].cpu().numpy()
    
    # LoveDA palette
    # Background, Building, Road, Water, Barren, Forest, Agriculture
    palette = np.array([
        [255, 255, 255], # 0: Background (White)
        [255, 0, 0],     # 1: Building (Red)
        [255, 255, 0],   # 2: Road (Yellow)
        [0, 0, 255],     # 3: Water (Blue)
        [159, 129, 183], # 4: Barren (Grey/Purple)
        [0, 255, 0],     # 5: Forest (Green)
        [255, 195, 128], # 6: Agriculture (Orange)
    ])
    
    color_mask = np.zeros_like(image)
    for c in range(min(len(palette), mask.max() + 1)):
        color_mask[mask == c] = palette[c]
        
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    cv2.imwrite(out_path, blended)

def main():
    os.makedirs('data/eval', exist_ok=True)
    os.makedirs('results/loveda_benchmark', exist_ok=True)
    
    drone_img = 'data/eval/drone_test.jpg'
    sat_img = 'data/eval/sat_test.jpg'

    video_path = pick_default_video(preferred_names=("GOPR0269.MP4", "GP010269.MP4"))
    if video_path is None:
        raise FileNotFoundError("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")

    extract_frame(str(video_path), drone_img, frame_idx=150)
    download_satellite_map(lat=55.086025, lon=38.149033, zoom=17, output_path=sat_img)
    
    models = {
        "DeepLabV3Plus_LoveDA": {
            "config": "configs/deeplabv3plus_r50-d8_4xb4-80k_loveda-512x512.py",
            "checkpoint": "configs/deeplabv3plus_r50-d8_512x512_80k_loveda_20211105_080442-f0720392.pth"
        }
    }

    for name, urls in models.items():
        print(f"\n[*] Инициализация модели: {name}")
        model = init_model(urls["config"], urls["checkpoint"], device='cuda:0')
        for img_name, img_path in [("drone", drone_img), ("satellite", sat_img)]:
            print(f"[*] Инференс {name} на {img_name}...")
            result = inference_model(model, img_path)
            out_file = f"results/loveda_benchmark/{name}_{img_name}.jpg"
            blend_mask(img_path, result, out_file)
            print(f"[+] Экспортирован: {out_file}")
            
    print("\n[+] Эксперименты LoveDA завершены.")

if __name__ == '__main__':
    main()
