import cv2
import logging
import numpy as np
from pathlib import Path
from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.feature_matching import FeatureMatcher
from src.video_discovery import pick_default_video

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def match_and_save(frame_drone, map_cv2, method, out_path):
    print(f"\n--- Запуск сопоставления фичей ({method.upper()}) ---")
    matcher = FeatureMatcher(method=method)
    
    # Run the internal methods manually to get BOTH raw and inlier matches
    gray_drone = matcher._preprocess_image(frame_drone)
    gray_map = matcher._preprocess_image(map_cv2)
    kp1, des1 = matcher.detector.detectAndCompute(gray_drone, None)
    kp2, des2 = matcher.detector.detectAndCompute(gray_map, None)
    
    good_matches = []
    if method == 'sift':
        matches = matcher.matcher.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    else:
        matches = matcher.matcher.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)
        keep_count = max(10, int(len(good_matches) * 0.15))
        good_matches = good_matches[:keep_count]
        
    inlier_matches = []
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            for i, is_inlier in enumerate(matchesMask):
                if is_inlier:
                    inlier_matches.append(good_matches[i])
                    
    # Draw CHAOS (raw good matches before RANSAC filtering)
    debug_img_raw = cv2.drawMatches(
        frame_drone, kp1, 
        map_cv2, kp2, 
        good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 0, 255) # Red lines for chaos
    )
    cv2.imwrite(out_path.replace('.jpg', '_raw.jpg'), debug_img_raw)
    
    # Draw filtered
    debug_img_filtered = cv2.drawMatches(
        frame_drone, kp1, 
        map_cv2, kp2, 
        inlier_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0) # Green lines for filtered
    )
    cv2.imwrite(out_path, debug_img_filtered)
    print(f"Результат сопоставления сохранен в: {out_path} и _raw.jpg")

def test_pipeline():
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
        return
        
    default_geo = (55.086025, 38.149033, 750.0)
    v_proc = VideoProcessor(str(video_path), default_geo=default_geo)
    frame_drone = v_proc.extract_frame(0)
    
    pos_t0 = v_proc.telemetry.get_position_at_frame(0.0)
    
    map_loader = MapDownloader(zoom=17)
    map_img_pil, bbox = map_loader.get_basemap_for_location(
        lat=pos_t0.latitude, 
        lon=pos_t0.longitude, 
        radius_tiles=1
    )
    
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    h_drone, w_drone = frame_drone.shape[:2]
    h_map, w_map = map_cv2.shape[:2]
    
    if h_drone != h_map:
        scale = h_drone / float(h_map)
        new_w = int(w_map * scale)
        map_cv2 = cv2.resize(map_cv2, (new_w, h_drone))

    Path("results").mkdir(exist_ok=True)
    match_and_save(frame_drone, map_cv2, 'sift', "results/sift_matches_baseline.jpg")
    match_and_save(frame_drone, map_cv2, 'orb', "results/orb_matches_baseline.jpg")

if __name__ == '__main__':
    test_pipeline()
