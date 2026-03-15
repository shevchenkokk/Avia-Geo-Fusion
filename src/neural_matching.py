"""
Модуль для нейросетевого сопоставления кадров со спутниковыми картами.
Использует архитектуру LoFTR (Local Feature TRansformer) через библиотеку Kornia.
"""

import cv2
import torch
import numpy as np
import logging
import kornia as K
import kornia.feature as KF
from src.segmentation import SkySegmenter

logger = logging.getLogger(__name__)

class NeuralMatcher:
    """Нейросетевой мэтчер на основе LoFTR."""

    def __init__(self, pretrained_model: str = 'outdoor', use_segmentation: bool = True):
        """
        Инициализация нейросети и определение устройства (CPU/MPS/CUDA).
        """
        self.device = torch.device("cpu")
        self.use_segmentation = use_segmentation
        self.segmenter = SkySegmenter() if use_segmentation else None

        
        # Поддержка Apple Silicon (M1/M2/M3)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Нейросеть LoFTR будет использовать GPU (Apple MPS)")
        # Поддержка NVIDIA
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Нейросеть LoFTR будет использовать GPU (CUDA)")
        else:
            logger.info("Нейросеть LoFTR будет использовать CPU")

        # Загружаем предобученную модель LoFTR
        # 'outdoor' - обучена на MegaDepth (ландшафты, здания, города)
        self.matcher = KF.LoFTR(pretrained=pretrained_model).to(self.device).eval()

    def _prepare_image(self, img: np.ndarray, max_size: int = 640, is_drone: bool = False) -> tuple:
        """
        Подготовка изображения для LoFTR:
        - Возвращает (тензор, чб_изображение_оригинального_размера, коэффициент_масштабирования)
        """
        if is_drone and self.use_segmentation and self.segmenter:
            img = self.segmenter.remove_sky(img)

        # Градации серого для оригинала (чтобы вернуть и рисовать по нему)
        if len(img.shape) == 3:
            img_gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray_orig = img

        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_gray_orig, (new_w, new_h))
        else:
            img_resized = img_gray_orig
            
        tensor = K.image_to_tensor(img_resized, keepdim=False).float() / 255.0
        return tensor.to(self.device), img_gray_orig, scale

    def match(self, img_drone: np.ndarray, img_map: np.ndarray) -> dict:
        """
        Поиск совпадений с помощью LoFTR.
        """
        tensor_drone, gray_drone_orig, scale_drone = self._prepare_image(img_drone, is_drone=True)
        tensor_map, gray_map_orig, scale_map = self._prepare_image(img_map, is_drone=False)
        
        # LoFTR требует, чтобы размеры сторон были кратны 8
        def pad_to_multiple(tensor, multiple=8):
            _, _, h, w = tensor.shape
            pad_h = (multiple - (h % multiple)) % multiple
            pad_w = (multiple - (w % multiple)) % multiple
            return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate'), h, w

        tensor_drone_pad, orig_h1, orig_w1 = pad_to_multiple(tensor_drone)
        tensor_map_pad, orig_h2, orig_w2 = pad_to_multiple(tensor_map)

        input_dict = {
            "image0": tensor_drone_pad,
            "image1": tensor_map_pad
        }

        with torch.no_grad():
            matches_dict = self.matcher(input_dict)
            
        mkpts0 = matches_dict['keypoints0'].cpu().numpy()
        mkpts1 = matches_dict['keypoints1'].cpu().numpy()
        confidence = matches_dict['confidence'].cpu().numpy()

        # Возвращаем координаты в исходный масштаб оригинального изображения
        mkpts0 = mkpts0 / scale_drone
        mkpts1 = mkpts1 / scale_map

        conf_thresh = 0.2
        valid = confidence > conf_thresh
        mkpts0 = mkpts0[valid]
        mkpts1 = mkpts1[valid]
        confidence = confidence[valid]
        
        inlier_mask = np.zeros(len(mkpts0), dtype=bool)
        if len(mkpts0) > 4:
            H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                
        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]

        logger.info(f"LoFTR нашел {len(mkpts0)} совпадений, после RANSAC: {len(mkpts0_inliers)}")

        debug_img = self._draw_matches(gray_drone_orig, gray_map_orig, mkpts0_inliers, mkpts1_inliers)

        return {
            'mkpts0': mkpts0_inliers,
            'mkpts1': mkpts1_inliers,
            'debug_img': debug_img
        }

    def _draw_matches(self, img1, img2, pts1, pts2):
        """Вспомогательная функция для отрисовки линий."""
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Создаем общее полотно
        out_h = max(h1, h2)
        out_w = w1 + w2
        out_img = np.zeros((out_h, out_w), dtype=np.uint8)
        
        out_img[:h1, :w1] = img1
        out_img[:h2, w1:w1+w2] = img2
        
        out_imgColor = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        
        # Рисуем линии
        for p1, p2 in zip(pts1, pts2):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0] + w1), int(p2[1]))
            cv2.line(out_imgColor, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(out_imgColor, pt1, 2, (0, 0, 255), -1)
            cv2.circle(out_imgColor, pt2, 2, (0, 0, 255), -1)
            
        return out_imgColor
