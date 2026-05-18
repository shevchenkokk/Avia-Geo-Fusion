import cv2
import torch
import numpy as np
import logging
from PIL import Image
import os

logger = logging.getLogger(__name__)


class UNetFormerSegmenter:
    """
    Legacy-бенчмарк: UNetFormer (Wang et al., ISPRS 2022), предобученный на LoveDA.
    В боевом пайплайне не используется — служит для сравнительных замеров
    в таблицах ВКР (см. docs/predefense_presentation_plan.md, слайд 6).

    Выход модели — канонический LoveDA (0=bg, 1=building, 2=road, ...), поэтому
    стабильные классы зафиксированы здесь как {1, 2}. На общий OVERTURE_RU_5
    этот модуль не мапится — у него принципиально другая вселенная меток.
    """

    # Канонический LoveDA: building=1, road=2 — то, что нужно для нав-матчинга.
    LOVEDA_STABLE_IDS = {1, 2}
    LOVEDA_NUM_CLASSES = 7

    # ADE20K-ID стабильных «наземных» классов — только для аварийного
    # отката на nvidia/segformer-b0-finetuned-ade-512-512, если чекпойнт
    # UNetFormer не найден.
    ADE20K_FALLBACK_STABLE_IDS = {1, 6, 16, 20, 25, 38, 42, 45, 59, 93}

    def __init__(self, weights_path: str = "weights/unetformer_loveda.pth"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Init UNetFormer (legacy) device={self.device}")

        from src.unetformer import UNetFormer

        try:
            self.model = UNetFormer(num_classes=self.LOVEDA_NUM_CLASSES).to(self.device).eval()

            if os.path.exists(weights_path):
                logger.info(f"Загрузка весов UNetFormer из {weights_path}...")
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                logger.warning(f"Веса {weights_path} не найдены! Fallback на ADE20K-SegFormer.")
                logger.warning("Скачайте веса отсюда: https://github.com/WangLibo1995/GeoSeg")
                self.model = None
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
                self.feature_extractor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                self.model_fallback = (
                    SegformerForSemanticSegmentation
                    .from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
                    .to(self.device).eval()
                )
        except Exception as e:
            logger.error(f"Не удалось инициализировать UNetFormer: {e}")
            self.model = None

    def get_stable_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        if self.model is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model_fallback(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            predicted_classes = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            stable_ids = self.ADE20K_FALLBACK_STABLE_IDS
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor_img = tensor_img.to(self.device)
            with torch.no_grad():
                logits = self.model(tensor_img)
            predicted_classes = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            stable_ids = self.LOVEDA_STABLE_IDS

        mask = np.zeros((h, w), dtype=np.uint8)
        for class_id in stable_ids:
            mask[predicted_classes == class_id] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def blend_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.3) -> np.ndarray:
        colored_mask = np.zeros_like(frame_bgr)
        colored_mask[mask == 255] = color
        return cv2.addWeighted(frame_bgr, 1.0, colored_mask, alpha, 0)
