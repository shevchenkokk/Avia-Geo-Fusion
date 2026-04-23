import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from mmseg.apis import inference_model, init_model

    MMSEG_AVAILABLE = True
except Exception:
    MMSEG_AVAILABLE = False

from src.segmentation import SkySegmenter
from src.class_schemas import ClassSchema, OVERTURE_RU_5

# Фиктивный class_id для эвристического бэкенда: все найденные «земля»-пиксели
# помечаются одним не-нулевым классом. Позволяет единообразно отдавать
# per-class карту в обоих режимах (mmseg / heuristic).
HEURISTIC_GROUND_ID = 255


class GeoSegmentor:
    """Формирует семантическую карту «стабильной» земли для матчинга кадр↔карта.

    Два режима работы:
      - backend='mmseg': настоящая per-class карта по схеме `schema`
        (в бою — OVERTURE_RU_5: 0 bg, 1 water, 2 veg, 3 buildings, 4 roads).
      - backend='heuristic': бинарная маска «не-небо + текстурно» под одним
        синтетическим классом HEURISTIC_GROUND_ID. Нужна только как fallback.

    Два выхода:
      - `stable_class_map(frame)` — uint8 HxW, 0 на не-стабильных пикселях,
        иначе id из схемы (или HEURISTIC_GROUND_ID). Основной выход: по нему
        строится и бинарь, и class-aware фильтр matches, и диагностика HUD.
      - `stable_ground_mask(frame)` — uint8 HxW со значениями {0, 255}.
        Оставлен для обратной совместимости с рендер-кодом.
    """

    def __init__(
        self,
        top_crop_ratio: float = 0.35,
        texture_percentile: float = 60.0,
        backend: str = "heuristic",
        mmseg_config: str | None = None,
        mmseg_checkpoint: str | None = None,
        mmseg_device: str = "cuda:0",
        schema: ClassSchema = OVERTURE_RU_5,
        stable_class_ids: set[int] | None = None,
    ):
        self.sky_segmenter = SkySegmenter()
        self.top_crop_ratio = top_crop_ratio
        self.texture_percentile = texture_percentile
        self.backend_requested = str(backend).lower()
        self.backend = "heuristic"

        self.mmseg_model = None
        self.mmseg_config = mmseg_config
        self.mmseg_checkpoint = mmseg_checkpoint
        self.mmseg_device = mmseg_device
        self.schema = schema
        self.stable_class_ids = stable_class_ids if stable_class_ids is not None else schema.stable_ids

        if self.backend_requested == "mmseg":
            self._try_init_mmseg()

    def _try_init_mmseg(self) -> None:
        if not MMSEG_AVAILABLE:
            logger.warning("GeoSegmentor: mmseg backend requested, but mmseg is unavailable. Fallback to heuristic.")
            self.backend = "heuristic"
            return

        if not self.mmseg_config or not self.mmseg_checkpoint:
            logger.warning("GeoSegmentor: mmseg backend requested, but cfg/checkpoint not provided. Fallback to heuristic.")
            self.backend = "heuristic"
            return

        try:
            # mmseg-чекпоинт от training-рана содержит mmengine HistoryBuffer и
            # прочее training-state; torch>=2.6 по умолчанию weights_only=True
            # и на такой сериализации падает. Свой файл — доверенный источник.
            import torch
            _orig_torch_load = torch.load
            def _trusted_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)
            torch.load = _trusted_load
            try:
                self.mmseg_model = init_model(self.mmseg_config, self.mmseg_checkpoint, device=self.mmseg_device)
            finally:
                torch.load = _orig_torch_load
            self.backend = "mmseg"
            logger.info(
                "GeoSegmentor: using mmseg backend cfg=%s ckpt=%s device=%s schema=%s stable_classes=%s",
                self.mmseg_config,
                self.mmseg_checkpoint,
                self.mmseg_device,
                self.schema.name,
                sorted(self.stable_class_ids),
            )
        except Exception as e:
            logger.warning("GeoSegmentor: failed to init mmseg backend: %s. Fallback to heuristic.", e)
            self.mmseg_model = None
            self.backend = "heuristic"

    # ------------------------------------------------------------------
    # Основной выход: per-class карта стабильных классов
    # ------------------------------------------------------------------

    def stable_class_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Возвращает HxW uint8 с id стабильного класса на каждом пикселе.
        Не-стабильные пиксели зануляются (0).
        """
        if self.backend == "mmseg" and self.mmseg_model is not None:
            return self._class_map_mmseg(frame_bgr)
        return self._class_map_heuristic(frame_bgr)

    def stable_ground_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Бинарная маска (0/255) стабильной земли. Обёртка над class_map."""
        cmap = self.stable_class_map(frame_bgr)
        return np.where(cmap > 0, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # mmseg backend: настоящая сегментация по схеме
    # ------------------------------------------------------------------

    def _class_map_mmseg(self, frame_bgr: np.ndarray) -> np.ndarray:
        pred = inference_model(self.mmseg_model, frame_bgr)
        raw = pred.pred_sem_seg.data[0].detach().cpu().numpy().astype(np.uint8)

        # Оставляем только стабильные классы, остальное — ноль.
        stable_mask = np.isin(raw, list(self.stable_class_ids))
        class_map = np.where(stable_mask, raw, 0).astype(np.uint8)

        # Обрезаем верх кадра (небо/горизонт на наклонной камере) — только для самолёта,
        # для ортофото top_crop_ratio=0 задаётся вызывающим кодом.
        h = class_map.shape[0]
        crop_y = int(h * self.top_crop_ratio)
        if crop_y > 0:
            class_map[:crop_y, :] = 0

        # Морфология по ролям: road тонкий — только CLOSE, остальное — OPEN+CLOSE.
        road_ids = self.schema.role_ids("road")
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        cleaned = np.zeros_like(class_map)
        for cid in self.stable_class_ids:
            cmask = (class_map == cid).astype(np.uint8) * 255
            if cid in road_ids:
                cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kernel_small)
            else:
                cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN, kernel_mid)
                cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kernel_mid)
            cleaned[cmask > 0] = cid

        return cleaned

    # ------------------------------------------------------------------
    # Heuristic backend: нет семантики — единый фиктивный класс
    # ------------------------------------------------------------------

    def _class_map_heuristic(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, _w = frame_bgr.shape[:2]

        no_sky = self.sky_segmenter.remove_sky(frame_bgr)
        non_sky_mask = np.any(no_sky > 0, axis=2).astype(np.uint8) * 255

        crop_y = int(h * self.top_crop_ratio)
        non_sky_mask[:crop_y, :] = 0

        gray = cv2.cvtColor(no_sky, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_strength = np.abs(lap)

        nz = texture_strength[non_sky_mask > 0]
        if nz.size == 0:
            return np.zeros_like(non_sky_mask)

        thr = np.percentile(nz, self.texture_percentile)
        texture_mask = (texture_strength >= thr).astype(np.uint8) * 255

        stable = cv2.bitwise_and(non_sky_mask, texture_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stable = cv2.morphologyEx(stable, cv2.MORPH_OPEN, kernel)
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, kernel)

        return np.where(stable > 0, HEURISTIC_GROUND_ID, 0).astype(np.uint8)
