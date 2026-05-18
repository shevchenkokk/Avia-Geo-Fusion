"""
Модуль для сопоставления кадров со спутниковой картой.

Поддерживаемые backend'ы:
- lightglue (если установлен пакет lightglue)
- loftr (kornia)
- orb (fallback без внешних нейросетевых зависимостей)

Возвращаемый интерфейс совместим с существующим кодом:
    {
      "mkpts0": np.ndarray[N,2],
      "mkpts1": np.ndarray[N,2],
      "debug_img": np.ndarray[H,W,3],
      "backend": str,
      "raw_matches": int,
      "inliers": int,
    }
"""

import os
from pathlib import Path

import cv2
import torch
import numpy as np
import logging
from typing import Any
try:
    import kornia.feature as KF

    LOFTR_AVAILABLE = True
except Exception:
    LOFTR_AVAILABLE = False

from src.aircraft_mask import apply_mask_to_image, filter_keypoints_by_mask
from src.segmentation import SkySegmenter

logger = logging.getLogger(__name__)

_XFEAT_REPO = "verlab/accelerated_features"
_XFEAT_CACHE_DIR = "verlab_accelerated_features_main"


def _torch_hub_dir() -> Path:
    return Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"

try:
    import lightglue as LG
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    LIGHTGLUE_AVAILABLE = True
except Exception:
    LG = None
    LIGHTGLUE_AVAILABLE = False

# XFeat (CVPR 2024) — основной матчер этапа 3.4. Загружается лениво через
# torch.hub при первом использовании бэкенда "xfeat".
XFEAT_AVAILABLE = True  # определяется в рантайме в _init_xfeat


class NeuralMatcher:
    """Матчер с backend'ами LightGlue/LoFTR/ORB."""

    def __init__(
        self,
        pretrained_model: str = "outdoor",
        use_segmentation: bool = True,
        backend: str = "auto",
        lightglue_features: str = "auto",
        lightglue_max_keypoints: int = 2048,
        max_size: int = 640,
        conf_thresh: float = 0.2,
        apply_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
    ):
        self.device = self._resolve_device()
        self.use_segmentation = use_segmentation
        self.segmenter = SkySegmenter() if use_segmentation else None
        self.max_size = max_size
        self.conf_thresh = conf_thresh
        self.backend_requested = backend.lower()
        self.lightglue_features_requested = lightglue_features.lower()
        self.lightglue_max_keypoints = int(lightglue_max_keypoints)
        self.backend = "uninitialized"
        self.lightglue_features_used = "none"
        self.apply_clahe = bool(apply_clahe)
        self._clahe = (
            cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
            if self.apply_clahe else None
        )

        self.loftr = None
        self.lg_extractor = None
        self.lg_matcher = None
        self.orb = None
        self.orb_bf = None
        self.xfeat = None

        self._init_backend(pretrained_model)
        logger.info(
            "NeuralMatcher backend=%s requested=%s lg_features=%s requested_lg=%s device=%s",
            self.backend,
            self.backend_requested,
            self.lightglue_features_used,
            self.lightglue_features_requested,
            self.device,
        )

    def _resolve_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_lightglue_extractor(self) -> tuple[Any | None, str]:
        if not LIGHTGLUE_AVAILABLE or LG is None:
            return None, "none"

        requested = self.lightglue_features_requested
        if requested == "auto":
            # Cохраняем обратную совместимость по умолчанию (SuperPoint),
            # но даем путь к более новым extractors без изменений API.
            candidates = ["superpoint", "aliked", "disk"]
        else:
            candidates = [requested]

        for name in candidates:
            try:
                if name == "superpoint":
                    extractor = SuperPoint(max_num_keypoints=self.lightglue_max_keypoints).eval().to(self.device)
                elif name == "aliked":
                    aliked_cls = getattr(LG, "ALIKED", None)
                    if aliked_cls is None:
                        continue
                    extractor = aliked_cls(max_num_keypoints=self.lightglue_max_keypoints).eval().to(self.device)
                elif name == "disk":
                    disk_cls = getattr(LG, "DISK", None)
                    if disk_cls is None:
                        continue
                    extractor = disk_cls(max_num_keypoints=self.lightglue_max_keypoints).eval().to(self.device)
                else:
                    continue
                return extractor, name
            except Exception as e:
                logger.warning("LightGlue extractor %s init failed: %s", name, e)

        return None, "none"

    def _init_xfeat(self) -> bool:
        """Ленивая загрузка XFeat через torch.hub. Возвращает True при успехе.

        XFeat принудительно запускается на CPU вне зависимости от ``self.device``:
        внутренний пайплайн модели перемещает тензоры между устройствами способами,
        ломающими Apple-MPS в текущих сборках torch.
        """
        try:
            repo = _torch_hub_dir() / _XFEAT_CACHE_DIR
            repo_or_dir = str(repo) if repo.exists() else _XFEAT_REPO
            source = "local" if repo.exists() else "github"
            self.xfeat = torch.hub.load(
                repo_or_dir,
                "XFeat",
                pretrained=True,
                top_k=4096,
                source=source,
                trust_repo=True,
                verbose=False,
                skip_validation=True,
            )
            self.xfeat = self.xfeat.cpu().eval()
            self.xfeat_device = torch.device("cpu")
            return True
        except Exception as e:
            logger.warning(f"XFeat init failed: {e}")
            self.xfeat = None
            return False

    def _init_backend(self, pretrained_model: str) -> None:
        requested = self.backend_requested

        if requested == "xfeat":
            if self._init_xfeat():
                self.backend = "xfeat"
                return
            logger.warning("XFeat requested but unavailable; falling through to LoFTR/ORB")

        if requested in ("auto", "lightglue") and LIGHTGLUE_AVAILABLE:
            try:
                self.lg_extractor, lg_name = self._resolve_lightglue_extractor()
                if self.lg_extractor is not None:
                    self.lg_matcher = LightGlue(features=lg_name).eval().to(self.device)
                    self.lightglue_features_used = lg_name
                else:
                    raise RuntimeError("No supported LightGlue extractor available")
                self.backend = "lightglue"
                return
            except Exception as e:
                logger.warning(f"LightGlue init failed, fallback: {e}")

        if requested in ("auto", "loftr", "lightglue") and LOFTR_AVAILABLE:
            try:
                self.loftr = KF.LoFTR(pretrained=pretrained_model).to(self.device).eval()
                self.backend = "loftr"
                return
            except Exception as e:
                logger.warning(f"LoFTR init failed, fallback to ORB: {e}")
        elif requested in ("loftr",):
            logger.warning("LoFTR backend requested, but kornia/LoFTR is unavailable. Fallback to ORB.")

        self.orb = cv2.ORB_create(nfeatures=2500)
        self.orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.backend = "orb"

    def _prepare_image(
        self,
        img: np.ndarray,
        is_drone: bool = False,
        aircraft_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, np.ndarray, float]:
        """Возвращает (тензор 1x1xH'xW', gray_original, scale).

        ``aircraft_mask`` (если задана) применяется только когда ``is_drone=True``
        и зануляет пиксели под маской ДО любых преобразований — текстуры там
        нет, поэтому ни LightGlue, ни LoFTR, ни ORB не выдадут keypoints
        внутри маски.
        """
        if is_drone and self.use_segmentation and self.segmenter is not None:
            img = self.segmenter.remove_sky(img)

        if img.ndim == 3:
            img_gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray_orig = img

        if self._clahe is not None:
            img_gray_orig = self._clahe.apply(img_gray_orig)

        if is_drone and aircraft_mask is not None:
            img_gray_orig = apply_mask_to_image(img_gray_orig, aircraft_mask)

        h, w = img_gray_orig.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_size:
            scale = self.max_size / float(max(h, w))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_resized = cv2.resize(img_gray_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_gray_orig

        tensor = torch.from_numpy(img_resized).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, img_gray_orig, scale

    def _pad_to_multiple(self, tensor: torch.Tensor, multiple: int = 8) -> torch.Tensor:
        _, _, h, w = tensor.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    def _match_loftr(self, tensor_drone: torch.Tensor, tensor_map: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.loftr is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        input_dict = {
            "image0": self._pad_to_multiple(tensor_drone),
            "image1": self._pad_to_multiple(tensor_map),
        }
        with torch.no_grad():
            matches_dict = self.loftr(input_dict)

        mkpts0 = matches_dict["keypoints0"].detach().cpu().numpy()
        mkpts1 = matches_dict["keypoints1"].detach().cpu().numpy()
        confidence = matches_dict["confidence"].detach().cpu().numpy()

        valid = confidence > self.conf_thresh
        return mkpts0[valid].astype(np.float32), mkpts1[valid].astype(np.float32)

    def _match_lightglue(self, tensor_drone: torch.Tensor, tensor_map: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.lg_extractor is None or self.lg_matcher is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        try:
            with torch.no_grad():
                feats0 = self.lg_extractor.extract(tensor_drone)
                feats1 = self.lg_extractor.extract(tensor_map)
                matches01 = self.lg_matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
            matches = matches01.get("matches", None)
            if matches is None or matches.numel() == 0:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

            mkpts0 = feats0["keypoints"][matches[:, 0]].detach().cpu().numpy().astype(np.float32)
            mkpts1 = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy().astype(np.float32)
            return mkpts0, mkpts1
        except Exception as e:
            logger.warning("LightGlue matching failed (%s): %s", self.lightglue_features_used, e)
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    def _match_xfeat(self, tensor_drone: torch.Tensor, tensor_map: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """XFeat dense matcher. Возвращает совпадения в пространстве уменьшенных пикселей
        (вызывающий масштабирует обратно через ``scale_drone`` / ``scale_map``).

        ``match_xfeat`` выполняет детекцию + матчинг за один проход и имеет свой
        RANSAC/ratio-test; мы всё равно применяем геометрический фильтр ниже по цепочке,
        так как он общий для всех backend'ов.
        """
        if self.xfeat is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        try:
            # match_xfeat ожидает (1, 3, H, W) RGB-тензоры в [0..1].
            # Наши tensor_* — (1, 1, H, W) grayscale; реплицируем до 3 каналов.
            # Принудительно CPU: XFeat ломается на Apple-MPS в текущих сборках torch.
            t0 = tensor_drone.cpu().repeat(1, 3, 1, 1)
            t1 = tensor_map.cpu().repeat(1, 3, 1, 1)
            with torch.no_grad():
                mkpts0, mkpts1 = self.xfeat.match_xfeat(t0, t1, top_k=4096)
            mkpts0 = np.asarray(mkpts0, dtype=np.float32)
            mkpts1 = np.asarray(mkpts1, dtype=np.float32)
            return mkpts0, mkpts1
        except Exception as e:
            logger.warning(f"XFeat matching failed: {e}")
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    def _match_orb(self, tensor_drone: torch.Tensor, tensor_map: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.orb is None or self.orb_bf is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        img0 = (tensor_drone.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
        img1 = (tensor_map.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)

        kp0, des0 = self.orb.detectAndCompute(img0, None)
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        if des0 is None or des1 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        knn = self.orb_bf.knnMatch(des0, des1, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.78 * n.distance:
                good.append(m)

        if not good:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        mkpts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
        mkpts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)
        return mkpts0, mkpts1

    def _geometric_filter(self, pts0: np.ndarray, pts1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RANSAC-фильтр: сначала гомография, затем запасной вариант через affine."""
        n = len(pts0)
        if n < 4:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

        mask = None
        try:
            _, hmask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
            if hmask is not None:
                mask = hmask.ravel().astype(bool)
        except Exception:
            mask = None

        if mask is None or int(mask.sum()) < 4:
            try:
                _, amask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if amask is not None:
                    mask = amask.ravel().astype(bool)
            except Exception:
                mask = None

        if mask is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.zeros(n, dtype=bool)

        return pts0[mask], pts1[mask], mask

    def match(
        self,
        img_drone: np.ndarray,
        img_map: np.ndarray,
        aircraft_mask: np.ndarray | None = None,
    ) -> dict:
        """Сопоставление кадра самолёта с тайлом карты.

        ``aircraft_mask`` (uint8 0/255, те же H×W, что img_drone) подавляет
        keypoints на корпусе самолёта. Маска применяется дважды:
        (1) зануление пикселей на входе — дёшево, блокирует большинство keypoints,
        (2) post-фильтрация оставшихся в исходном разрешении — гарантия корректности
            критерия Stage 1.2.
        """
        tensor_drone, gray_drone_orig, scale_drone = self._prepare_image(
            img_drone, is_drone=True, aircraft_mask=aircraft_mask
        )
        tensor_map, gray_map_orig, scale_map = self._prepare_image(img_map, is_drone=False)

        if self.backend == "lightglue":
            mkpts0, mkpts1 = self._match_lightglue(tensor_drone, tensor_map)
        elif self.backend == "loftr":
            mkpts0, mkpts1 = self._match_loftr(tensor_drone, tensor_map)
        elif self.backend == "xfeat":
            mkpts0, mkpts1 = self._match_xfeat(tensor_drone, tensor_map)
        else:
            mkpts0, mkpts1 = self._match_orb(tensor_drone, tensor_map)

        raw_matches = len(mkpts0)
        if raw_matches > 0:
            mkpts0 = (mkpts0 / scale_drone).astype(np.float32)
            mkpts1 = (mkpts1 / scale_map).astype(np.float32)

        masked_kept = -1
        if aircraft_mask is not None and raw_matches > 0:
            keep = filter_keypoints_by_mask(mkpts0, aircraft_mask)
            masked_kept = int(keep.sum())
            mkpts0 = mkpts0[keep]
            mkpts1 = mkpts1[keep]

        mkpts0_inliers, mkpts1_inliers, _ = self._geometric_filter(mkpts0, mkpts1)
        inliers = len(mkpts0_inliers)
        logger.info(
            f"Matcher[{self.backend}] raw={raw_matches} after_mask={masked_kept} inliers={inliers}"
        )

        debug_img = self._draw_matches(gray_drone_orig, gray_map_orig, mkpts0_inliers, mkpts1_inliers)
        return {
            "mkpts0": mkpts0_inliers,
            "mkpts1": mkpts1_inliers,
            "debug_img": debug_img,
            "backend": self.backend,
            "features": self.lightglue_features_used if self.backend == "lightglue" else "none",
            "raw_matches": raw_matches,
            "matches_after_mask": masked_kept if masked_kept >= 0 else raw_matches,
            "inliers": inliers,
        }

    def _draw_matches(self, img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        out_h = max(h1, h2)
        out_w = w1 + w2
        out_img = np.zeros((out_h, out_w), dtype=np.uint8)
        out_img[:h1, :w1] = img1
        out_img[:h2, w1 : w1 + w2] = img2
        out_img_color = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)

        for p1, p2 in zip(pts1, pts2):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0] + w1), int(p2[1]))
            cv2.line(out_img_color, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(out_img_color, pt1, 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(out_img_color, pt2, 2, (0, 0, 255), -1, cv2.LINE_AA)

        return out_img_color
