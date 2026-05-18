"""Визуальный ретривер (DINOv2-B + косинусный NN).

Решает задачу бутстрапа: в начале полёта (или после потери трека) нужна
грубая абсолютная позиция для центрирования тайлового окна MapManager
до того, как локальный матчер сможет сойтись.

Архитектура (CLS-токен напрямую, без k-means VLAD):
  - референсная БД: (N, 768) float32 дескрипторы + (N, 2) (lat, lon)
  - запрос: изменить размер/центрировать кроп → 224×224 → DINOv2 → CLS (768) → L2-норм.
  - поиск: np.dot(query, ref.T) → топ-K по косинусному сходству
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


_DINOV2_REPO = "facebookresearch/dinov2"
_DINOV2_ENTRY = "dinov2_vitb14"
_DESCRIPTOR_DIM = 768
_INPUT_SIZE = 224  # сетка 16×16 патчей при patch_size=14
_DINOV2_CACHE_DIR = "facebookresearch_dinov2_main"

# Нормировка ImageNet (ожидаемая статистика DINOv2).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _select_device(prefer: str = "auto") -> str:
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_hub_dir() -> Path:
    return Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"


def _load_dinov2(model_name: str) -> torch.nn.Module:
    cache_dir = _torch_hub_dir() / _DINOV2_CACHE_DIR
    if cache_dir.exists():
        return torch.hub.load(
            str(cache_dir),
            model_name,
            source="local",
            verbose=False,
        )
    return torch.hub.load(
        _DINOV2_REPO,
        model_name,
        verbose=False,
        trust_repo=True,
        skip_validation=True,
    )


@dataclass
class ReferenceDatabase:
    """Хранит (дескриптор, lat, lon, tile_id) для N тайлов.

    Формат на диске: .npz с массивами + .json с строковыми метаданными.
    """
    descriptors: np.ndarray       # (N, 768) float32, L2-нормированные
    centers_latlon: np.ndarray    # (N, 2) float64 [lat, lon]
    tile_ids: list[str]           # ["{z}/{x}/{y}", ...]
    zoom: int
    model_name: str = _DINOV2_ENTRY
    input_size: int = _INPUT_SIZE

    def __len__(self) -> int:
        return len(self.descriptors)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            descriptors=self.descriptors,
            centers_latlon=self.centers_latlon,
        )
        meta = {
            "tile_ids": self.tile_ids,
            "zoom": int(self.zoom),
            "model_name": self.model_name,
            "input_size": int(self.input_size),
            "n_entries": int(len(self)),
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "ReferenceDatabase":
        path = Path(path)
        npz = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            descriptors=npz["descriptors"].astype(np.float32),
            centers_latlon=npz["centers_latlon"].astype(np.float64),
            tile_ids=list(meta["tile_ids"]),
            zoom=int(meta["zoom"]),
            model_name=meta["model_name"],
            input_size=int(meta["input_size"]),
        )

    def query(self, descriptor: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Косинусный NN-поиск. Возвращает [(db_index, score), ...] по убыванию.

        Оба дескриптора L2-нормированы, поэтому косинусное сходство = скалярное произведение.
        """
        scores = self.descriptors @ descriptor
        order = np.argsort(-scores)
        out = []
        for i in order[:top_k]:
            out.append((int(i), float(scores[i])))
        return out


class Retriever:
    """Кодировщик изображений DINOv2-B + поиск в референсной БД."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = _DINOV2_ENTRY,
        input_size: int = _INPUT_SIZE,
    ) -> None:
        self.device = _select_device(device)
        self.input_size = int(input_size)
        self.model_name = model_name

        self.model = _load_dinov2(model_name).to(self.device).eval()
        self._db: Optional[ReferenceDatabase] = None

    def load_database(self, path: Path | str) -> None:
        self._db = ReferenceDatabase.load(path)
        if self._db.model_name != self.model_name:
            raise ValueError(
                f"DB built with {self._db.model_name}, retriever uses {self.model_name}"
            )

    @property
    def database(self) -> Optional[ReferenceDatabase]:
        return self._db

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H, W, 3) → float32 тензор (3, S, S) на устройстве.

        Центральный кроп до квадрата, ресайз до input_size.
        DINOv2 требует кратность 14; стандарт: 224 = 16*14.
        """
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img_bgr[y0:y0 + s, x0:x0 + s]
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_rgb = cv2.resize(img_rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        img_rgb = (img_rgb - _MEAN) / _STD
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).contiguous()

    @torch.no_grad()
    def encode(self, img_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(img_bgr).unsqueeze(0).to(self.device)
        out = self.model(x)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    @torch.no_grad()
    def encode_batch(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        if len(imgs_bgr) == 0:
            return np.empty((0, _DESCRIPTOR_DIM), dtype=np.float32)
        batch = torch.stack([self._preprocess(im) for im in imgs_bgr]).to(self.device)
        out = self.model(batch)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.detach().cpu().numpy().astype(np.float32)

    def query(self, descriptor: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        if self._db is None:
            raise RuntimeError("Retriever has no database loaded; call load_database() first")
        return self._db.query(descriptor, top_k=top_k)

    def query_image(self, img_bgr: np.ndarray, top_k: int = 5) -> list[dict]:
        """Полный проход: кодирование + косинусный NN. Возвращает список словарей
        с ключами ``index``, ``score``, ``lat``, ``lon``, ``tile_id``.
        """
        if self._db is None:
            raise RuntimeError("Retriever has no database loaded; call load_database() first")
        d = self.encode(img_bgr)
        out = []
        for idx, score in self.query(d, top_k=top_k):
            lat, lon = self._db.centers_latlon[idx]
            out.append({
                "index": idx,
                "score": score,
                "lat": float(lat),
                "lon": float(lon),
                "tile_id": self._db.tile_ids[idx],
            })
        return out
