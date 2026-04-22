"""
Единая схема классов семантической сегментации.

В проекте используется собственный 5-классный датасет Overture-RU,
собранный под РФ-ландшафты. Этот модуль — единственное место, где
определены ID классов. Все сегментаторы и дебаг-оверлеи берут
имя/цвет/роль/флаг стабильности отсюда, чтобы исключить рассинхрон
ID между рантаймом и чекпойнтом SegFormer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClassRole = Literal[
    "background", "building", "road", "water", "vegetation",
    "forest_edge",     # морф. граница лес↔поле, основной structural feature в тайге
    "field_boundary",  # геом. границы сельхозполей, видны летом
    "built_up",        # городская застройка как площадь (а не отдельные здания)
    "snow",            # masker-класс — keypoints поверх снега дропаются
    "cloud",           # masker-класс — keypoints в облаках дропаются
]


@dataclass(frozen=True)
class ClassEntry:
    id: int
    name: str
    color: tuple[int, int, int]  # BGR
    role: ClassRole
    stable: bool       # использовать ли класс при матчинге кадр↔карта
    masker: bool = False    # masker-класс (§4): keypoints в зоне дропаются
    weight: float = 1.0     # вес класса в appearance-матчере (§4.4)


@dataclass(frozen=True)
class ClassSchema:
    name: str
    entries: tuple[ClassEntry, ...]

    def by_id(self) -> dict[int, ClassEntry]:
        return {e.id: e for e in self.entries}

    @property
    def stable_ids(self) -> set[int]:
        return {e.id for e in self.entries if e.stable}

    @property
    def colors(self) -> dict[int, tuple[int, int, int]]:
        return {e.id: e.color for e in self.entries}

    @property
    def names(self) -> dict[int, str]:
        return {e.id: e.name for e in self.entries}

    def role_ids(self, role: ClassRole) -> set[int]:
        return {e.id for e in self.entries if e.role == role}


# Overture-RU 5-классная схема (устарела, до v1.4).
# Сохранена для существующих чекпойнтов SegFormer-B0. Новая работа — OVERTURE_RU_7.
OVERTURE_RU_5 = ClassSchema(
    name="overture_ru_5",
    entries=(
        ClassEntry(0, "background", (80, 80, 80), "background", stable=False),
        ClassEntry(1, "water", (230, 120, 0), "water", stable=False),
        ClassEntry(2, "vegetation", (40, 170, 40), "vegetation", stable=True),
        ClassEntry(3, "buildings", (0, 140, 255), "building", stable=True),
        ClassEntry(4, "roads", (0, 255, 0), "road", stable=True),
    ),
)


# Overture-RU 7-классная схема — целевая (PROJECT_PLAN.md v1.4 §4).
#
# Приоритеты под российскую географию:
#   1. water           — устойчив круглый год, сильный структурный якорь
#   2. forest_edge     — граница лес↔поле, основной структурный признак на 50%+ РФ
#   3. roads           — Overture/OSM в буфере; зимой = расчищенный снег
#   4. field_boundary  — границы сельхозполей, лето
#   5. built_up        — города как площадь (не отдельные здания)
#   maskers: snow, cloud — дропают keypoints в матчере
#
# Веса классов (§4.4):
#   roads, forest_edge, built_up, field_boundary  -> 2.0
#   water                                          -> 0.5 (риск: лёд = неотличимо от берега)
#   snow, cloud                                    -> 0.0 (дроп)
#   background                                     -> 1.0
OVERTURE_RU_7 = ClassSchema(
    name="overture_ru_7",
    entries=(
        ClassEntry(0, "background",     (80, 80, 80),     "background",     stable=False, weight=1.0),
        ClassEntry(1, "water",          (230, 120, 0),    "water",          stable=True,  weight=0.5),
        ClassEntry(2, "forest_edge",    (40, 170, 40),    "forest_edge",    stable=True,  weight=2.0),
        ClassEntry(3, "roads",          (0, 255, 0),      "road",           stable=True,  weight=2.0),
        ClassEntry(4, "field_boundary", (0, 200, 200),    "field_boundary", stable=True,  weight=2.0),
        ClassEntry(5, "built_up",       (0, 140, 255),    "built_up",       stable=True,  weight=2.0),
        ClassEntry(6, "snow",           (255, 255, 255),  "snow",           stable=False, masker=True, weight=0.0),
        # cloud генерится per-frame детектором §1.5, а не из векторных данных.
    ),
)


# Порядок каналов структурного матчера (§4.5): вода первой как наиболее устойчивый якорь.
STRUCTURAL_PRIORITY_IDS = (1, 2, 3, 4, 5)  # water, forest_edge, roads, field_boundary, built_up
