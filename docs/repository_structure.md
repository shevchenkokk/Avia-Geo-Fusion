# Структура репозитория

Сейчас runtime-пакет остаётся в каталоге `src`, потому что большинство скриптов импортируют модули в формате `from src.<module> import ...`.

Рекомендуемая логическая группировка:

- `src/video_processor.py`, `src/undistort.py`, `src/bev_rectifier.py` — ввод видео, модель камеры и геометрия BEV.
- `src/aircraft_mask.py`, `src/obstruction_detector.py`, `src/snow_mask.py` — препроцессинг через маски и защитные гейты.
- `src/optical_flow_vo.py`, `src/ekf.py`, `src/frame_bridge.py` — VO, преобразования координат и EKF-фьюжн.
- `src/map_loader.py`, `src/map_manager.py`, `src/dem_lookup.py`, `src/retriever.py` — инфраструктура карт, DEM и retrieval.
- `src/neural_matching.py`, `src/map_measurement.py`, `src/structural_matcher.py`, `src/match_filters.py` — appearance/structural matching и конвертация в измерения фильтра.
- `src/geo_segmentor.py`, `src/segmentation.py`, `src/unetformer*.py`, `src/class_schemas.py` — утилиты семантической сегментации.
- `src/geolocator*.py`, `src/ipm.py`, `src/feature_matching.py` — legacy/диагностические модули для сравнения.

До стабилизации критичного для ВКР пайплайна не рекомендуется переносить эти модули во вложенные пакеты. Безопасный следующий шаг рефакторинга: ввести подкаталоги вроде `src/vision/`, `src/maps/`, `src/fusion/`, `src/matching/`, затем обновить импорты и в том же коммите прогнать полный smoke-набор.
