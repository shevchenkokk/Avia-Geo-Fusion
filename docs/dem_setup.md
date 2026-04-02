# Подготовка DEM (этап 2.1)

Модуль `src/dem_lookup.py` не привязан к конкретному источнику DEM: подойдёт любой GeoTIFF в EPSG:4326 с высотами в метрах над уровнем моря (MSL).

Чтобы заменить синтетический DEM на реальный рельеф под другую миссию, обновите `configs/mission.yaml` и укажите новый путь в `dem.path`.

Проверка `scripts/verify_dem_lookup.py` по умолчанию использует синтетический DEM, потому что только в этом случае можно сравнить результат интерполяции с аналитически известным ground truth. Реальный SRTM нужен для рантайма, синтетика остаётся в репозитории как регрессионный тест.

## Рекомендуемые источники

| Источник | Разрешение | Авторизация | Примечание |
|---|---|---|---|
| **AWS Open Data (Mapzen -> SRTMGL1)** | ~30 м | Нет | `s3.amazonaws.com/elevation-tiles-prod/skadi/N{lat}/N{lat}E{lon}.hgt.gz` — используется в проекте |
| SRTM 1-arc-sec (NASA / USGS) | ~30 м | Earthdata login | Оригинальный источник (по сути те же данные, что в AWS-зеркале) |
| SRTM 3-arc-sec (CGIAR-CSI mirror) | ~90 м | Нет | Прямые GeoTIFF-загрузки, тайлы 5x5 градусов |
| ALOS AW3D30 (JAXA) | ~30 м | Бесплатная регистрация | Обычно лучше SRTM в горном рельефе |
| Copernicus DEM GLO-30 | ~30 м | Учётная запись Copernicus | Наиболее качественный вариант из открытых |

## Покрытие для миссии GP010269

Стартовая точка полёта: `(55.086, 38.149)` (Коломна, центральная Россия). Бокс миссии попадает в один SRTM-тайл 1x1 градус: `N55E038`.

## Процедура, используемая в репозитории

```bash
# 1. Скачать SRTM 1-arc-sec HGT с AWS Open Data (без авторизации).
curl -o data/dem/N55E038.hgt.gz \
  https://s3.amazonaws.com/elevation-tiles-prod/skadi/N55/N55E038.hgt.gz
gunzip data/dem/N55E038.hgt.gz

# 2. Конвертировать HGT (raw big-endian int16) в GeoTIFF EPSG:4326.
python scripts/srtm_hgt_to_geotiff.py data/dem/N55E038.hgt \
  --out data/dem/srtm_n55_e038.tif

# 3. Быстрая sanity-проверка lookup.
python -c "from src.dem_lookup import DemLookup; \
           d = DemLookup('data/dem/srtm_n55_e038.tif'); \
           print(d.elevation(55.086, 38.149))"
# Ожидаемо около 170 м MSL возле точки старта.
```

В текущем состоянии `configs/mission.yaml` уже указывает на этот DEM. Для другой миссии смените имя HGT/GeoTIFF и обновите `bbox`.

Если DEM пришёл в другой системе координат, предварительно перепроецируйте его:

```bash
gdalwarp -t_srs EPSG:4326 input.tif output_4326.tif
```

## Для чего нужна синтетика

Синтетический DEM не заменяет реальные высоты. Его задача — строгая численная валидация `verify_dem_lookup.py`: аналитический рельеф (наклон + Gaussian hill) позволяет сравнить билинейную интерполяцию с закрытой формулой до субметровой точности.

Для рабочего пайплайна нужен реальный DEM: на этапе 2.2 optical-flow VO должен получать корректный AGL, чтобы переводить пиксельные скорости в метрические.
