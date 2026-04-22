# Overture Dataset Pipeline (RU Regions, Zoom 17)

Этот пайплайн строит датасет под семантическую сегментацию SegFormer:

- спутниковые тайлы (`images/tiles`),
- маски из Overture (`masks/tiles`),
- QC-отчет (`qc_report.json`),
- тренировочные патчи `512x512` (`images/masks patches_512`),
- метаданные (`meta.csv`, `meta_patches.csv`, `splits.json`, `class_map.json`).

## Конфиг

Используется файл: `configs/overture_dataset_ru.json`.

Ключевые параметры:

- `zoom`: 17 (зафиксирован по задаче),
- `regions`: 8 регионов РФ под все нужные типы местности,
- `overture_layers`: water, land_cover, land_use, buildings, transportation,
- `class_mapping`: целевая схема классов 0..5,
- `raster_priority`: `[5,4,1,2,3,0]`.

## Установка зависимостей

```bash
pip install -e .
```

Если не установлен CLI Overture:

```bash
pip install overturemaps
```

## Запуск

Полный прогон:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage all
```

Этапами:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage download
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage rasterize
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage qc
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage patches
```

## Ожидаемая структура

```text
data/overture_ru_dataset/
  images/
    tiles/
    patches_512/
  masks/
    tiles/
    patches_512/
  vectors/
    <region_id>/
      water.parquet
      land_cover.parquet
      land_use.parquet
      buildings.parquet
      transportation.parquet
  meta_tiles.csv
  meta.csv
  meta_patches.csv
  splits.json
  class_map.json
  qc_report.json
  segformer_augmentations.json
```

## Как использовать для SegFormer

1. Берите `meta_patches.csv`, фильтруйте `split=train/val/test`.
2. Для `train` применяйте аугментации из `segformer_augmentations.json`.
3. Карты классов использовать из `class_map.json`.
4. Для контроля смещений и дублей проверяйте `qc_report.json`.

## Важно

- Команды `overturemaps download` в конфиге уже заданы, но их флаги могут отличаться в вашей версии CLI. Если команда не выполняется, поправьте только строки в `overture_download_commands`.
- Для первых итераций лучше запускать 1-2 региона и проверять качество оверлея маска/тайл перед полным прогоном.
