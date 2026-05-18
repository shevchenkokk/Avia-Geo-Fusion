# Avia-Geo-Fusion

Проект посвящён визуальной геолокализации лёгкого воздушного судна по бортовому видео и заранее подготовленной спутниковой карте. Система совмещает относительную оценку движения по видео с периодическими абсолютными привязками к карте, ведёт состояние в расширенном фильтре Калмана и явно фиксирует причины отказа на сложных кадрах.

Репозиторий содержит исследовательский прототип, воспроизводимые проверки отдельных модулей, скрипты для внешних датасетов и документацию по экспериментам. Тяжёлые видео, веса моделей и результаты прогонов в git не добавляются.

## Что реализовано

- выпрямление широкоугольного кадра и преобразование к виду сверху;
- маска корпуса воздушного судна и детектор закрытия обзора;
- оптико-потоковая одометрия для относительного движения;
- поиск кандидатных спутниковых тайлов по дескрипторам DINOv2;
- локальное сопоставление XFeat, LightGlue, LoFTR или ORB с оценкой гомографии;
- структурное сопоставление с векторными слоями Overture;
- семантическая фильтрация SegFormer, дообученного на Overture-RU;
- расширенный фильтр Калмана, гейт согласованности и автомат расписания каналов;
- отчёты, графики траектории, диагностические журналы и проверочные скрипты.

## Структура

```text
configs/     конфигурации камер, датасетов и запусков
data/        небольшие воспроизводимые артефакты; крупные данные лежат локально
docs/        описание архитектуры, экспериментов и готовности валидации
scripts/     основные запускные и проверочные скрипты
src/         программные модули пайплайна
results/     локальные результаты прогонов, не коммитятся
```

Главный интеграционный запуск находится в `scripts/run_full_pipeline.py`. Старые визуальные и диагностические запуски оставлены в `scripts/pipeline/`, ручные исследовательские утилиты — в `scripts/manual/`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Для геоданных и семантической сегментации нужны дополнительные зависимости:

```bash
pip install -e ".[geo]"
pip install -e ".[semantic]"
```

Для работы с видео должен быть установлен `ffmpeg`. На macOS:

```bash
brew install ffmpeg
```

## Быстрая проверка без полётного видео

Если локальные видео недоступны, можно проверить воспроизводимые части системы на синтетике и подготовленных метаданных.

```bash
python scripts/run_validation_pack.py --output results/validation_pack
```

Команда запускает набор коротких проверок и сохраняет сводку в `results/validation_pack`. По умолчанию она не требует реального видео.

Отдельно можно проверить структурное сопоставление по Overture-векторам:

```bash
python scripts/verify_structural_matching.py \
  --dataset-root data/overture_ru_dataset_starter \
  --region-id moscow_city_small \
  --output results/stage5_structural
```

Эта проверка строит синтетический вид сверху по векторным классам, смещает стартовую точку и проверяет, что структурный канал возвращает позицию рядом с известным центром.

## Полный запуск на локальном видео

Для основного пайплайна нужны локальные артефакты:

```text
data/videos/GP010269.MP4
configs/camera_gopro_hx.yaml
data/masks/anchors/index.json
data/retrieval/db_z14.npz
data/retrieval/db_z14.json
```

Пример запуска:

```bash
python scripts/run_full_pipeline.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --dem data/dem/test_synthetic.tif \
  --start-s 45 \
  --end-s 90 \
  --backend xfeat \
  --snow-mask \
  --output results/full_pipeline
```

На выходе создаются:

- `state.csv` — покадровое состояние фильтра, причины отказов, принятые измерения и времена этапов;
- `summary.md` и `summary.json` — краткая сводка запуска;
- `trajectory.png` — траектория и принятые картографические фиксации;
- `scale_history.png` — история поправки масштаба визуальной одометрии.

Для включения структурного канала:

```bash
python scripts/run_full_pipeline.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --structural-match \
  --structural-vectors-root data/overture_ru_dataset_starter/vectors \
  --structural-region-id kolomna_cruise \
  --output results/full_pipeline_structural
```

Для включения семантического структурного сопоставления:

```bash
python scripts/run_full_pipeline.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --semantic-structural-match \
  --structural-vectors-root data/overture_ru_dataset_starter/vectors \
  --structural-region-id kolomna_cruise \
  --output results/full_pipeline_semantic_structural
```

## Проверки отдельных компонентов

```bash
python scripts/verify_aircraft_mask.py --video data/videos/GP010269.MP4
python scripts/verify_obstruction_detector.py --video data/videos/GP010269.MP4 --output-dir results/stage1_5
python scripts/verify_dem_lookup.py --dem data/dem/test_synthetic.tif --output results/stage2_1
python scripts/verify_optical_flow_vo.py --camera-config configs/camera_gopro_hx.yaml
python scripts/verify_retriever.py --db data/retrieval/db_z14
python scripts/verify_ekf.py
```

Для быстрой сквозной проверки без полного фильтра:

```bash
python scripts/end_to_end_smoke.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --output results/end_to_end
```

## Валидация на внешних датасетах

VPair используется как проверка слоя грубого поиска места:

```bash
python scripts/download_vpair_sample.py --output-root data/external/vpair_sample
python scripts/prepare_vpair.py \
  --dataset-root data/external/vpair_sample \
  --output data/vpair/manifests
python scripts/evaluate_vpair_vpr.py \
  --dataset-root data/external/vpair_sample \
  --manifest-dir data/vpair/manifests \
  --output results/vpair_vpr \
  --include-distractors
```

AerialVL используется для проверки полного контура с эталонными траекториями:

```bash
python scripts/prepare_aerialvl.py \
  --dataset-root data/external/aerialvl \
  --output data/aerialvl/manifests
python scripts/run_aerialvl_pipeline.py \
  --dataset-root data/external/aerialvl \
  --manifest-dir data/aerialvl/manifests \
  --output results/aerialvl_full_pipeline \
  --backend xfeat
```

Графики для экспериментальной части собираются ручными скриптами из `scripts/manual/`, например:

```bash
python scripts/manual/render_final_two_datasets.py
```

## Сборка Overture-RU

Датасет спутниковых тайлов и масок по российским регионам собирается по конфигурации:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py \
  --config configs/overture_dataset_ru.json \
  --stage all
```

Этапы можно запускать по отдельности:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage download
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage rasterize
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage qc
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage patches
```

Подробности описаны в `docs/overture_dataset_pipeline.md`.

## Docker

Для запуска без локального виртуального окружения можно собрать контейнер:

```bash
docker compose build
docker compose run --rm avia-geo-fusion python scripts/run_full_pipeline.py --help
```

Видео, результаты и пользовательские артефакты не вшиваются в образ. Рабочая директория монтируется внутрь контейнера, поэтому локальные файлы доступны по тем же путям.

## Что не хранится в git

- исходные видеозаписи;
- веса моделей и чекпоинты;
- каталоги `results/`;
- локальные выгрузки внешних датасетов;
- тяжёлые сгенерированные изображения, маски и промежуточные продукты Overture.

Политика артефактов описана в `docs/artifacts.md`, логическая структура модулей — в `docs/repository_structure.md`.
