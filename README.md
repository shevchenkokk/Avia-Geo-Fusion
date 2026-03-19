# Avia-Geo-Fusion

Совмещение спутниковых карт и бортового видео для уточнения геолокализации в задачах БПЛА.

## Текущее состояние репозитория

- Canonical integration runner: `scripts/run_full_pipeline.py`.
- Legacy/HUD/diagnostic runners лежат в `scripts/pipeline/`.
- Component verify/smoke скрипты лежат в `scripts/`.
- Видео хранится в `data/videos/`.
- Предподготовленные локальные артефакты для `GP010269.MP4`: `configs/camera_gopro_hx.yaml`, `data/masks/anchors/`, `data/retrieval/db_z14.*`.

## Структура

```text
Avia-Geo-Fusion/
  configs/
  data/
    videos/
  docs/
  notebooks/
  results/
  scripts/
    run_full_pipeline.py
    end_to_end_smoke.py
    build_retrieval_db.py
    recover_intrinsics.py
    seed_aircraft_masks.py
    verify_obstruction_detector.py
    verify_dem_lookup.py
    verify_optical_flow_vo.py
    pipeline/
      quick_analysis.py
      run_rigid_experiment.py
      run_3d_experiment.py
      run_hud_video.py
      run_trajectory.py
      run_video_pipeline.py
      run_overture_dataset_pipeline.py
    manual/
      seg_bench/
  src/
  pyproject.toml
  README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Опциональные зависимости:

```bash
pip install -e ".[geo]"       # DEM/Overture/rasterio/geopandas
pip install -e ".[semantic]"  # mmseg semantic backend
pip install -e ".[dev]"       # notebooks
```

Системно нужен ffmpeg/ffprobe.

macOS:

```bash
brew install ffmpeg
```

## Docker

Минимальный Docker-образ фиксирует системное и Python-окружение для CLI и smoke-проверок без локальной `.venv`:

```bash
docker compose build
docker compose run --rm avia-geo-fusion python scripts/run_full_pipeline.py --help
```

Видео, model weights и результаты не вшиваются в образ, чтобы не раздувать его на гигабайты и не смешивать код с экспериментальными данными. В `docker-compose.yml` рабочая директория монтируется внутрь контейнера, поэтому локальные файлы вроде `data/videos/GP010269.MP4` будут доступны по тем же путям, если они есть на машине.

## Быстрый старт

Команды ниже предполагают активированное проектное окружение:

```bash
source .venv/bin/activate
```

1) Проверить локальные входные артефакты:

```text
data/videos/GP010269.MP4
configs/camera_gopro_hx.yaml
data/masks/anchors/index.json
data/retrieval/db_z14.npz
data/retrieval/db_z14.json
```

2) Быстрый component smoke для retriever + undistort + mask + BEV + matcher:

```bash
python scripts/end_to_end_smoke.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --output results/end_to_end
```

3) Основной integration pipeline:

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

Скрипт пишет:

- `results/full_pipeline/state.csv`
- `results/full_pipeline/trajectory.png`
- `results/full_pipeline/scale_history.png`

`state.csv` содержит состояние EKF, AGL/DEM-поля и obstruction diagnostics.

Для зимнего `GP010269.MP4` `--snow-mask` включён по умолчанию: он убирает из drone-BEV яркий снег, который плохо сопоставляется с летними/осенними спутниковыми тайлами. Дополнительно можно включить sat-side class gate:

```bash
python scripts/run_full_pipeline.py ... --semantic-mask
```

4) Диагностика и component checks:

```bash
python scripts/pipeline/run_diagnostic.py --video data/videos/GP010269.MP4 --run-name smoketest
python scripts/verify_obstruction_detector.py --video data/videos/GP010269.MP4 --output-dir results/stage1_5
python scripts/verify_dem_lookup.py --dem data/dem/test_synthetic.tif --output results/stage2_1
python scripts/verify_optical_flow_vo.py --camera-config configs/camera_gopro_hx.yaml
```

5) Legacy visual/HUD runners остаются полезными для ручной отладки:

```bash
python scripts/pipeline/quick_analysis.py
python scripts/pipeline/run_rigid_experiment.py
python scripts/pipeline/run_3d_experiment.py
python scripts/pipeline/run_hud_video.py
python scripts/pipeline/run_trajectory.py
python scripts/pipeline/run_video_pipeline.py
```

## Overture dataset pipeline

Сборка датасета по регионам РФ с фиксированным zoom=17:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage all
```

Этапно:

```bash
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage download
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage rasterize
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage qc
python scripts/pipeline/run_overture_dataset_pipeline.py --config configs/overture_dataset_ru.json --stage patches
```

Подробности: `docs/overture_dataset_pipeline.md`.

## Recovery intrinsics (фаза 0a)

```bash
python scripts/recover_intrinsics.py \
  --video data/videos/GP010269.MP4 \
  --profiles configs/camera_profile_candidates_gopro_hx.yaml \
  --output configs/camera_gopro_hx.yaml
```

Скрипт формирует:

- `configs/camera_gopro_hx.yaml`
- `results/intrinsics_recovery_report.json`

## Важно про config.json

`config.json` больше не является обязательным файлом проекта и не используется основным пайплайном.
Если файл присутствует локально, это исторический артефакт старой структуры.

## Git / artifacts

В обычный git не добавляются:

- raw video: `data/videos/GP010269.MP4`;
- model checkpoints: `*.pth`, `*.pt`, `*.ckpt`;
- `results/`;
- тяжёлые regenerated Overture folders: `images/`, `masks/`, `quality_preview/`.

Политика артефактов описана в `docs/artifacts.md`.
Логическая структура модулей описана в `docs/repository_structure.md`.
