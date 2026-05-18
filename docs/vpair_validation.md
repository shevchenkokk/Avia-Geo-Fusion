# VPAIR proxy validation

VPAIR полезен как внешний proxy-бенчмарк для диплома: это не то же самое, что
`GP010269.MP4`, но он ближе к самолётному сценарию, чем большинство UAV-наборов:
камера смотрит вниз, высота больше 300 м AGL, есть длинная траектория, reference
renders, depth и 6-DoF poses.

## Что добавлено

- `scripts/download_vpair_sample.py` — скачивает публичный `vpair_sample.zip` с
  Zenodo и выборочно распаковывает `queries`, `reference_views`, pose-файлы и
  опциональные `distractors`.
- `scripts/prepare_vpair.py` — строит CSV-манифесты:
  - `data/vpair/manifests/queries.csv`;
  - `data/vpair/manifests/references.csv`;
  - `data/vpair/manifests/distractors.csv`.
- `scripts/evaluate_vpair_vpr.py` — считает DINOv2 retrieval metrics через
  существующий `src.retriever.Retriever`.

## Быстрый запуск

```bash
python scripts/download_vpair_sample.py \
  --output-root data/external/vpair_sample

python scripts/prepare_vpair.py \
  --dataset-root data/external/vpair_sample \
  --output data/vpair/manifests

python scripts/evaluate_vpair_vpr.py \
  --dataset-root data/external/vpair_sample \
  --manifest-dir data/vpair/manifests \
  --output results/vpair_vpr \
  --device auto \
  --include-distractors
```

Для smoke-прогона без полного времени на все query:

```bash
python scripts/evaluate_vpair_vpr.py \
  --dataset-root data/external/vpair_sample \
  --manifest-dir data/vpair/manifests \
  --output results/vpair_vpr_smoke \
  --max-queries 20 \
  --device cpu
```

## Метрики

`evaluate_vpair_vpr.py` пишет:

- `summary.json`;
- `summary.md`;
- `predictions.csv`;
- `vpair_dinov2_db.npz/json`.

Основные поля:

- `exact_pair_top1_pct` — top-1 попал ровно в paired reference image;
- `top1_reference_pct` / `top1_distractor_pct` — ушёл ли top-1 в reference или
  hard negative distractor;
- `median_top1_reference_error_m`, `p95_top1_reference_error_m` — метрическая
  ошибка по ECEF-позам, если top-1 является reference;
- `R@K_Rm` — Recall@K внутри заданного радиуса по ECEF-дистанции.

## Ограничения интерпретации

- Это VPR/retrieval benchmark, а не полный `VO → EKF → map-measurement` flight
  pipeline.
- VPAIR надирный; `GP010269.MP4` — oblique/fisheye со значимой частью самолёта,
  облаками, снегом и unknown mounting. Поэтому VPAIR нельзя использовать как
  hard validation, но можно использовать как soft validation архитектурной
  поддержки aerial place recognition.
- Reference views — rendered geodata, а не реальные спутниковые тайлы. Это
  хорошо для pose ground truth, но domain gap отличается от нашего
  satellite-vs-aircraft matching.

## Как использовать в защите

Формулировка для диплома: VPAIR закрывает soft-вопрос «умеет ли retriever
работать на самолётной высоте и длинной aerial trajectory». Hard-качество
локализации всё равно подтверждается отдельно на `GP010269.MP4` и/или размеченных
landmarks.
