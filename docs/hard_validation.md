# Hard validation на GP010269.MP4

Цель — получить честную таблицу S0–S6 для защиты: ATE, `% TRACK`, false_fix_count,
recovery time и поведение dead reckoning на проблемных сегментах.

Текущий статус: harness, schema и docs готовы. Финальные числа не готовы, пока
example-точки не заменены вручную подтверждёнными landmarks и не прогнан
полный/canonical `state.csv`.

## 1. Подготовить входы

Нужны:

```text
data/videos/GP010269.MP4
results/full_pipeline/state.csv
data/ground_truth/gp010269_landmarks.json
```

Если `state.csv` ещё не создан:

```bash
python scripts/run_full_pipeline.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --output results/full_pipeline
```

По умолчанию runner использует XFeat, `window-radius=2`, retrieval top-1,
snow mask и `bootstrap-buffer=5`. Structural channel можно добавить явно через
`--structural-match`, если нужен VO+appearance+structural smoke.

## 2. Разметить landmarks

Скопировать схему:

```bash
cp data/ground_truth/gp010269_landmarks.example.json \
   data/ground_truth/gp010269_landmarks.json
```

В `landmarks` заменить example-точки на реальные ориентиры:

```json
{
  "id": "road_crossing_001",
  "t_sec": 123.4,
  "lat": 55.123456,
  "lon": 38.123456,
  "sigma_m": 75.0,
  "source": "manual_satellite_alignment",
  "note": "пересечение дороги и границы поля"
}
```

Рекомендуемый минимум для дипломной таблицы — 20–30 точек на весь ролик,
примерно по 2 точки на сегмент S0–S6. Точность таких GT-точек ограничена
примерно `sigma_m`, поэтому итоговый ATE нужно интерпретировать как порядок
ошибки, а не RTK-grade ground truth.

## 3. Запустить оценку

```bash
python scripts/evaluate_hard_validation.py \
  --state-csv results/full_pipeline/state.csv \
  --landmarks data/ground_truth/gp010269_landmarks.json \
  --output results/hard_validation
```

Для проверки самой схемы без реальной разметки:

```bash
python scripts/evaluate_hard_validation.py \
  --state-csv results/full_pipeline/state.csv \
  --landmarks data/ground_truth/gp010269_landmarks.example.json \
  --allow-example-landmarks \
  --output results/hard_validation_example
```

## 4. Выходы

- `summary.md` — компактный отчёт для защиты.
- `summary.json` — машинно-читаемая версия.
- `segment_metrics.csv` — таблица S0–S6.
- `timeline_metrics.csv` — per-row метрики по `state.csv`.
- `trajectory_vs_landmarks.png` — траектория EKF и landmarks в ENU.

## 5. Метрики

- `ATE` считается как расстояние между EKF state и piecewise-linear
  landmark-интерполяцией.
- `% TRACK` — доля строк сегмента, где `sigma_pos_m <= max_sigma_pos_m` и нет
  obstruction flag.
- `false_fix_count` — map-fix во время obstruction или любой map-fix внутри
  сегмента, помеченного как expected dead reckoning.
- `recovery_time_s` — время от конца expected-DR сегмента до первого принятого
  map-fix в следующем сегменте.
