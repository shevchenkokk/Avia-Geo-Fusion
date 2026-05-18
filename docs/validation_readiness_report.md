# Validation readiness report

Цель документа — честно показать, что уже можно защищать без нового полётного
видео, что проверяется синтетически, а что остаётся ограничением до получения
реальной записи с ground truth или хотя бы телеметрией.

## 1. Краткий статус

| Блок | Статус | Чем подтверждается | Ограничение |
|---|---:|---|---|
| Camera/BEV geometry | Готово для smoke/интеграции | `configs/camera_gopro_hx.yaml`, `BevRectifier`, `verify_optical_flow_vo.py` stage 1/2 | pitch/AGL всё ещё задаются параметрами или DEM, без IMU-attitude |
| Optical-flow VO | Готово синтетически | `scripts/verify_optical_flow_vo.py` восстанавливает известную скорость на synthetic LK pair | без телеметрии реального видео drift оценивается только sanity-check |
| DEM/AGL lookup | Готово | `scripts/verify_dem_lookup.py` на аналитическом synthetic DEM | качество зависит от внешнего DEM для новой локации |
| EKF fusion + Mahalanobis gating | Готово синтетически | `scripts/verify_ekf.py` проверяет propagation, fusion, outage, outlier rejection | нет factor graph / IMU preintegration |
| Scale-bias correction | Готово синтетически | `scripts/verify_scale_correction.py` проверяет convergence и drift reduction | требует регулярных map fixes для устойчивой калибровки |
| Appearance map measurement | Готово синтетически, fragile на cross-season | `scripts/verify_map_measurement.py`, `run_full_pipeline.py --backend xfeat/orb` | зимний drone frame vs летние/осенние тайлы даёт мало accepted fixes |
| Visual retriever | Интегрирован как DINOv2-B CLS + cosine NN | `src/retriever.py`, `data/retrieval/db_z14.*` | это не AnyLoc/VLAD; AnyLoc остаётся future-work/research reference |
| StructuralMatcher / Overture channel | Интегрирован opt-in + проверен synthetic/proxy | `scripts/verify_structural_matching.py`; `run_full_pipeline.py --structural-match` обновляет EKF через `update_map_position_wgs84` | выключен по умолчанию; full-flight ablation ещё не готов |
| Full pipeline runner | Интегрирован | `scripts/run_full_pipeline.py` объединяет VO, DEM, appearance, optional structural, EKF; defaults: XFeat, `window-radius=2`, retrieval top-1, `bootstrap-buffer=5` | end-to-end accuracy нельзя доказать без landmarks/GT |
| Hard validation S0–S6 | Harness готов, финальные числа pending | `scripts/evaluate_hard_validation.py`, `data/ground_truth/gp010269_landmarks.example.json`, `docs/hard_validation.md` | нужны 20–30 manually confirmed landmarks и полный/canonical `state.csv` |

Итоговая формулировка для защиты: проект готов как воспроизводимый прототип
геолокализации без GPS с проверенными компонентами и интегрированным full
pipeline; абсолютная end-to-end точность на новом полёте остаётся неподтверждённой
до получения видео/телеметрии, поэтому сейчас она заменена synthetic/proxy
валидацией.

Подробный статус относительно исходного исследовательского плана вынесен в
`docs/implementation_status.md`. Ключевые расхождения: retriever — DINOv2-B CLS,
не AnyLoc/VLAD; MASt3R focal cross-check и DPVO не реализованы; aircraft mask
сейчас nearest-anchor Step A; structural channel уже интегрирован, но включается
явно.

## 2. Minimal validation pack без flight video

Команды ниже не требуют нового полётного видео. Они проверяют компоненты, где
ground truth задаётся аналитически или через известный synthetic target.

One-command вариант для защиты:

```bash
python scripts/run_validation_pack.py --output results/validation_pack
```

Он запускает проверки ниже, сохраняет stdout/stderr каждого шага и пишет:

- `results/validation_pack/summary.md` — короткий Markdown-отчёт;
- `results/validation_pack/summary.json` — машинно-читаемый статус;
- `results/validation_pack/structural/synthetic_bev.png`;
- `results/validation_pack/structural/score_map.png`.

Для hard validation на реальном `GP010269.MP4` используется отдельный harness:

```bash
python scripts/evaluate_hard_validation.py \
  --state-csv results/full_pipeline/state.csv \
  --landmarks data/ground_truth/gp010269_landmarks.json \
  --output results/hard_validation
```

Схема разметки лежит в `data/ground_truth/gp010269_landmarks.example.json`,
подробный порядок — в `docs/hard_validation.md`. Example-файл нужен только как
schema; его числа нельзя использовать как финальный ATE.

По умолчанию runner не требует локального видео: VO-скрипт выполняет synthetic
stage 1/2, а real-cruise advisory stage пропускается. Если старое видео есть и
его нужно включить в отчёт, добавьте:

```bash
python scripts/run_validation_pack.py \
  --output results/validation_pack \
  --video data/videos/GP010269.MP4
```

```bash
python -m compileall -q src scripts

python scripts/verify_dem_lookup.py \
  --dem data/dem/test_synthetic.tif \
  --output results/validation/dem

python scripts/verify_ekf.py \
  --output results/validation/ekf

python scripts/verify_scale_correction.py \
  --output results/validation/scale

python scripts/verify_map_measurement.py \
  --output results/validation/map_measurement

python scripts/verify_optical_flow_vo.py \
  --camera-config configs/camera_gopro_hx.yaml \
  --video data/videos/GP010269.MP4

python scripts/verify_structural_matching.py \
  --dataset-root data/overture_ru_dataset_starter \
  --region-id moscow_city_small \
  --output results/validation/structural
```

Ожидаемые критерии:

- все команды завершаются с кодом `0`;
- `verify_structural_matching.py` печатает `CRITERION PASSED`;
- в `results/validation/structural/summary.json` поле `accepted=true`, а
  `position_error_m <= max_position_error_m` из CLI;
- для EKF/scale/map-measurement скрипты печатают `CRITERION PASSED`;
- для optical-flow VO stage 1/2 проходят; если локального `GP010269.MP4` нет,
  stage 3 пропускается автоматически.

## 3. Optional validation с локальным `GP010269.MP4`

Если старое видео есть локально, можно показать, что runner запускается и логирует
каналы отдельно. Это smoke, а не доказательство абсолютной точности, потому что
в видео нет ground truth telemetry.

```bash
python scripts/run_full_pipeline.py \
  --video data/videos/GP010269.MP4 \
  --camera-config configs/camera_gopro_hx.yaml \
  --anchors-dir data/masks/anchors \
  --retriever-db data/retrieval/db_z14 \
  --dem data/dem/srtm_n55_e038.tif \
  --start-s 45 \
  --end-s 55 \
  --backend orb \
  --structural-match \
  --structural-vectors-root data/overture_ru_dataset_starter/vectors \
  --structural-region-id kolomna_cruise \
  --output results/validation/full_pipeline_smoke
```

Проверять в выводе:

- `structural match : on`;
- `map attempts`, `map accepted`;
- `structural : attempts`, `structural accepted`;
- наличие `state.csv`, `trajectory.png`, `scale_history.png`.

## 4. Что говорить преподавателю про отсутствие нового видео

1. Реальное видео не замалчивается: это внешний блокер для финальной полётной
   метрики ATE/RPE.
2. Вместо него сделан воспроизводимый validation pack:
   - аналитическая синтетика для VO, EKF, DEM, scale correction;
   - synthetic cross-season BEV для structural matching;
   - optional smoke на старом `GP010269.MP4`, если файл доступен локально.
3. Архитектурный риск cross-season appearance matching явно закрывается opt-in
   Overture-based structural channel, который не зависит от цвета/сезона тайла.
4. Для `GP010269.MP4` следующий эксперимент уже определён: вручную разметить
   landmarks, прогнать `run_full_pipeline.py`, затем получить S0–S6 metrics через
   `scripts/evaluate_hard_validation.py`.

## 5. Readiness decision

Проект можно показывать как инженерный прототип с reproducible validation.
Нельзя честно утверждать, что end-to-end точность на новом полётном видео уже
доказана. Правильная формулировка: “готовность компонентов подтверждена, full
pipeline интегрирован, полётная метрика ожидает внешний видео/telemetry artifact”.
