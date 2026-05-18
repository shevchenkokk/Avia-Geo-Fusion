# Implementation status vs project plan

Этот файл фиксирует, что в репозитории уже реализовано, а что остаётся
ограничением плана. Он нужен, чтобы не смешивать исследовательские цели
`PROJECT_PLAN.md` с фактическим состоянием кода.

## Закрыто в текущей реализации

- **Camera / fisheye calibration.** `configs/camera_gopro_hx.yaml` содержит
  восстановленные `K`, `D`, `K_rectified`, confidence/margin и score table по
  кандидатам GoPro. Выбранный профиль — data-driven
  `gopro_hero4_black_medium_1080p_seed`, не плановый Wide-профиль.
- **Seed aircraft masks.** `data/masks/anchors/` содержит 30 anchor-масок для
  `GP010269.MP4`.
- **Dynamic aircraft mask, Steps A+B.** `src/aircraft_mask.py` делает
  nearest-anchor lookup плюс per-frame Lucas-Kanade propagation от ближайшего
  anchor (forward+backward tracking, fb-error gate, coverage gate, авто-fallback
  при низкой confidence). Periodic SAM refresh между anchors остаётся future
  work.
- **BEV view-centre correction.** `BevRectifier.view_centre_body_m()` отдаёт
  forward/right смещение центра BEV-кадра в body NED. Главный пайплайн
  (`scripts/run_full_pipeline.py`) применяет коррекцию перед тем как класть
  appearance/structural fix в EKF: lat/lon, полученные из гомографии, — это
  «точка под центром BEV» (≈1073 м впереди при pitch=-30°, AGL=620), и без
  компенсации yaw_rate * forward_m давал бы боковую невязку в разворотах.
- **Obstruction detector.** `src/obstruction_detector.py` считает low-texture /
  blur / cloud-like metrics и применяет hysteresis.
- **VO + DEM + EKF.** `src/optical_flow_vo.py`, `src/dem_lookup.py` и
  `src/ekf.py` покрывают OF VO, DEM/AGL lookup, 9-state ENU EKF, Joseph-form
  covariance update, Mahalanobis gates, heading prior и scale-bias update.
- **Appearance matching.** `src/neural_matching.py` поддерживает XFeat, LoFTR,
  LightGlue/SuperPoint и ORB fallback.
- **Structural / semantic geometry matching.** `src/structural_matcher.py` есть
  и подключён в `scripts/run_full_pipeline.py` как opt-in EKF measurement
  channel через `--structural-match`. Режим `--semantic-structural-match`
  добавляет drone-side SegFormer class map и class-wise mask-to-mask NCC против
  Overture-v7 каналов. Канал выключен по умолчанию и пока подтверждён
  synthetic/proxy тестом, а не полной flight ablation.
- **Hard-validation harness.** `scripts/evaluate_hard_validation.py`,
  `data/ground_truth/gp010269_landmarks.example.json` и
  `docs/hard_validation.md` задают воспроизводимый путь к таблице S0–S6:
  ATE, `% TRACK`, `false_fix_count`, `recovery_time_s`.

## Важные расхождения с исследовательским планом

- **Retriever is not AnyLoc/VLAD.** Фактический retriever — DINOv2-B CLS
  descriptor + cosine nearest-neighbour (`src/retriever.py`). AnyLoc/VLAD
  остаётся research/future-work формулировкой, а не реализованным компонентом.
- **MASt3R focal cross-check не реализован.** Intrinsics выбираются по
  reproducible grid-search/quality-gate, без MASt3R cross-check.
- **DPVO не реализован.** Текущий VO канал — optical-flow ray-cast на AGL plane.
- **Dynamic roll from VO не подаётся в BEV.** BEV runner использует CLI/config
  pitch/roll parameters; IMU/VO attitude feedback loop не закрыт.
- **Hard metrics ещё не финальные.** Harness готов, но example landmarks нельзя
  использовать в дипломных числах. Нужны вручную подтверждённые
  `data/ground_truth/gp010269_landmarks.json` и полный `state.csv`.
- **Full-flight empirical ablation не закрыт.** Нет финальной таблицы
  VO / VO+appearance / VO+appearance+structural на всём `GP010269.MP4`.

## Текущие runner defaults

- `scripts/run_full_pipeline.py` по умолчанию использует `--backend xfeat`,
  `--window-radius 2`, `--bootstrap-buffer 5`, retrieval top-1 и snow mask.
- Structural channel включается явно: `--structural-match`.
- Semantic geometry channel включается явно: `--semantic-structural-match`; он
  автоматически включает structural channel и использует SegFormer class map на
  drone-BEV как template для class-wise NCC.
- `--semantic-mask` выключен по умолчанию, потому что cross-domain effect
  проверяется отдельно. В текущем коде это sat-side class gate для keypoint
  matches; полноценный mask-to-mask режим включается отдельным
  `--semantic-structural-match`. Детали см. `docs/semantic_segmentation_pipeline.md`.

## Следующий эмпирический шаг

1. Скопировать `data/ground_truth/gp010269_landmarks.example.json` в
   `data/ground_truth/gp010269_landmarks.json`.
2. Вручную разметить 20–30 visually confirmed landmarks на `GP010269.MP4`.
3. Прогнать `scripts/run_full_pipeline.py` на нужном окне/полном видео.
4. Запустить `scripts/evaluate_hard_validation.py` и использовать
   `results/hard_validation/segment_metrics.csv` как основу слайда S0–S6.
