# Defense demo plan

Этот сценарий рассчитан на демонстрацию без нового видео от преподавателя.

## 1. 30-second opening

Фраза:

> Система оценивает положение БПЛА без GPS: между редкими абсолютными фиксациями
> по спутниковой карте работает optical-flow VO, а EKF сглаживает траекторию и
> отбрасывает выбросы. Так как новое полётное видео пока недоступно, я отдельно
> показываю воспроизводимые synthetic/proxy проверки и структурный канал,
> устойчивый к смене сезона.

## 2. Что открыть в репозитории

1. `README.md` — раздел “Валидация без реального видео”.
2. `scripts/run_full_pipeline.py` — canonical integration runner.
3. `scripts/verify_structural_matching.py` — no-video structural benchmark.
4. `docs/validation_readiness_report.md` — таблица готовности и ограничений.
5. `docs/implementation_status.md` — честная сверка плана и текущего кода.

## 3. Live commands

### Быстрая проверка импортов

```bash
python -m compileall -q src scripts
```

### Главная no-video демонстрация

One-command вариант:

```bash
python scripts/run_validation_pack.py --output results/demo_validation_pack
```

После запуска открыть `results/demo_validation_pack/summary.md`.
Эта команда не требует локального видео: real-cruise advisory stage будет
пропущен. Если старое `GP010269.MP4` есть локально и нужно включить его в отчёт,
добавьте `--video data/videos/GP010269.MP4`.

Если нужно показать только structural benchmark:

```bash
python scripts/verify_structural_matching.py \
  --dataset-root data/overture_ru_dataset_starter \
  --region-id moscow_city_small \
  --output results/demo_structural
```

Что показать после запуска:

- терминал: `CRITERION PASSED`;
- `results/demo_structural/summary.json`: `accepted`, `position_error_m`,
  `peak_score`, `sigma_xy_m`;
- `results/demo_structural/synthetic_bev.png`;
- `results/demo_structural/score_map.png`.

### EKF synthetic proof

```bash
python scripts/verify_ekf.py --output results/demo_ekf
```

Что сказать:

- stage 1 показывает рост uncertainty без фиксов;
- stage 2 показывает fusion с noisy VO + map fixes;
- stage 3 показывает bounded outage;
- stage 4 показывает Mahalanobis rejection выброса.

## 4. Если есть локальное старое видео

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
  --output results/demo_full_pipeline
```

Формулировка:

> Это smoke интеграционного runner’а на старом видео. Он доказывает, что каналы
> VO/DEM/appearance/structural/EKF связаны в одном процессе и пишут общий
> `state.csv`. Но без ground truth это не финальная accuracy-оценка.

## 5. Слайд “что готово / что осталось”

Готово:

- BEV/camera geometry;
- aircraft/snow masking;
- optical-flow VO;
- DEM-based AGL;
- EKF с Mahalanobis gating;
- scale-bias correction;
- appearance map measurement;
- Overture structural matching;
- full pipeline runner и CSV/plot artifacts.
- hard-validation harness для будущей таблицы S0–S6.

Осталось для финальных flight metrics:

- разметить 20–30 landmarks для `GP010269.MP4`;
- прогнать полный/canonical маршрут;
- посчитать ATE/%TRACK/false_fix_count/recovery_time_s по S0–S6;
- сделать ablation VO / VO+appearance / VO+appearance+structural;
- подобрать `pitch_deg`, `agl_m`/DEM и `structural_region_id` под новую зону;
- оформить финальные графики trajectory/error для диплома.

## 6. Short Q&A answers

**Почему synthetic validation допустима?**  
Потому что она не подменяет финальный flight benchmark, а закрывает проверку
модулей с известным ground truth: геометрия, EKF, DEM, VO и structural matching.

**Что является главным риском?**  
Не кодовая интеграция, а отсутствие нового видео/telemetry для финальной
полётной метрики.

**Почему structural matching важен?**  
Appearance matching ломается при смене сезона (снег/лето) и texture gap. Overture-векторы
сравнивают дороги, границы леса, воду, застройку и поля, поэтому менее зависимы
от сезона.

**Что будет сделано сразу после получения видео?**  
Запуск `run_full_pipeline.py` на коротких сегментах, проверка accepted fixes по
каналам, затем полный прогон и расчёт метрик по telemetry/ground truth.
