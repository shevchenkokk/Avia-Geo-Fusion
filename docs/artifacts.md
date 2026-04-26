# Политика артефактов

В git хранятся исходный код, конфиги, документация и небольшие артефакты воспроизводимости.

Крупные локальные артефакты не должны попадать в обычную историю git:

- `data/videos/GP010269.MP4` — исходное видео GoPro (~2 GB).
- `configs/*.pth` — сторонние чекпоинты сегментации (сотни MB).
- `results/**/*.pth` — чекпоинты обучения и крупные выходы экспериментов.
- `data/overture_ru_dataset_starter/images/` — сгенерированные спутниковые патчи (~3.3 GB).
- `data/overture_ru_dataset_starter/masks/` — сгенерированные растровые маски.
- `results/` — все прогоны, графики, CSV, логи и каталоги обучения.

Небольшие артефакты, которые допустимы в git:

- `configs/*.yaml`, `configs/*.json`, `configs/*.py` — конфиги камеры, миссии и пайплайнов.
- `data/dem/*.tif` — небольшие DEM-файлы для smoke-проверок и локальных прогонов.
- `data/masks/anchors/**` — набор anchor-масок самолета для `GP010269.MP4`.
- `data/retrieval/db_z14.json` и `data/retrieval/db_z14.npz` — компактный индекс retriever для smoke-тестов.
- `data/overture_ru_dataset_starter/*.json` и `*.csv` — метаданные датасета.
- `data/overture_ru_dataset_starter/vectors/**/*.parquet` — компактные Overture-векторы для структурного матчинга.

Для воспроизведения полных экспериментов тяжелые артефакты публикуются отдельно: private object storage, GitHub Releases или Git LFS (если владелец репозитория его включил).

Ожидаемая локальная раскладка после клонирования:

```text
data/videos/GP010269.MP4
configs/<external-model>.pth
results/segformer_overture_b0_ade1400_nocw_best/best_mIoU_iter_932.pth
```

Базовый (не семантический) пайплайн должен импортироваться и показывать `--help` даже без этих крупных артефактов.
