# Semantic segmentation in the localization pipeline

Этот документ фиксирует фактическое состояние семантической сегментации и то,
как её нужно расширить до полноценного mask-to-mask matching, если следовать
идее «маска спутника ↔ маска изображения с самолёта ↔ сопоставление геометрии».

## Что реализовано сейчас

В текущем `scripts/run_full_pipeline.py` сегментация включается флагом
`--semantic-mask` и работает как gate для appearance matching:

1. `GeoSegmentor` инициализируется через mmseg/SegFormer.
2. Appearance matcher (`XFeat`, `LoFTR`, `LightGlue`, `ORB`) сначала находит
   keypoint matches между drone-BEV и спутниковым окном.
3. Для спутникового окна строится `sat_cmap = seg.stable_class_map(map_cv2)`.
4. Из matches оставляются только те, у которых map-side keypoint лежит на
   стабильном классе (`sat_cmap[y, x] > 0`).
5. `compute_map_measurement(...)` строит гомографию уже по отфильтрованным
   matches и отдаёт абсолютное измерение в EKF.

То есть `--semantic-mask` остаётся фильтром ложных keypoints на спутниковой
стороне, а не самостоятельным matcher.

Отдельно есть `StructuralMatcher`: он растеризует Overture-векторы в классовую
карту спутниковой стороны и сопоставляет её с Canny edges на drone-BEV через
multi-channel NCC. Теперь у него есть второй режим: при запуске
`--semantic-structural-match` на drone-BEV строится `GeoSegmentor.stable_class_map(...)`,
а затем выполняется class-wise mask-to-mask NCC против Overture-v7 каналов.
Это и есть первый реализованный вариант семантического геометрического matcher.

## Что это значит для дипломной формулировки

Корректная текущая формулировка:

- «Семантика используется как фильтр/взвешивание для appearance measurements и
  как источник спутниковых структурных каналов через Overture».

Новая корректная формулировка для режима `--semantic-structural-match`:

- «Система поддерживает отдельный semantic geometry channel: SegFormer class map
  на drone-BEV сопоставляется с Overture-derived class map спутниковой стороны
  через class-wise NCC; результат проходит consistency gate и EKF Mahalanobis
  gate как независимое абсолютное измерение».

## Реализованная схема mask-to-mask matcher

Сейчас semantic mask-to-mask режим встроен как вариант `StructuralMatcher`, а не
как отдельный класс `SemanticMaskMatcher`:

1. **Drone-side mask**
   - взять исходный кадр;
   - убрать самолёт/фюзеляж через `AircraftMaskTracker`;
   - построить BEV через `BevRectifier`;
   - применить SegFormer к drone-BEV или применить SegFormer до BEV и
     спроецировать class map в BEV;
   - получить `drone_class_map_bev`.

2. **Satellite-side mask**
   - предпочтительно строить class map из Overture/OSM/Sentinel layers, потому
     что это геометрически стабильнее, чем сегментировать RGB-спутник;
   - альтернативно применить SegFormer к спутниковому окну;
   - получить `sat_class_map_window` в той же метрической сетке, что drone-BEV.

3. **Geometric matching**
   - вокруг pose seed от EKF/retriever строится search window;
   - считается class-wise NCC по каналам water, forest_edge, roads,
     field_boundary, built_up;
   - для текущего 5-классного SegFormer используется маппинг:
     water→water, vegetation→forest_edge, buildings→built_up, roads→roads;
   - выдаётся `(lat, lon, sigma_xy, score)`. Yaw пока не оценивается этим
     каналом, но может быть добавлен как следующий шаг через перебор yaw.

4. **EKF update**
   - measurement проходит Mahalanobis gate;
   - covariance зависит от score peak sharpness, class support, площади масок и
     согласия с appearance/structural каналом.

## Минимальный next step

Самый короткий путь к более чистой архитектуре:

1. Вынести appearance, structural edges и semantic geometry в общий интерфейс
   `MapMeasurementChannel`.
2. Добавить отдельный `SemanticMaskMatcher` как тонкую обёртку над текущим
   semantic-режимом `StructuralMatcher`.
3. Добавить yaw-search для semantic geometry channel.
4. В отчёте показать ablation:
   - appearance only;
   - appearance + sat-side semantic gate;
   - Overture structural edges;
   - full semantic mask-to-mask.
