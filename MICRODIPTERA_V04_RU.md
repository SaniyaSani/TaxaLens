> **LEGACY / reference:** этот документ сохранён для воспроизводимости старого pipeline. Для текущей архитектуры используй `README_RU.md` и `WHOLE_FLY_PIPELINE_RU.md`.

# MicroDiptera v0.4

v0.4 — это family-first модель для маленьких Diptera. Она не обещает species-ID
там, где на общей фотографии не видны диагностические признаки.

## Что изменилось относительно v0.3

Одна photograph больше не превращается только в один уменьшенный habitus. Для
каждого изображения v0.4 получает embedding целой мухи и четырёх detail tiles при
разрешении 518 px, нормализует и объединяет их. Если для specimen есть несколько
views, их embeddings дополнительно объединяются на specimen level.

```text
whole image + 4 tiles
          ↓
    view embedding
          ↓
dorsal + lateral + wing + head + terminalia
          ↓
  specimen embedding
          ↓
family → conditional genus → conditional species
```

## Самый простой запуск в Colab

1. Открой `notebooks/MicroDiptera_v04_Colab.ipynb`.
2. Выбери T4 GPU.
3. Проверь `REPO_URL`.
4. Нажми Run all.

По умолчанию собирается до 40 лицензированных iNaturalist observations на каждое
из 18 семейств. Это bounded API pilot, а не массовый scraping. После успешной
проверки увеличивай `PER_FAMILY` постепенно; для большого корпуса используй
официальные Open Data tables и v0.3 corpus adapters.

Legacy v0.9 target families находятся в `configs/target_diptera_families.json`:
исходные 18 MicroDiptera плюс Muscidae и Tachinidae. Старый файл
`configs/microdiptera_families.json` сохранён только для совместимости и содержит
тот же расширенный список. Для v1.0 scale-up используется отдельный Swiss-28
scope `configs/target_diptera_families_v10.json`.

## Локальный запуск

```bash
python scripts/build_microdiptera_pilot.py \
  --per-family 40 \
  --download
```

```bash
python scripts/embed_multiview.py \
  --manifest data/microdiptera/microdiptera_manifest.csv \
  --out-dir models_microdiptera \
  --image-size 518 \
  --tile-grid 2 \
  --batch-size 4
```

```bash
python scripts/train_hierarchical.py \
  --model-dir models_microdiptera \
  --min-family 12 \
  --min-genus 6 \
  --min-species 4
```

```bash
python scripts/build_retrieval_index.py --model-dir models_microdiptera
```

## Multi-view verified specimens

Если одна папка содержит views одного specimen, используй общий specimen ID, а
view обозначай в имени файла:

```text
PHOR-001_dorsal.jpg
PHOR-001_lateral.jpg
PHOR-001_head.jpg
PHOR-001_wing.jpg
PHOR-001_terminalia.jpg
```

```bash
python scripts/add_local_specimens.py \
  --image-dir /path/to/PHOR-001 \
  --family Phoridae \
  --genus Megaselia \
  --specimen-id PHOR-001 \
  --collector "Sanny"
```

`view_type=auto` распознаёт dorsal, lateral, head, wing, antenna, thorax,
legs и terminalia. Все views получают один `specimen_group_id` и никогда не
разделяются между train/test.

## Иерархия и open set

Сначала обучается общий family head. Затем для каждой достаточно представленной
family обучается собственный genus head; species head существует только внутри
конкретного genus. Это не даёт, например, species из Sciaridae появиться под
предсказанной Phoridae.

Для каждого класса сохраняется centroid embedding и нижний порог сходства. Если
query находится вне известного распределения, Workbench возвращает open-set
rejection вместо уверенного выдуманного taxon.

Это baseline rejection, а не математическая гарантия неизвестности. Порог нужно
позже калибровать на отдельном наборе неизвестных taxa и новых imaging domains.

## Species policy

По умолчанию species heads используют только:

- `A`: DNA-backed или project-verified specimen;
- `B`: curated museum determination.

iNaturalist Research Grade (`C`) полезен для family/genus representation, но не
становится автоматически species truth для cryptic MicroDiptera. После ручной
проверки policy можно расширить:

```bash
python scripts/train_hierarchical.py \
  --model-dir models_microdiptera \
  --species-label-quality A,B,C
```

## Как оценивать

Challenge set должен быть отделён от training corpus. В него нужны:

- новые specimens, которых нет среди источников обучения;
- field photos и pinned/microscope views отдельно;
- easy и cryptic taxa;
- правильный максимально подтверждённый rank;
- эксперт/ключ/DNA как provenance определения.

Одна правильная `Eristalis tenax` показывает, что pipeline работает. Научную
пригодность покажут macro-averaged family accuracy, source/domain metrics,
open-set false-accept rate и доля честных abstentions на challenge set.
