# TaxaLens v0.9 — Multi-Source PoC + Genus Key Finder 🪰🔑

Это восстановленная последняя версия проекта. Она соединяет 100k multi-source
pipeline и автоматический поиск определительных ключей после предсказания рода.

## Бесплатный публичный сайт

Готовую fusion-модель DINOv3 + BioCLIP можно опубликовать как бесплатный
Hugging Face ZeroGPU Space. Посетитель загружает фотографию прямо на сайте,
получает иерархические кандидаты, похожие образцы и может выбрать предложенный
род, чтобы открыть morphology navigator и опубликованные ключи. Однократная
инструкция и автоматический скрипт находятся в `ZEROGPU_DEPLOY_RU.md`.

Биологический scope теперь содержит **20 целевых семейств Diptera**: исходные
18 MicroDiptera плюс **Muscidae** и **Tachinidae**. Единый список находится в
`configs/target_diptera_families.json`; `plan` применяет его ко всем четырём
источникам и отдельно проверяет присутствие Muscidae и Tachinidae.

```text
FULL IMAGE -> DINOv3 ViT-B/16 -> family -> top genera -> selective species
                                   |             |
                                   |             +-> Genus Key Finder
                                   +-> open-set rejection + similar specimens
```

Genus Key Finder берёт до четырёх предложенных родов, добавляет наиболее
вероятное семейство и ищет кандидаты-определители в локальном проверяемом
каталоге, Crossref и OpenAlex. Найденная публикация — это **кандидат на ключ**,
а не подтверждение определения: интерфейс явно просит проверить регион,
таксономический охват и диагностические признаки.

## Одна команда-оркестратор

```bash
python scripts/run_multisource_poc_v09.py --stage doctor --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage download --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN --reuse-existing-bioscan
python scripts/run_multisource_poc_v09.py --stage ingest --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage assemble --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage plan --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage cache --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage embed --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN --max-shards 1
python scripts/run_multisource_poc_v09.py --stage train --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
python scripts/run_multisource_poc_v09.py --stage evaluate --data-root ~/TaxaLensData --bioscan-root /PATH/TO/EXISTING/BIOSCAN
```

Для уже завершённого `Foundation_v06` сначала запускается отдельный selective
top-up. Он сохраняет готовые 30k и докачивает только недостающее до 1 500
Muscidae и 1 500 Tachinidae:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage bioscan-topup \
  --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 \
  --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan \
  --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
```

`--reuse-existing-bioscan` остаётся отдельным строгим локальным режимом для
обычного `download`: он не делает BIOSCAN-запросов, пока загружаются или
подготавливаются другие источники. Подробности находятся в
`BIOSCAN_MUSCIDAE_TACHINIDAE_TOPUP_RU.md`.

Проверить Key Finder отдельно:

```bash
python scripts/find_genus_keys.py --family Syrphidae --genera Eristalis,Helophilus
```

Для полностью офлайн-проверки добавь `--offline`. Web-интерфейс показывает
локальные рекомендации сразу, а затем автоматически дополняет их live-поиском.

## 100k PoC

| Источник | Цель |
| --- | ---: |
| BIOSCAN-5M | 30,000 |
| iNaturalist | 30,000 |
| GBIF | 25,000 |
| DiSSCo | 15,000 |
| **Всего** | **100,000** |

---

## Сохранённая логика baseline A из v0.8

## Сейчас делаем только baseline A

```text
FULL IMAGE -> DINOv3 -> family -> genus -> species -> uncertainty -> similar specimens
```

**Никаких anatomy masks. Никаких 2x2 tiles. Никакого обязательного crop.**

Цель v0.9 — сначала честно измерить, насколько далеко мы можем уехать на одной полной картинке, а затем направить человека к подходящему морфологическому ключу.

## Данные

Первый target corpus:

| Источник | Цель | Роль |
| --- | ---: | --- |
| BIOSCAN-5M | 30,000 | standardized preserved/DNA-linked specimens |
| iNaturalist | 30,000 | field photos and natural backgrounds |
| GBIF | 25,000 | museum/preserved specimen media |
| DiSSCo | 15,000 | European digital specimens/media |

Cross-source deduplication остаётся обязательной.

## Default backbone

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

Default representation:

```text
one complete image -> one normalized DINOv3 embedding
```

Перед backbone изображение вписывается в 512x512 с сохранением aspect ratio и padding. Мы не center-crop'аем муху и не режем её на части.

## Основные entry points

```text
configs/wholefly_foundation_v08.json
configs/sources_wholefly_v08.example.json
scripts/source_status_v08.py
scripts/run_wholefly_v08.py
notebooks/WholeFly_Foundation_v08_Colab.ipynb
```

## Порядок работы

### 1. Подготовить/скачать источники

BIOSCAN можно докачивать текущим selective workflow. iNaturalist, GBIF и DiSSCo подключаются как отдельные source manifests.

### 2. Проверить, что уже есть

```bash
python scripts/source_status_v08.py --config configs/sources_wholefly_v08.example.json
```

### 3. Собрать master manifest

```bash
python scripts/prepare_corpus.py \
  --config configs/sources_wholefly_v08.example.json \
  --stage assemble
```

Ожидаемый путь:

```text
data/corpus_v08/master_manifest.parquet
```

### 4. Сделать plan

```bash
python scripts/run_wholefly_v08.py \
  --master-manifest data/corpus_v08/master_manifest.parquet \
  --stage plan
```

### 5. Построить whole-image embeddings

```bash
python scripts/run_wholefly_v08.py \
  --master-manifest data/corpus_v08/master_manifest.parquet \
  --stage embed \
  --max-shards 2
```

Завершённые shards пропускаются при следующем запуске.

### 6. Train

```bash
python scripts/run_wholefly_v08.py \
  --master-manifest data/corpus_v08/master_manifest.parquet \
  --stage train
```

## Важно: tiles НЕ удалены навсегда

Код multi-crop сохранён только как **эксперимент C**, но он не включается default config'ом. После baseline A мы можем создать отдельный config с `tile_grid: 2` и сравнить его на том же validation split.

Если C не выигрывает — выкидываем его. Никакой дополнительной ручной работы не требуется.

## Anatomy pilot

20 хороших экспертных anatomy annotations можно сохранить как research seed. Они не нужны для baseline A и не блокируют обучение.
