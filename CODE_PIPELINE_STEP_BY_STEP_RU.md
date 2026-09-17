# TaxaLens v0.9 — что делать с готовыми файлами

## Часть 1. Обновить GitHub безопасно

Не загружай `TaxaLens_v0.9_multisource_PoC_with_Genus_Key_Finder.zip` в репозиторий как один ZIP-файл. Его нужно
распаковать **в существующую локальную папку TaxaLens**, сохранив внутри неё папку `.git`.

1. Скачай `TaxaLens_v0.9_multisource_PoC_with_Genus_Key_Finder.zip` в `Downloads`.
2. Открой Terminal.
3. Напиши `cd ` с пробелом и перетащи существующую папку TaxaLens в окно Terminal. Нажми Enter.
4. Создай безопасную новую ветку:

```bash
git switch -c multisource-poc-v09
```

5. Распакуй содержимое ZIP в эту папку. На Mac команда обычно такая:

```bash
unzip -o ~/Downloads/TaxaLens_v0.9_multisource_PoC_with_Genus_Key_Finder.zip -d .
```

6. Проверь список изменений:

```bash
git status
```

7. Запусти тесты:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-corpus.txt
pytest -q
```

Ожидаемый результат: `48 passed`.

8. Отправь новую версию в GitHub:

```bash
git add .
git commit -m "Add TaxaLens v0.9 multi-source 100k PoC pipeline"
git push -u origin multisource-poc-v09
```

9. На GitHub открой **Pull request**, сравни `multisource-poc-v09` с `main` и только потом нажми
**Merge**. В GitHub не должны попасть картинки, dataset archives, пароли, `.venv` или model weights.

## Часть 2. Подготовить полное окружение

После GitHub-тестов установи ML-зависимости:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

DINOv3 имеет отдельные условия доступа. Открой страницу модели, войди в Hugging Face и прими
условия: <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>

После этого в Terminal:

```bash
hf auth login
```

## Часть 3. Проверить, что уже скачано

Твой готовый BIOSCAN остаётся по существующему пути:

```text
/content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
```

Проверка без скачивания:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage doctor \
  --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 \
  --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan \
  --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
```

`doctor` ничего не скачивает. Он только показывает `ready` или `missing` для BIOSCAN,
iNaturalist, GBIF, DiSSCo, training plan и модели.

Если BIOSCAN уже находится в другой папке, **не переноси и не скачивай его заново**.
Запомни путь к папке, внутри которой находятся `diptera_30k_images`,
`diptera_30k_selection.csv` и/или `bioscan5m`. Эту папку дальше передаём через
`--bioscan-root`.

## Часть 4. Докачать только Muscidae и Tachinidae из BIOSCAN

```bash
python scripts/run_multisource_poc_v09.py \
  --stage bioscan-topup \
  --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 \
  --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan \
  --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
```

Top-up считает уже имеющиеся изображения и обеспечивает минимум 1 500
Muscidae и 1 500 Tachinidae. Он скачивает только недостающие JPEG, не удаляет
готовые 30k и безопасно продолжается после остановки.

## Часть 4b. Подготовить остальные источники

Запусти:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage download \
  --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 \
  --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan \
  --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet \
  --reuse-existing-bioscan
```

Что произойдёт:

- BIOSCAN на этом шаге больше не скачивается: top-up уже выполнен отдельно;
- iNaturalist скачает official metadata bundle;
- DiSSCo продолжится до 15,000 records;
- для GBIF будет создан request, потому что GBIF export асинхронный.

Для отправки GBIF request:

```bash
export GBIF_USER="YOUR_GBIF_USERNAME"
export GBIF_PASSWORD="YOUR_GBIF_PASSWORD"
export GBIF_EMAIL="YOUR_EMAIL"

python scripts/run_multisource_poc_v09.py \
  --stage download \
  --data-root ~/TaxaLensData \
  --submit-gbif-request
```

Пароли вводятся только локально и не сохраняются в GitHub. GBIF выдаст download key. Когда
его статус станет `SUCCEEDED`, выполни:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage download \
  --data-root ~/TaxaLensData \
  --gbif-download-key YOUR_GBIF_DOWNLOAD_KEY
```

## Часть 5. Сделать единый корпус ровно на 100k

Конфигурация `configs/target_diptera_families.json` содержит 20 целевых семейств:
18 прежних MicroDiptera плюс **Muscidae** и **Tachinidae**. На стадии `plan`
данные всех четырёх источников фильтруются по этому списку; отсутствие Muscidae
или Tachinidae считается ошибкой валидации, а не замалчивается.

Команды выполняем по одной:

```bash
python scripts/run_multisource_poc_v09.py --stage ingest --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
python scripts/run_multisource_poc_v09.py --stage assemble --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
python scripts/run_multisource_poc_v09.py --stage plan --data-root /content/drive/MyDrive/EntoKey/Foundation_v06 --bioscan-root /content/drive/MyDrive/EntoKey/Foundation_v06/raw/bioscan --bioscan-manifest /content/drive/MyDrive/EntoKey/Foundation_v06/manifests/bioscan_raw.parquet
```

`plan` должен подтвердить:

| Источник | Количество |
| --- | ---: |
| BIOSCAN-5M | 30,000 |
| iNaturalist | 30,000 |
| GBIF | 25,000 |
| DiSSCo | 15,000 |
| **Всего** | **100,000** |

Посмотри два отчёта:

```text
~/TaxaLensData/poc_v09/manifest_validation.md
~/TaxaLensData/poc_v09/corpus_report.md
```

Если источник отсутствует, квота не набрана, есть duplicate IDs, плохие лицензии или split
leakage, строгий `plan` остановится до дорогого обучения.

## Часть 6. Докачать сами картинки

```bash
python scripts/run_multisource_poc_v09.py \
  --stage cache \
  --data-root ~/TaxaLensData
```

Команду можно перезапускать. Валидные картинки переиспользуются, битые и незавершённые
файлы скачиваются снова. Готовые dataset bytes остаются в `~/TaxaLensData`, а не в GitHub.

## Часть 7. Проверить биологическую пригодность изображений

Перед embeddings обязательно включи GPU и запусти:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage quality \
  --data-root ~/TaxaLensData
```

Для этапа нужны `torch` и `transformers>=4.56`. OWLv2 ищет мух на неоднородных
музейных кадрах GBIF/DiSSCo. Кадры без видимой мухи и слишком мелкие экземпляры
попадают в `training_plan_quality_review.parquet`; исходные JPEG не удаляются.
Пригодный небольшой экземпляр обрезается в отдельную папку `images_quality_crops`.
После прерывания повтори ту же команду: готовые checkpoints переиспользуются.

Проверь отчёт и контактный лист:

```text
~/TaxaLensData/poc_v09/image_quality_report.json
~/TaxaLensData/poc_v09/image_quality_review_contact_sheet.jpg
```

## Часть 8. Сделать маленький DINOv3 + BioCLIP smoke test дома

Сначала только один shard примерно на 1,000 записей:

```bash
python scripts/run_multisource_poc_v09.py \
  --stage embed \
  --data-root ~/TaxaLensData \
  --max-shards 1
```

Это проверит один и тот же shard двумя независимыми encoders. Кэш JPEG повторно не
скачивается. DINOv3 сохраняется в `quality_embedding_shards`, BioCLIP — в
`quality_embedding_shards_bioclip`; каждый можно безопасно продолжить после отключения.
Не запускай сразу все shards, пока первый не завершился у обоих encoders с `complete.json`.

## Часть 9. Полный PoC

Когда один shard прошёл нормально:

```bash
python scripts/run_multisource_poc_v09.py --stage embed --data-root ~/TaxaLensData
python scripts/run_multisource_poc_v09.py --stage train --data-root ~/TaxaLensData
python scripts/run_multisource_poc_v09.py --stage evaluate --data-root ~/TaxaLensData
```

Главный результат для Overleaf:

```text
~/TaxaLensData/models_poc_v09/evaluation_by_source.md
~/TaxaLensData/models_poc_v09/encoder_comparison.md
```

Второй отчёт честно сравнивает DINOv3, BioCLIP и fusion на test specimens. Метрики рода и
вида помечены как условные; они не выдаются за end-to-end точность всей системы.

## Часть 10. Проверить Genus Key Finder

Он работает отдельно ещё до обучения модели:

```bash
python scripts/find_genus_keys.py \
  --family Syrphidae \
  --genera Eristalis,Helophilus
```

В web-интерфейсе ничего отдельно нажимать не нужно: как только появляются
top-варианты рода, локальные ключи показываются сразу, а Crossref/OpenAlex
дополняют список автоматически. Для офлайн-режима:

```bash
python scripts/find_genus_keys.py \
  --family Syrphidae \
  --genera Eristalis,Helophilus \
  --offline
```

Важно: результат поиска — это список возможных определителей и ревизий, а не
доказательство правильности AI-предсказания. Проверяем регион, год, охват рода
и диагностические признаки.

## Часть 10. Что отправить Оливеру сейчас

Уже сейчас можно показать provisional Overleaf. Он честно фиксирует архитектуру, 100k design,
ресурсный запрос и отсутствие выдуманных результатов. После первого обучения обновим только
results block и, при необходимости, уточним Science IT storage/GPU estimate по реальной скорости.
