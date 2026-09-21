# TaxaLens v1.0 на ScienceCluster: 200k → 500k → 1M

## Что мы делаем

Старый v0.9 остаётся контрольной точкой и не переписывается. Новый workflow
использует отдельный runner `scripts/run_multisource_scale.py` и три
последовательных профиля:

| Gate | Raw target | BIOSCAN | iNaturalist | GBIF | DiSSCo |
|---|---:|---:|---:|---:|---:|
| `raw200k` | 200 000 | 60 000 | 60 000 | 50 000 | 30 000 |
| `raw500k` | 500 000 | 150 000 | 150 000 | 125 000 | 75 000 |
| `raw1m` | 1 000 000 | 300 000 | 300 000 | 250 000 | 150 000 |

Это **raw targets**, а не обещание такого же количества финальных обучающих
изображений. После лицензий, дедупликации, недоступных URL и quality gate
чистый корпус будет меньше. Это нормальное и измеряемое поведение.

Мы начинаем только с `raw200k`. Переход к 500k и затем к 1M происходит после
проверки отчётов и качества предыдущего gate.

## Версионированный Swiss-28 scope

Все v1.0-профили используют отдельный файл
`configs/target_diptera_families_v10.json` с **28 целевыми семействами**.
Это исходное ядро из 20 семейств плюс:

- `Culicidae`, `Syrphidae`, `Simuliidae`, `Anthomyiidae`;
- `Dolichopodidae`, `Empididae`, `Hybotidae`, `Calliphoridae`.

Legacy v0.9 по-прежнему читает `configs/target_diptera_families.json` с 20
семействами, поэтому прежний результат остаётся воспроизводимым. В v1.0 уже
сама BIOSCAN-выборка ограничена Swiss-28: её 60k изображений не расходуются на
семейства, которые затем всё равно были бы отброшены стадией `plan`.

## Где что хранится

| Содержимое | Путь | Почему |
|---|---|---|
| Код | `~/projects/TaxaLens` | маленький, версионируемый |
| Общие исходные metadata | `/scratch/$USER/taxalens/source_cache` | iNaturalist и GBIF не скачиваются заново для каждого gate |
| Общий JPEG cache | `/scratch/$USER/taxalens/image_cache` | повторно использует уже скачанные изображения |
| Результаты 200k | `/scratch/$USER/taxalens/v10_raw200k_swiss28` | отдельная воспроизводимая попытка |
| Результаты 500k | `/scratch/$USER/taxalens/v10_raw500k_swiss28` | не смешивается с 200k |
| Результаты 1M | `/scratch/$USER/taxalens/v10_raw1m_swiss28` | не смешивается с предыдущими |
| Принятые модели и отчёты | `/shares/hawlitschek.ieu.uzh/TaxaLens/runs/v1.0/...` | сохраняются после успешного gate; изображения туда не копируем |

`/scratch` — рабочее, временное хранилище. Неактивные файлы могут удаляться
после 30 дней. `/shares` не является резервной копией, поэтому важные
результаты дополнительно нужно хранить вне кластера.

## Защита от смешивания запусков

При первом реальном этапе runner создаёт `run_identity.json`. В нём записаны
config hash, объём, source targets и пути общих cache. Если позже попытаться
запустить другой профиль в том же `DATA_ROOT`, workflow остановится, а не
смешает данные.

BIOSCAN также получает разные имена (`60k`, `150k`, `300k`), а DiSSCo —
отдельный export для каждого объёма. Legacy-пути
`diptera_30k_selection.csv`, `poc_v09` и `models_poc_v09` новым runner-ом не
используются.

## Один раз перед первым запуском

```bash
cd ~/projects/TaxaLens
mkdir -p logs

export CONFIG="configs/multisource_scale_v10_raw200k.json"
export DATA_ROOT="/scratch/$USER/taxalens/v10_raw200k_swiss28"
export SOURCE_ROOT="/scratch/$USER/taxalens/source_cache"
export IMAGE_ROOT="/scratch/$USER/taxalens/image_cache"
```

Проверить план без скачиваний и обучения:

```bash
module load apptainer

apptainer exec \
  "/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif" \
  "$HOME/.venvs/taxalens-v09/bin/python" \
  scripts/run_multisource_scale.py \
  --stage doctor \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --image-root "$IMAGE_ROOT"
```

В начале почти всё будет `missing`: doctor только показывает состояние и
ничего не скачивает.

## Порядок этапов raw200k

Каждый следующий job запускаем только после `COMPLETED 0:0` у предыдущего:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 80 logs/ИМЯ_ЛОГА.out
```

### 1. Предварительно скачать модели

Это предотвращает одновременную загрузку DINOv3/BioCLIP десятками array jobs.
DINOv3 может потребовать заранее принятые условия доступа Hugging Face.
Секретный токен нельзя вставлять в скрипт или отправлять в чат.

```bash
sbatch --export=ALL,STAGE=prefetch slurm/10_scale_stage.sbatch
```

### 2. BIOSCAN 60k

Сначала из metadata выбираются только представители Swiss-28, и лишь после
этого скачиваются соответствующие изображения. Это не 60k случайных Diptera.

```bash
sbatch --export=ALL,STAGE=bioscan slurm/10_scale_stage.sbatch
```

Если уже существует именно выборка `60k` с локальными файлами и нужно только
проиндексировать её без повторного скачивания:

```bash
sbatch --export=ALL,STAGE=bioscan,REUSE_EXISTING_BIOSCAN=1 slurm/10_scale_stage.sbatch
```

После успешного BIOSCAN job гарантируем квоту Muscidae/Tachinidae. Уже
скачанные изображения проверяются и переиспользуются:

```bash
sbatch --export=ALL,STAGE=bioscan-topup slurm/10_scale_stage.sbatch
```

### 3. iNaturalist metadata

Для масштаба больше 50k API fallback намеренно запрещён. Используется
официальный bulk metadata bundle. Это metadata, а не скачивание всех мировых
JPEG. После проверки свободного места:

```bash
sbatch \
  --export=ALL,STAGE=download,SOURCE=inat,ALLOW_INAT_BULK_DOWNLOAD=1 \
  slurm/10_scale_stage.sbatch
```

Если официальный archive уже лежит на кластере, можно передать
`INAT_ARCHIVE=/полный/путь/file.tar.gz` вместо повторного скачивания.

### 4. GBIF metadata

Для масштабируемой версии используется воспроизводимый GBIF download, а не
поисковый API. Сначала создаётся/отправляется request, затем после готовности
используется выданный download key. Учётные данные задаются только как
переменные окружения и не сохраняются в Git.

```bash
export GBIF_USER="ТВОЙ_GBIF_LOGIN"
export GBIF_EMAIL="ТВОЙ_EMAIL"
read -s -p "GBIF password: " GBIF_PASSWORD
export GBIF_PASSWORD
echo

module load apptainer

apptainer exec \
  "/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif" \
  "$HOME/.venvs/taxalens-v09/bin/python" \
  scripts/run_multisource_scale.py \
  --stage download \
  --source gbif \
  --submit-gbif-request \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --image-root "$IMAGE_ROOT"
```

Короткая команда отправит request и напечатает download key. Пароль не попадёт
в файл задания Slurm. Когда GBIF
покажет статус `SUCCEEDED`:

```bash
export GBIF_DOWNLOAD_KEY="КЛЮЧ_ОТ_GBIF"

sbatch \
  --export=ALL,STAGE=download,SOURCE=gbif,GBIF_DOWNLOAD_KEY="$GBIF_DOWNLOAD_KEY" \
  slurm/10_scale_stage.sbatch
```

После отправки задания лучше удалить секрет из текущей shell:

```bash
unset GBIF_PASSWORD
```

### 5. DiSSCo 30k

```bash
sbatch --export=ALL,STAGE=download,SOURCE=dissco slurm/10_scale_stage.sbatch
```

### 6. Нормализация и план корпуса

Запускаем **по одному** и проверяем завершение каждого job. Сначала:

```bash
sbatch --export=ALL,STAGE=ingest slurm/10_scale_stage.sbatch
```

Только после `COMPLETED 0:0`:

```bash
sbatch --export=ALL,STAGE=assemble slurm/10_scale_stage.sbatch
```

И только после успешного `assemble`:

```bash
sbatch --export=ALL,STAGE=plan slurm/10_scale_stage.sbatch
```

`assemble` дедуплицирует записи и назначает split по specimen/duplicate group,
поэтому одна и та же муха не должна оказаться одновременно в train и test.

### 7. Скачать только выбранные JPEG

```bash
sbatch --export=ALL,STAGE=cache slurm/10_scale_stage.sbatch
```

Это не скачивает все фотографии из iNaturalist/GBIF. Берутся только записи,
которые вошли в детерминированный training plan; существующий общий cache
переиспользуется.

### 8. Quality gate на L4

```bash
sbatch --export=ALL slurm/11_scale_quality.sbatch
```

Этап создаёт decisions/report, отдельные crops и group-safe manifest shards.
Оригинальные JPEG не изменяются.

### 9. Заморозить evaluation cohort 200k

После успешного quality gate:

```bash
sbatch --export=ALL,STAGE=freeze-eval slurm/10_scale_stage.sbatch
```

Создаются `evaluation_cohort.parquet` и отчёт с SHA-256. Файл содержит только
quality-approved `val`/`test` и проверяется на group leakage. Его сохраняем для
честного отдельного cross-scale benchmark следующих масштабов. Текущий training
report каждого gate по-прежнему использует его собственный детерминированный
`val`/`test`; frozen cohort не подмешивается в обучение.

### 10. Embeddings как Slurm array

Сначала узнать фактическое число shards после quality gate:

```bash
SHARD_COUNT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_count"])' \
  "$DATA_ROOT/pipeline/quality_manifest_shards/shards.json")

LAST_SHARD=$((SHARD_COUNT - 1))
echo "$SHARD_COUNT shards; last index = $LAST_SHARD"
```

Затем запустить по одному GPU job на shard. `%10` ограничивает одновременную
нагрузку десятью jobs; при политике кластера число можно уменьшить.

```bash
sbatch \
  --array=0-${LAST_SHARD}%10 \
  --export=ALL,ENCODER=all \
  slurm/12_scale_embed_array.sbatch
```

Каждый task делает один и тот же shard сначала DINOv3, затем BioCLIP. Готовые
checkpoint-и не пересчитываются, поэтому array можно безопасно отправить снова
после отдельных сбоев.

### 11. Merge, обучение и evaluation

Только когда все array tasks завершились успешно:

```bash
sbatch --export=ALL slurm/13_scale_train.sbatch
```

Скрипт блокирует обучение, если отсутствует хотя бы один актуальный DINO или
BioCLIP shard. Затем строятся DINO-only, BioCLIP-only и fused модели,
retrieval index и отчёты по источникам.

## Gate перед 500k

Не масштабируемся автоматически. Сначала проверяем:

1. `training_plan_report.json`: присутствуют все четыре источника и все
   обязательные семейства.
2. `image_quality_report.json`: `complete=true`; число принятых/отклонённых и
   review rows объяснимо.
3. Все embedding shards обоих encoders завершены; training job не сообщал
   missing/stale checkpoints.
4. `evaluation_by_source.md` и `encoder_comparison.md`: нет деградации на
   отдельных источниках, а fused модель сравнивается на одном cohort.
5. Frozen evaluation cohort и модели скопированы в `/shares`.
6. Ошибки, лицензии, дубликаты и дисковое потребление просмотрены вручную.

Только после этого меняем две переменные; source/image cache остаются общими:

```bash
export CONFIG="configs/multisource_scale_v10_raw500k.json"
export DATA_ROOT="/scratch/$USER/taxalens/v10_raw500k_swiss28"
```

После успешного 500k gate аналогично:

```bash
export CONFIG="configs/multisource_scale_v10_raw1m.json"
export DATA_ROOT="/scratch/$USER/taxalens/v10_raw1m_swiss28"
```

## Что сохранить в `/shares`

После успешного raw200k (не раньше):

```bash
DEST="/shares/hawlitschek.ieu.uzh/TaxaLens/runs/v1.0/raw200k"
mkdir -p "$DEST"/{models,manifests,reports}

rsync -a "$DATA_ROOT/models/" "$DEST/models/"
rsync -a "$DATA_ROOT/run_identity.json" "$DEST/reports/"
rsync -a "$DATA_ROOT/pipeline/evaluation_cohort.parquet" "$DEST/manifests/"
rsync -a "$DATA_ROOT/pipeline/evaluation_cohort_report.json" "$DEST/reports/"
rsync -a "$DATA_ROOT/pipeline/training_plan_report.json" "$DEST/reports/"
rsync -a "$DATA_ROOT/pipeline/corpus_report.json" "$DEST/reports/"
rsync -a "$DATA_ROOT/pipeline/corpus_report.md" "$DEST/reports/"
rsync -a "$DATA_ROOT/pipeline/image_quality_report.json" "$DEST/reports/"
```

JPEG cache и raw exports в `/shares` не копируем. Они воспроизводимы и занимают
основной объём. Но `/shares` тоже не backup: важные модели и frozen cohort
нужно дополнительно сохранить в другом надёжном месте.

## Быстрые проверки

```bash
python3 -m json.tool "$CONFIG" >/dev/null
bash -n slurm/10_scale_stage.sbatch
bash -n slurm/11_scale_quality.sbatch
bash -n slurm/12_scale_embed_array.sbatch
bash -n slurm/13_scale_train.sbatch
git status --short
```

Полный проектный тест на ScienceCluster:

```bash
sbatch slurm/02_project_tests.sbatch
```
