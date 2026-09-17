// Generate the regular fail-fast notebook and, optionally, a portable recovery launcher.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import crypto from 'node:crypto';

const root = path.resolve(import.meta.dirname, '..');
const cells = [];
const add = (type, s) => cells.push({cell_type:type, metadata:{}, ...(type === 'code' ? {execution_count:null,outputs:[]} : {}), source:s.trim().split('\n').map(s=>s+'\n')});
const md=s=>add('markdown',s), code=s=>add('code',s);
md(`# TaxaLens 0.9.2 — DINOv3 + BioCLIP
Работаем по одной ячейке. Данные остаются в прежней Foundation_v06.
После нового подключения Colab сначала снова выполните две ячейки настройки — вторая определяет команду stage().
**Если cache уже завершился сообщением про 47 shard'ов:** не повторяйте download, top-up, ingest, assemble, plan и cache; переходите сразу к шагу 9 (quality).
После cache обязателен отдельный quality gate: сомнительные музейные обзоры и этикетки автоматически отправляются на проверку, а результат показывается в preview.
Кэш JPEG общий: гибридная модель использует только одобренные локальные файлы и не скачивает исходные фотографии заново. DINOv3 и BioCLIP получают отдельные возобновляемые checkpoints.
Список отбора сохранялся отдельно от самих изображений. По наличию одного списка нельзя считать докачку законченной.
Не нажимайте «Выполнить всё»: обучение и загрузка остальных источников — отдельные этапы.`);
code(`from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys, subprocess, json
DATA_ROOT = '/content/drive/MyDrive/EntoKey/Foundation_v06'
BIOSCAN_ROOT = DATA_ROOT + '/raw/bioscan'
BIOSCAN_MANIFEST = DATA_ROOT + '/manifests/bioscan_raw.parquet'
REPO = '/content/TaxaLens'
if not Path(DATA_ROOT).is_dir():
    raise RuntimeError('STOP: папка Foundation_v06 не найдена. Проверьте аккаунт Google Drive; новая папка с данными не создавалась.')
print('Drive подключён; существующая папка найдена.')`);
code(`if not Path(REPO).exists():
    subprocess.run(['git','clone','https://github.com/SaniyaSani/TaxaLens.git',REPO], check=True)
elif not (Path(REPO)/'.git').exists():
    raise RuntimeError('STOP: REPO существует, но это не Git-репозиторий. Он не изменён.')
os.chdir(REPO)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-corpus.txt'], check=True)
def stage(name, *extra):
    import tempfile, signal
    from collections import deque
    if not Path(DATA_ROOT).is_dir():
        raise RuntimeError('Сначала подключите Google Drive')
    command = [sys.executable,'-u','scripts/run_multisource_poc_v09.py','--stage',name,
               '--data-root',DATA_ROOT,'--bioscan-root',BIOSCAN_ROOT,'--bioscan-manifest',BIOSCAN_MANIFEST,*extra]
    tail = deque(maxlen=100)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=DATA_ROOT,
                                    prefix='taxalens_'+name+'_', suffix='.log', delete=False) as log:
        print('Журнал:', log.name, flush=True)
        with subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              start_new_session=True) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                result = process.wait()
            except BaseException:
                # Stop the runner AND its downloader if the cell or logging is interrupted.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise
    if result:
        raise RuntimeError(f'STOP: этап {name}, код {result}. Журнал: {log.name}\\n' + ''.join(tail))
    print('✓ Этап завершён:',name)
print('Исправленный код и зависимости готовы.')`);
md(`## 1. Проверить, что уже сохранено — без скачивания
Эта проверка не запускает старый отбор. doctor показывает и старый bulk-формат, и новый PoC-формат: они альтернативные, оба одновременно не нужны.`);
code(`# run_multisource_poc_v09.py --stage doctor
stage('doctor')
import pandas as pd
root = Path(BIOSCAN_ROOT)
for name in ['diptera_added_families_topup_selection.csv','diptera_added_families_topup_selection_report.json','diptera_added_families_topup_download_report.json']:
    p = root/name
    print(name, 'СОХРАНЁН' if p.is_file() else 'не найден')
    if p.is_file() and p.suffix == '.json':
        print(json.dumps(json.loads(p.read_text()), ensure_ascii=False, indent=2)[:6000])
if Path(BIOSCAN_MANIFEST).is_file():
    frame = pd.read_parquet(BIOSCAN_MANIFEST)
    print('BIOSCAN: строк в сохранённом манифесте:',len(frame))
    display(frame[frame.family.isin(['Muscidae','Tachinidae'])].groupby(['family','source_split']).size().unstack(fill_value=0))`);
md(`## 2. Сохранить контрольную точку
Создаются проверенные резервные копии небольших файлов — манифеста, отбора и отчётов — рядом с оригиналами в recovery_backups.
Изображения и огромный CSV метаданных не копируются и не удаляются.`);
code(`stage('checkpoint')`);
md(`## 3. Продолжить BIOSCAN top-up
Если сохранённый список и отчёт целые, появится **REUSING SAVED TOP-UP**. Тогда большой CSV не перечитывается.
Проверяются существующие JPEG и скачиваются только отсутствующие/повреждённые. Старый манифест заменяется лишь после успешной загрузки и проверки сохранения прежних ID.
Если список отсутствует, начнётся новый отбор с видимым прогрессом. Если список повреждён, программа остановится, не перезаписывая его.`);
code(`# run_multisource_poc_v09.py --stage bioscan-topup
stage('bioscan-topup')
frame = pd.read_parquet(BIOSCAN_MANIFEST)
print('BIOSCAN TOP-UP СОХРАНЁН. Всего строк:',len(frame))
display(frame[frame.family.isin(['Muscidae','Tachinidae'])].groupby(['family','source_split']).size().unstack(fill_value=0))`);
md(`## Здесь можно спокойно закончить на сегодня
Отбор, изображения и новый манифест — на Drive. При следующем запуске повторите настройку и проверки.
Следующие источники не запускаются автоматически.`);
md(`## 4. iNaturalist — ограниченное подмножество
До 30 000 изображений (метаданные и ссылки, не сами JPEG) из **официального iNaturalist Research-grade Observations dataset через GBIF**.
Это данные iNaturalist, отдельно от музейного GBIF. Учитываются лицензии изображений CC0/CC-BY/CC-BY-SA, авторство и исходные наблюдения.
Отбор по целевым семействам, не статистически случайная выборка. Некоторые квоты могут не заполниться; фактический размер указан в отчёте. Четыре огромные мировые таблицы не нужны.
Ссылки: [iNaturalist dataset](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7), [GBIF API](https://techdocs.gbif.org/en/openapi/v1/occurrence).
Страницы сохраняются по семействам. При сетевой ошибке повторяется эта же ячейка.`);
code(`stage('download','--source','inat','--reuse-existing-bioscan')`);
md(`## 5. GBIF — музейные экземпляры
До 25 000 записей с открыто лицензированными изображениями, только PRESERVED_SPECIMEN, ограниченный поиск.
Для этого PoC-пути аккаунт и отправка серверного задания GBIF не нужны. Большие научные выгрузки с DOI остаются отдельным режимом.`);
code(`stage('download','--source','gbif','--reuse-existing-bioscan')`);
md(`## 6. DiSSCo
До 15 000 экземпляров; число пригодных изображений может быть меньше. Контрольная точка позволяет продолжать страницы.
DiSSCo сохраняется как кандидатный музейный источник, но не включается в обучение ради количества. Если после quality gate ни одна его картинка не показывает достаточно крупный экземпляр, источник честно переносится целиком в quarantine.`);
code(`stage('download','--source','dissco','--reuse-existing-bioscan')`);
md(`## 7. Проверить источники и собрать корпус
Все входы проверяются **до** обработки. Если чего-то не хватает, выполнение остановится с одним понятным сообщением.
100 000 — целевой объём, не утверждение о полученном результате. Четыре источника должны действительно присутствовать.`);
code(`stage('ingest')`);
code(`stage('assemble')`);
code(`stage('plan')
report = json.loads((Path(DATA_ROOT)/'poc_v09/training_plan_report.json').read_text())
print('РЕАЛЬНО отобрано:',report['selected_total'])
print('По источникам:',report['selected_by_source'])`);
md(`## 8. Кэширование изображений — только когда готов весь план
Этот этап скачивает JPEG и занимает место. Повторный запуск использует уже сохранённые картинки.
Он не запускает обучение. Исходные JPEG остаются неизменными и позже проходят отдельную проверку качества.`);
code(`stage('cache')`);
md(`## 9. Обязательная проверка изображений перед embedding
Переключите Colab на **T4 GPU**, затем запустите ячейку. OWLv2 ищет муху на неоднородных музейных кадрах GBIF/DiSSCo.
Если экземпляр достаточно детальный, но вокруг много фона, создаётся отдельный crop. Исходный JPEG не изменяется.
Кадры без найденной мухи, с крошечным экземпляром, одними этикетками или повреждённым файлом уходят в quarantine, а не в обучение.
Очень большие музейные JPEG обрабатываются уменьшенной копией в памяти; исходный файл на Drive остаётся неизменным. Поэтому предупреждение DecompressionBomb из cache не требует новой загрузки.
BIOSCAN и iNaturalist проходят техническую проверку без строгого музейного детектора. Для текущего PoC обязательны BIOSCAN, iNaturalist и качественный GBIF. DiSSCo остаётся дополнительным: если полезных кадров нет, он сохраняется в quarantine и не блокирует модель.
Этап держит в памяти только две музейные картинки и сохраняет checkpoints на Drive. После отключения или SIGKILL запускайте эту же ячейку повторно: готовые части будут показаны как reused.`);
code(`import pandas as pd
subprocess.run([sys.executable,'-m','pip','install','-q','torch','torchvision','transformers>=4.56'], check=True)
stage('quality')
quality_report = json.loads((Path(DATA_ROOT)/'poc_v09/image_quality_report.json').read_text())
print('Принято:', quality_report['accepted_rows'])
print('Создано crops:', quality_report['cropped_rows'])
print('Оставлено для проверки:', quality_report['quarantined_rows'])
display(pd.DataFrame(quality_report['by_source']).T)
if quality_report.get('missing_optional_sources'):
    print('Не включены из-за качества:', ', '.join(quality_report['missing_optional_sources']))
preview = Path(DATA_ROOT)/'poc_v09/image_quality_review_contact_sheet.jpg'
if preview.is_file():
    from IPython.display import Image as DisplayImage
    display(DisplayImage(filename=str(preview)))`);
md(`## 10. Необязательный тест DINOv3 + BioCLIP
Для этого нужен GPU Colab и доступ к модели DINOv3 на Hugging Face (при запросе доступа следуйте условиям модели; токены не вставляйте в чат).
Один запуск проверяет один shard DINOv3 и тот же shard BioCLIP. Оба результата сохраняются отдельно; повторный запуск продолжает незавершённое.
В Colab используется BioCLIP 2 ViT-L/14: новая BioCLIP 2.5 Huge требует заметно больше памяти и остаётся будущим кластерным сравнением.
По умолчанию выключено. Меняйте флажок только после успешного quality gate.`);
code(`RUN_EMBEDDING_TEST = False
if RUN_EMBEDDING_TEST:
    subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-embeddings.txt'], check=True)
    stage('embed','--max-shards','1') # --max-shards 1
else:
    print('Тест эмбеддингов выключен. Данные сохранены.')`);
md(`## 11. Позже: полные эмбеддинги, обучение и сравнение
Не запускайте сегодня автоматически. Сначала проверяется один shard; затем весь набор.
Training строит три версии: DINOv3, BioCLIP и equal-weight fusion на одном и том же наборе экземпляров. Лучшая версия выбирается только по сохранённому сравнению; заранее считать fusion победителем нельзя.`);
code(`RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-embeddings.txt'], check=True)
    stage('embed')
    stage('train')
    stage('evaluate')`);
md(`## 12. Определительные ключи: независимая офлайн-проверка`);
code(`subprocess.run([sys.executable,'scripts/find_genus_keys.py','--family','Syrphidae','--genera','Eristalis,Helophilus','--offline'],cwd=REPO,check=True)`);
const notebook={cells,metadata:{kernelspec:{display_name:'Python 3',language:'python',name:'python3'},language_info:{name:'python',version:'3.13'},colab:{name:'TaxaLens_Recovery_v0_9_2_Hybrid.ipynb'}},nbformat:4,nbformat_minor:5};
fs.writeFileSync(path.join(root,'notebooks/TaxaLens_v0.9_Multisource_PoC_Colab.ipynb'),JSON.stringify(notebook,null,2)+'\n');
if (process.argv[2]) {
  const base='cc76d6dfb92badfd85367e6f13897426914e3b42';
  // git diff includes newly authored files once staged, and no dataset files.
  const patch=execFileSync('git',['diff','--binary',base],{cwd:root,maxBuffer:5e6});
  const b64=patch.toString('base64'), sha=crypto.createHash('sha256').update(patch).digest('hex');
  const expected={};
  const names=execFileSync('git',['diff','--name-only',base],{cwd:root,encoding:'utf8'}).trim().split('\n');
  for(const name of names) if(fs.existsSync(path.join(root,name))) expected[name]=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,name))).digest('hex');
  const portable=structuredClone(notebook);
  portable.cells[1].source=portable.cells[1].source.map(s=>s.replace(
    "REPO = '/content/TaxaLens'", `REPO = '/content/TaxaLens_recovery_v092_${sha.slice(0,12)}'`
  ));
  const setup=portable.cells[2].source.join('');
  const start=setup.indexOf('os.chdir(REPO)');
  const bootstrap=`import base64, hashlib, tempfile\nBASE_COMMIT = '${base}'\nPATCH = base64.b64decode('${b64}')\nPATCH_SHA = '${sha}'\nEXPECTED_FILES = ${JSON.stringify(expected)}\nif hashlib.sha256(PATCH).hexdigest() != PATCH_SHA:\n    raise RuntimeError('Повреждён пакет исправлений')\nif not Path(REPO).exists():\n    subprocess.run(['git','clone','https://github.com/SaniyaSani/TaxaLens.git',REPO],check=True)\nmarker = Path(REPO)/'.taxalens_recovery_patch'\nif marker.exists():\n    if marker.read_text().strip() != PATCH_SHA:\n        raise RuntimeError('В папке уже другой recovery-пакет. Он не изменён.')\nelse:\n    dirty = subprocess.run(['git','-C',REPO,'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()\n    if dirty:\n        raise RuntimeError('В recovery-папке есть локальные изменения. Они не перезаписаны.')\n    subprocess.run(['git','-C',REPO,'checkout','--detach',BASE_COMMIT],check=True)\n    with tempfile.NamedTemporaryFile(suffix='.patch') as f:\n        f.write(PATCH); f.flush()\n        subprocess.run(['git','-C',REPO,'apply','--check',f.name],check=True)\n        subprocess.run(['git','-C',REPO,'apply',f.name],check=True)\n    marker.write_text(PATCH_SHA)\nfor name, expected in EXPECTED_FILES.items():\n    if hashlib.sha256((Path(REPO)/name).read_bytes()).hexdigest() != expected:\n        raise RuntimeError('Файл изменился после исправления: '+name+'. Автоматическая перезапись отключена.')\nprint('Recovery 0.9.1 проверен. Старая /content/TaxaLens не изменена.')\n`;
  portable.cells[2].source=(bootstrap+setup.slice(start)).split('\n').map(s=>s+'\n');
  portable.cells[2].source=portable.cells[2].source.map(s=>s.replace(
    'Recovery 0.9.1 проверен.', 'Recovery 0.9.2 hybrid проверен.'
  ));
  fs.mkdirSync(path.dirname(process.argv[2]),{recursive:true});
  fs.writeFileSync(process.argv[2],JSON.stringify(portable,null,2)+'\n');
  console.log('Portable recovery notebook:',process.argv[2],fs.statSync(process.argv[2]).size,'bytes');
}
