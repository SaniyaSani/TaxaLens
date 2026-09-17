# TaxaLens: запустить сайт с готовой моделью в Colab

После подключения Google Drive и загрузки текущего кода выполните одну ячейку. Путь должен
указывать на папку, где лежат `classifiers.joblib`, `retrieval_vectors.npy` и
`retrieval_metadata.csv`.

```python
import os, subprocess, sys, time
from pathlib import Path
from google.colab import output

REPO = "/content/TaxaLens"
MODEL_DIR = "/content/drive/MyDrive/EntoKey/Foundation_v06/poc_v09/models_poc_v09"

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "requirements.txt",
    "-r", "requirements-embeddings.txt",
], cwd=REPO, check=True)

# DINOv3 is gated. In a new Colab runtime log in with the Hugging Face account
# that already has accepted DINOv3 access. Never paste the token into notebook code.
from huggingface_hub import notebook_login
notebook_login(skip_if_logged_in=True)

required = ["classifiers.joblib", "retrieval_vectors.npy", "retrieval_metadata.csv"]
missing = [name for name in required if not (Path(MODEL_DIR) / name).exists()]
if missing:
    raise FileNotFoundError(f"Неполная папка модели {MODEL_DIR}; отсутствуют: {missing}")

server = subprocess.Popen(
    [sys.executable, "scripts/serve_taxalens.py", "--model-dir", MODEL_DIR, "--port", "8000"],
    cwd=REPO,
)
time.sleep(4)
output.serve_kernel_port_as_window(8000)
```

Откроется настоящий интерфейс с готовыми весами. Он показывает family/genus/species-кандидатов,
ближайшие эталонные изображения, список ключей и Morphology Key Navigator. Для Chironomidae
Можно нажать на любой из четырёх genus-кандидатов: после нажатия список опубликованных ключей и
морфологический чек-лист перестраиваются для выбранного рода. Для Chironomidae навигатор
отдельно просит проверить пол/стадию, антенну, крыло и squama, щетинки груди,
голени/лапки и гипопигий.

Это окно удобно показывать Оливеру через демонстрацию экрана. Публичная временная ссылка требует
отдельного туннеля и должна включаться осознанно: загружаемые изображения тогда проходят через
сторонний сервис. Сам Colab и открытая вкладка не превращают приложение в постоянный сайт.
