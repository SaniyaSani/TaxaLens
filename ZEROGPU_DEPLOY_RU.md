# TaxaLens: бесплатный публичный сайт на Hugging Face ZeroGPU

Это **однократная публикация**, а не способ пользоваться сайтом. После неё Colab
можно закрыть: TaxaLens получает публичную ссылку, которую можно отправить Оливеру.

## Что публикуется

- код интерфейса — в публичный Hugging Face Space;
- `classifiers.joblib`, retrieval-векторы и metadata — в отдельный **приватный**
  model repository;
- исходные JPEG, training corpus и Google Drive не загружаются;
- токен хранится как зашифрованный Space secret и не виден посетителям.

## Однократный запуск из Colab, где подключён Drive

```python
import os, subprocess, sys
from pathlib import Path

REPO = "/content/TaxaLens_public"
if not Path(REPO).exists():
    subprocess.run(
        ["git", "clone", "https://github.com/SaniyaSani/TaxaLens.git", REPO],
        check=True,
    )
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=1.0"],
    check=True,
)

MODEL_DIR = "/content/drive/MyDrive/EntoKey/Foundation_v06/models_poc_v09"
subprocess.run(
    [
        sys.executable,
        "scripts/publish_hf_demo.py",
        "--model-dir", MODEL_DIR,
        "--space-name", "TaxaLens",
    ],
    cwd=REPO,
    check=True,
)
```

Hugging Face сначала покажет безопасное окно входа. Используйте аккаунт, которому
уже разрешён DINOv3. После загрузки модели скрипт попросит отдельный **Read token**.
Создайте его на `https://huggingface.co/settings/tokens` и вставьте в появившееся
скрытое поле. Токен не появится в коде или выводе; он сохранится только как
зашифрованный Space secret.

После сообщения `PUBLIC TAXALENS DEPLOYMENT STARTED` откройте напечатанную ссылку
`https://huggingface.co/spaces/.../TaxaLens`. Первая сборка и первое пробуждение
могут занять несколько минут. Затем посетитель загружает фото прямо на сайте.

## Ограничение бесплатного режима

ZeroGPU использует очередь и засыпает без посетителей. Первый запрос после сна
может быть медленнее, но работа сайта не зависит от Colab. Hugging Face разрешает
создание бесплатного ZeroGPU Space только подходящим аккаунтам с подтверждённой
почтой и хорошим статусом; для нового аккаунта может действовать ограничение по
возрасту. Скрипт остановится с понятным сообщением и не потеряет уже загруженную
приватную модель.

Если аккаунт пока слишком новый для ZeroGPU, временный полностью бесплатный
публичный адрес можно открыть с Mac командой `GRADIO_SHARE=1 python app.py` после
указания `TAXALENS_MODEL_DIR`. Такой адрес работает, пока Mac и приложение
включены; модель и фотографии при этом не проходят через Colab.
