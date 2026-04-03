# videoQwen — Video Analyzer

Сервис анализа видео на базе Qwen VL моделей (vLLM). Принимает видеофайл или URL, возвращает: описание, ориентацию, категории, водяные знаки, SEO-метаданные, опционально — исполнителей.

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                  video_processor.py                 │
│              FastAPI :8000  (основной сервис)        │
└────────────────┬───────────────┬────────────────────┘
                 │               │
         Pass 1  │               │  Cats (tagger)
    25 frames    │               │  4 grids или 16 frames
                 ▼               ▼
    ┌────────────────┐   ┌───────────────────────┐
    │  model_server  │   │   tagger/VideoTagger  │
    │  :8080/generate│   │  (scene→model)        │
    │  vLLM          │   └───────────────────────┘
    └────────────────┘               │ HTTP → model_server
```

### Компоненты

| Файл | Назначение |
|------|-----------|
| `model_server.py` | vLLM-сервер на порту `:8080`. Принимает prompt + base64-изображения, возвращает текст |
| `video_processor.py` | FastAPI-сервис на порту `:8000`. Оркестрирует весь пайплайн |
| `tagger/tagger.py` | `VideoTagger` — определение категорий по сценам |
| `tagger/frames.py` | Извлечение кадров, детекция сцен, сборка 2×2 гридов |
| `tagger/categories.py` | Загрузка `categories.json`, построение промпта, canonical-карты, JSON-схемы |
| `tagger/model.py` | HTTP-клиент к `model_server` |
| `tagger/validate.py` | Постобработка категорий: разрешение конфликтов, логические правила |
| `categories.json` | Эталонный список категорий с алиасами и описаниями |
| `performer_finder.py` | Модуль распознавания исполнителей по face-embeddings |
| `setup.sh` | Скрипт деплоя: создаёт venv, ставит зависимости, генерирует start-скрипты |

---

## Пайплайн обработки одного видео

```
video_processor.process_video_v2()
│
├── Pass 1 — Анализ (25 кадров, 4%–92% видео)
│   POST → model_server (25 изображений)
│   Возвращает:
│     • description  — текстовое описание (vulgar/clean/cinematic)
│     • orientation  — straight | gay | shemale
│     • watermarks   — массив видимых водяных знаков/брендов
│
├── Cats — Определение категорий (VideoTagger)
│   1. detect_scenes() — гистограммная корреляция (2fps, 64×36 downscale) → до 8 сцен
│   2. Каждая сцена → 4 временных окна × 4 кадра = 16 кадров на сцену
│   3. TAGGER_MODE=grid:  16 кадров → 4 грида 2×2 → 4 изображения в модель
│      TAGGER_MODE=video: 16 кадров → 16 изображений напрямую в модель
│   4. vLLM Structured Output (guided_json enum) → только имена из categories.json
│   5. Агрегация по частоте, Solo требует присутствия во всех сценах
│   6. validate_categories() — разрешение конфликтов по правилам
│   7. Сортировка по частоте → обрезка до category_count
│
├── Pass SEO — SEO-метаданные (text-only)
│   POST → model_server с description + categories + orientation
│   Возвращает: meta_title, meta_description, seo_description, primary_tags, secondary_tags
│
├── SEO-translate (для каждого дополнительного языка)
│
├── Performer recognition (опционально)
│   100 кадров → face embeddings (InsightFace) → clustering → match против DB
│
└── Результат: JSON со всеми метаданными
```

---

## Выбор модели и режима

### Модели (MODEL_PRESET в `start_model_server.sh`)

| `MODEL_PRESET` | Модель | VRAM | Особенности |
|---|---|---|---|
| `qwen3vl` | Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 | ~40 GB | Проверенная vision модель, быстрее |
| `qwen35` | Qwen/Qwen3.5-27B-FP8 | ~28 GB | Нативный multimodal, меньше VRAM |

**Переключение модели** — в `start_model_server.sh`:
```bash
export MODEL_PRESET="qwen3vl"   # или "qwen35"
```
После изменения **обязательно перезапустить model server**.

### Режим таггера (TAGGER_MODE в `start_app.sh`)

| `TAGGER_MODE` | Описание | Изображений на сцену |
|---|---|---|
| `grid` | 16 кадров → 4 грида 2×2 → 4 изображения в модель | 4 |
| `video` | 16 кадров напрямую в модель | 16 |

**grid** — рекомендуется для `qwen3vl` (меньше токенов, быстрее).  
**video** — для `qwen35` (видит каждый кадр отдельно, больше токенов).

**Переключение режима** — в `start_app.sh`:
```bash
export TAGGER_MODE="grid"   # или "video"
```
После изменения перезапустить app server.

### Рекомендуемые комбинации

| Сценарий | MODEL_PRESET | TAGGER_MODE |
|---|---|---|
| Быстро, стабильно | `qwen3vl` | `grid` |
| Меньше VRAM | `qwen35` | `grid` |

---

## Деплой

```bash
bash setup.sh
```

Интерактивно предлагает на выбор:
- **1) Qwen3-VL-30B-FP8** → `MODEL_PRESET=qwen3vl`, `TAGGER_MODE=grid`
- **2) Qwen3.5-27B-FP8** → `MODEL_PRESET=qwen35`, `TAGGER_MODE=video`

Генерирует `start_model_server.sh` и `start_app.sh` с нужными переменными.

### Запуск

```bash
# Терминал 1 — модельный сервер
bash start_model_server.sh

# Терминал 2 — основной сервис
bash start_app.sh
```

### Ручное переключение без setup.sh

```bash
# Переключить модель
sed -i 's/MODEL_PRESET="qwen3vl"/MODEL_PRESET="qwen35"/' start_model_server.sh
# затем рестарт model server

# Переключить режим таггера
sed -i 's/TAGGER_MODE=grid/TAGGER_MODE=video/' start_app.sh
# затем рестарт app server
```

### CLI-тестирование категорий

```bash
python tag_video.py video.mp4
python tag_video.py video.mp4 --passes 4 --scenes 8
python tag_video.py video.mp4 --min-count 2 --output results.json
```

---

## API

### `POST /api/v2/analyze`
Основной JSON API. Принимает URL видео, возвращает `task_id`.

```json
{
  "video_url": "https://example.com/video.mp4",
  "languages": ["en", "de"],
  "style": "standard",
  "webhook_url": "https://your-server.com/webhook",
  "client_reference_id": "optional-id",
  "category_count": 10,
  "tag_count": 5,
  "secondary_tag_count": 7
}
```

### `GET /api/v2/status/{task_id}`
Статус задачи. Готово — полный результат.

### `POST /v2/analyze-upload`
Загрузка файла напрямую (multipart form).
```bash
curl -X POST http://localhost:8000/v2/analyze-upload \
  -H 'X-API-Key: ...' \
  -F "files=@/path/to/video.mp4" \
  -F "language=English"
```

### `POST /v2/analyze-url`
Скачивание через yt-dlp по URL страницы.
```bash
curl -X POST http://localhost:8000/v2/analyze-url \
  -F "url=https://example.com/video-page/" \
  -F "language=English"
```

---

## Webhook

```json
{
  "success": true,
  "task_id": "...",
  "result": {
    "orientation": "straight",
    "description": "...",
    "categories": ["Blonde", "Blowjob", "Gangbang"],
    "watermarks": ["brazzers.com"],
    "primary_tags": ["..."],
    "secondary_tags": ["..."],
    "meta_title": "...",
    "meta_description": "...",
    "seo_description": "...",
    "performers": [{"name": "Jane Doe", "confidence": 87}],
    "preview_thumbnail": "data:image/jpeg;base64,...",
    "meta_title_de": "...",
    "meta_description_de": "...",
    "seo_description_de": "..."
  }
}
```

---

## Конфигурация (env vars)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `MODEL_PRESET` | `qwen3vl` | Модель: `qwen3vl` или `qwen35` |
| `MODEL_SERVER_URL` | `http://localhost:8080/generate` | URL модельного сервера |
| `TAGGER_MODE` | `grid` | Режим таггера: `grid` или `video` |
| `UPLOAD_DIR` | `/workspace/video/videos` | Директория для загрузок |
| `OUTPUT_DIR` | `/workspace/video/result` | Директория результатов |
| `PERFORMER_DB_PATH` | `/workspace/my_performers.pkl` | БД face-embeddings исполнителей |
| `API_KEY` | `` (пусто = отключено) | API ключ для `/api/v2/analyze` |

---

## Стили описания (`style`)

| Значение | Описание |
|----------|----------|
| `standard` | Вульгарный, explicit стиль (по умолчанию) |
| `clean` | Тактичное описание для mainstream-платформ |
| `cinematic` | Стиль кинокритика — композиция, свет, химия |

---

## Категории

Полный список хранится в `categories.json`. Структура:

```json
{
  "section_key": {
    "title": "Section Title",
    "categories": [
      {
        "name": "Canonical Name",
        "description": "Rule for tagging this category",
        "aliases": ["alias1", "alias2"]
      }
    ]
  }
}
```

`VideoTagger` при старте строит из него:
- промпт с описаниями для модели
- canonical-карту алиасов для нормализации вывода
- JSON-схему с enum всех допустимых имён (vLLM constrained decoding)

Менять категории — только в `categories.json`, код трогать не нужно. После изменения — рестарт app server.
