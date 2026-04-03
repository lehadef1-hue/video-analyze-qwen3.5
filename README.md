# videoQwen — Video Analyzer

Сервис анализа видео на базе **Qwen3-VL-30B-A3B-Instruct-FP8** (vLLM) или **Qwen3.5-27B GGUF** (llama.cpp). Принимает видеофайл или URL, возвращает: описание, ориентацию, категории, водяные знаки, SEO-метаданные, опционально — исполнителей.

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                  video_processor.py                 │
│              FastAPI :8000  (основной сервис)        │
└────────────────┬───────────────┬────────────────────┘
                 │               │
         Pass 1  │               │  Cats (tagger)
    25 frames    │               │  scene grids / frames
                 ▼               ▼
    ┌────────────────┐   ┌───────────────────────┐
    │  model_server  │   │   tagger/VideoTagger  │
    │  :8080/generate│   │  (scene→model)        │
    │  Qwen3-VL-30B  │   └───────────────────────┘
    │  vLLM / llama  │               │ HTTP → model_server
    └────────────────┘
```

### Компоненты

| Файл | Назначение |
|------|-----------|
| `model_server.py` | vLLM-сервер Qwen3-VL-30B на порту `:8080`. Принимает prompt + base64-изображения (или video+fps), возвращает текст |
| `model_server_llama.py` | llama.cpp-сервер (GGUF). Та же `/generate` API |
| `video_processor.py` | FastAPI-сервис на порту `:8000`. Оркестрирует весь пайплайн |
| `tagger/tagger.py` | `VideoTagger` — определение категорий по сценам |
| `tagger/frames.py` | Извлечение кадров, детекция сцен, сборка 2×2 гридов |
| `tagger/categories.py` | Загрузка `categories.json`, построение промпта, canonical-карты, JSON-схемы |
| `tagger/model.py` | HTTP-клиент к `model_server` (используется внутри `VideoTagger`) |
| `tagger/validate.py` | Постобработка категорий: разрешение конфликтов, логические правила |
| `categories.json` | Эталонный список категорий с алиасами и описаниями |
| `performer_finder.py` | Опциональный модуль распознавания исполнителей по face-embeddings |
| `setup.sh` | Скрипт деплоя: создаёт venv, ставит зависимости, генерирует start-скрипты |

---

## Пайплайн обработки одного видео

```
video_processor.process_video_v2()
│
├── Pass 1 — Анализ (25 кадров, 4%–92% видео)
│   POST → model_server (изображения в image mode)
│   Возвращает:
│     • description  — текстовое описание (vulgar/clean/cinematic)
│     • orientation  — straight | gay | shemale
│     • watermarks   — массив видимых водяных знаков/брендов
│
├── Cats — Определение категорий (VideoTagger)
│   1. detect_scenes() — гистограммная корреляция (2fps, 64×36 downscale) → до 8 сцен
│   2. Каждая сцена → 4 временных окна × 4 кадра = 16 кадров
│
│   Режим "grid" (vLLM, по умолчанию):
│     16 кадров → 4 грида 2×2 → 1 вызов model_server
│
│   Режим "video" (llama.cpp):
│     16 кадров отправляются как video sequence (fps=2) → 1 вызов model_server
│
│   3. vLLM Structured Output (guided_json enum):
│      JSON-схема со всеми допустимыми именами категорий → constrained decoding
│      Модель физически не может вернуть имя не из списка
│   4. Агрегация: union всех детекций с частотными счётчиками
│   5. validate_categories() — разрешение конфликтов по правилам
│
├── Pass SEO — SEO-метаданные (text-only, без изображений)
│   POST → model_server с description + categories + orientation
│   Возвращает:
│     • meta_title        — 50–60 символов
│     • meta_description  — 140–160 символов
│     • seo_description   — 80–120 слов
│     • primary_tags      — до N long-tail фраз
│     • secondary_tags    — до N коротких фраз
│
├── SEO-translate (для каждого дополнительного языка)
│   Переводит meta_title + meta_description + seo_description
│
├── Performer recognition (опционально)
│   100 кадров → face embeddings (InsightFace) → clustering → match против DB
│
└── Сохранение результатов
    {base_name}_meta.json — все метаданные
```

---

## VideoTagger — детали

```
categories.json
     │
     ├── build_category_prompt()   → текст с описаниями для промпта
     ├── build_canonical_map()     → {alias_lower: "Canonical Name"}
     └── build_guided_schema()     → JSON schema enum для vLLM structured output

Для каждой сцены (grid mode):
  [grid1, grid2, grid3, grid4]  ← 4 грида 2×2 = 16 кадров
       └──────────────────────► 1 вызов model_server (guided_json)
                                 → {"categories": ["Cat1", "Cat2", ...]}

Для каждой сцены (video mode):
  [frame1 ... frame16]  ← 16 сырых кадров + fps=2
       └──────────────► 1 вызов model_server (guided_json)
                         → {"categories": ["Cat1", "Cat2", ...]}

После всех сцен:
  category_counter: {"Anal": 3, "Blonde": 7, ...}
  → лог: "█████ 5x  Gangbang" для отладки

validate_categories(raw_cats, orientation, counts=category_counter)
  → финальный список категорий
```

### Выбор режима (TAGGER_MODE)

| Значение | Бэкенд | Описание |
|----------|--------|----------|
| `grid` | vLLM (default) | 2×2 грид-изображения |
| `video` | llama.cpp | Сырые кадры + fps |

Задаётся переменной окружения `TAGGER_MODE` в `start_app.sh`. `setup.sh` генерирует правильный скрипт автоматически.

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
Статус задачи. Пока обрабатывается — `{"status": "processing", "stage": "..."}`. Готово — полный результат.

### `POST /v2/analyze-upload`
Загрузка файла напрямую (multipart form). Параметры: `files`, `language`, `style`.

### `POST /v2/analyze-url`
Скачивание через yt-dlp. Параметры формы: `url`, `language`, `style`.

---

## Webhook

После завершения обработки сервис делает `POST` на `webhook_url`:

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
| `MODEL_SERVER_URL` | `http://localhost:8080/generate` | URL модельного сервера |
| `UPLOAD_DIR` | `/workspace/video/videos` | Директория для загрузок |
| `OUTPUT_DIR` | `/workspace/video/result` | Директория результатов |
| `PERFORMER_DB_PATH` | `/workspace/my_performers.pkl` | БД face-embeddings исполнителей |
| `API_KEY` | `` (пусто = отключено) | Bearer-ключ для `/api/v2/analyze` |
| `TAGGER_MODE` | `grid` | Режим тaggerа: `grid` или `video` |

---

## Деплой

```bash
bash setup.sh
```

Интерактивно спрашивает бэкенд:
- **1) vLLM** — Qwen3-VL-30B-A3B-Instruct-FP8, ~40 GB VRAM, TAGGER_MODE=grid
- **2) llama.cpp** — Qwen3.5-27B GGUF + mmproj, ~28 GB VRAM, TAGGER_MODE=video

Генерирует `start_model_server.sh` (или `start_model_server_llama.sh`) и `start_app.sh`.

### Запуск

```bash
# Терминал 1 — модельный сервер
bash start_model_server.sh         # vLLM
# или
bash start_model_server_llama.sh   # llama.cpp

# Терминал 2 — основной сервис
bash start_app.sh
```

### CLI-тестирование категорий

```bash
python tag_video.py video.mp4
python tag_video.py video.mp4 --passes 4 --scenes 8
python tag_video.py video.mp4 --min-count 2 --output results.json
```

---

## Стили описания (`style`)

| Значение | Описание |
|----------|----------|
| `standard` | Вульгарный, explicit стиль (по умолчанию) |
| `clean` | Тактичное описание для mainstream-платформ |
| `cinematic` | Стиль кинокритика — композиция, свет, химия |

---

## Категории

Полный список и алиасы хранятся в `categories.json`. Структура:

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
- JSON-схему с enum всех допустимых имён для vLLM structured output

Менять категории — только в `categories.json`, код трогать не нужно.
