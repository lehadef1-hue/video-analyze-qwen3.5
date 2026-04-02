# videoqwen_merged — Video Analyzer

Сервис анализа видео на базе **Qwen3-VL-30B**. Принимает видеофайл или URL, возвращает: описание, ориентацию, категории, водяные знаки, ключевые сцены, SEO-метаданные, опционально — исполнителей.

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                  video_processor.py                 │
│              FastAPI :8001  (основной сервис)       │
└────────────────┬───────────────┬───────────────────┘
                 │               │
         Pass 1  │               │  Cats (tagger)
    25 frames    │               │  scene grids
                 ▼               ▼
    ┌────────────────┐   ┌───────────────────────┐
    │  model_server  │   │   tagger/VideoTagger  │
    │  :8080/generate│   │  (scene→grid→model)   │
    │  Qwen3-VL-30B  │   └───────────────────────┘
    │    (vLLM)      │               │
    └────────────────┘               │ HTTP → model_server
```

### Компоненты

| Файл | Назначение |
|------|-----------|
| `model_server.py` | vLLM-сервер Qwen3-VL-30B на порту `:8080`. Принимает prompt + base64-изображения, возвращает текст |
| `video_processor.py` | FastAPI-сервис на порту `:8001`. Оркестрирует весь пайплайн |
| `tagger/tagger.py` | `VideoTagger` — определение категорий по сценам и гридам |
| `tagger/frames.py` | Извлечение кадров, детекция сцен, сборка 2×2 гридов |
| `tagger/categories.py` | Загрузка `categories.json`, построение промпта и canonical-карты |
| `tagger/model.py` | HTTP-клиент к `model_server` (используется внутри `VideoTagger`) |
| `tagger/validate.py` | Постобработка категорий: разрешение конфликтов, логические правила |
| `categories.json` | Эталонный список категорий с алиасами и описаниями |
| `performer_finder.py` | Опциональный модуль распознавания исполнителей по face-embeddings |

---

## Пайплайн обработки одного видео

```
video_processor.process_video_v2()
│
├── Pass 1 — Анализ (25 кадров, 4%–92% видео)
│   POST → model_server
│   Возвращает:
│     • description  — текстовое описание (vulgar/clean/cinematic)
│     • orientation  — straight | gay | shemale
│     • watermarks   — массив видимых водяных знаков/брендов
│     • key_scenes   — 5–8 ключевых моментов с временными метками
│
├── Cats — Определение категорий (VideoTagger)
│   1. detect_scenes() — гистограммная корреляция → до 8 сцен
│   2. Каждая сцена → 4 временных окна → из каждого окна 4 кадра → 2×2 грид
│   3. Все гриды сцены → 1 вызов model_server
│   4. Модель возвращает orientation + ["Cat1", "Cat2", ...]
│   5. Агрегация: union всех детекций с частотными счётчиками
│   6. Orientation = взвешенное большинство (вес ∝ длительность сцены)
│   7. validate_categories() — разрешение конфликтов по правилам
│   Источник категорий: categories.json (алиасы, canonical-имена)
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
├── Performer recognition (опционально, если performer_finder доступен)
│   100 кадров → face embeddings → clustering → match против DB
│
└── Сохранение результатов
    {base_name}_meta.json — все метаданные
    {base_name}_thumb.jpg — миниатюра (если есть)
    {base_name}_frame_NNN.jpg — лучшие кадры (если есть)
```

---

## VideoTagger — детали

```
categories.json
     │
     ▼
load_categories() → build_category_prompt() → prompt с описаниями из JSON
build_canonical_map() → {alias_lower: "Canonical Name"}

Для каждой сцены:
  [grid1, grid2, grid3, grid4]  ← 4 грида 2×2 = 16 кадров
       └──────────────────────► 1 вызов model_server
                                 → {"orientation": "...", "categories": [...]}

parse_model_output(raw, canonical_map) → (orientation, [canonical_cats])

После всех сцен:
  category_counter: {"Anal": 3, "Blonde": 7, ...}
  orientation_counter: {"straight": 5, ...}  (взвешенный)

validate_categories(raw_cats, orientation, counts=category_counter)
  → финальный список категорий
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
Статус задачи. Пока обрабатывается — `{"status": "processing", "stage": "..."}`.
Готово — полный результат.

### `POST /v2/analyze-upload`
Загрузка файла напрямую (multipart form). Параметры: `files`, `language`, `style`.

### `POST /v2/analyze-url`
Скачивание через yt-dlp. Параметры формы: `url`, `language`, `style`.

### `GET /v2/status/{task_id}`
Статус задачи для UI-эндпоинтов.

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
    "categories": ["Blonde", "Blowjob", "..."],
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
| `MODEL_SERVER_URL` | `http://localhost:8080/generate` | URL vLLM-сервера |
| `UPLOAD_DIR` | `/workspace/video/videos` | Директория для загрузок |
| `OUTPUT_DIR` | `/workspace/video/result` | Директория результатов |
| `PERFORMER_DB_PATH` | `/workspace/my_performers.pkl` | БД face-embeddings исполнителей |
| `API_KEY` | `` (пусто = отключено) | Bearer-ключ для `/api/v2/analyze` |

---

## Запуск

### 1. Модельный сервер
```bash
python model_server.py
# Слушает :8080
# Модель: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
# Требуется GPU с ≥40 GB VRAM
```

### 2. Основной сервис
```bash
uvicorn video_processor:app --host 0.0.0.0 --port 8001
```

### 3. CLI-тестирование категорий
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

`VideoTagger` загружает этот файл при старте — менять категории достаточно только в `categories.json`, код трогать не нужно.
