# PianoMagic Backend

API для транскрипции аудио в ноты для фортепиано.

## Технологии
- FastAPI — веб-фреймворк
- Basic Pitch (Spotify) — AI-транскрипция audio→MIDI
- MuseScore — MIDI→PDF
- yt-dlp — скачивание с YouTube

## Деплой на Render.com (бесплатно)

1. Создайте новый репозиторий на GitHub и залейте туда эти файлы
2. Зарегистрируйтесь на [render.com](https://render.com)
3. New → Web Service → Connect your GitHub repo
4. Укажите:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
5. Нажмите Create Web Service
6. Через 5–10 минут получите URL вида `https://pianomagic-api.onrender.com`

## API Endpoints

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/transcribe/file` | Загрузка файла |
| POST | `/transcribe/url` | Ссылка на аудио |
| GET | `/download/{id}.pdf` | Скачать PDF |
| GET | `/health` | Проверка статуса |

## Пример запроса

```bash
curl -X POST "https://your-api.onrender.com/transcribe/file" \
  -F "file=@song.mp3"
```

Ответ:
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "pdf_url": "/download/a1b2c3d4.pdf"
}
```

## Локальный запуск

```bash
pip install -r requirements.txt
# Установите MuseScore и FFmpeg
uvicorn backend_api:app --reload
```

## Ограничения бесплатного тарифа Render
- Сервер "засыпает" после 15 минут без запросов
- Первый запрос после сна занимает 30–60 секунд
- Лимит: 750 часов в месяц
