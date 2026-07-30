FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    musescore3 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY backend_api.py .

# Создание директорий
RUN mkdir -p temp output

# Порт
ENV PORT=8000
EXPOSE 8000

# Запуск
CMD ["python", "backend_api.py"]
