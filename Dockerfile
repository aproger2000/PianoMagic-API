FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    musescore3 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend_api.py .

ENV QT_QPA_PLATFORM=offscreen

CMD ["sh", "-c", "uvicorn backend_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
