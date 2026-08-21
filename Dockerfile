FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    musescore3 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# basic-pitch pulls tflite-runtime as a default Linux dependency, and then
# prefers it over ONNX when choosing which model graph to load. The pinned
# tflite-runtime (2.14) predates NumPy 2.x, which this image installs, so
# its interpreter cannot initialise and the chosen .tflite graph fails with
# "cannot be loaded into either TensorFlow, CoreML, TFLite or ONNX" - which
# is what made every v7.6.0/v7.6.1 request fall back to the old monophonic
# engine. The backend now picks the ONNX graph explicitly, but dropping the
# unusable runtime also repairs basic-pitch's own default selection so the
# two can't disagree. Guarded so the build still succeeds if it is absent.
RUN pip uninstall -y tflite-runtime || true

# voices.py is imported by backend_api.py. It has to be copied
# explicitly: this image copies named files, not the directory, so a new
# module that is not listed here simply is not in the container and the
# service dies on import at startup.
COPY backend_api.py voices.py .

ENV QT_QPA_PLATFORM=offscreen

CMD ["sh", "-c", "uvicorn backend_api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
