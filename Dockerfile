# Assistive Vision — container image.
#
# The device normally talks to a real camera.  This image is for:
#   * CI / headless smoke tests
#   * deploying the dashboard + API on a host where the camera is
#     passed through (or a network/virtual camera is used)
#
# Build:
#   docker build -t assistive-vision .
# Run (dashboard + API on :5000):
#   docker run --rm -p 5000:5000 assistive-vision
#
# GPU note: on-device inference is CPU (OpenCV DNN + RapidOCR), so this
# base image intentionally has no CUDA toolchain — keeps it small and
# portable.  See docs/productization/containerization.md.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASSIST_HEADLESS=1 \
    ASSIST_HOST=0.0.0.0

WORKDIR /app

# System deps: OpenCV needs libGL; TTS (pyttsx3) needs espeak on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        espeak \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights are git-ignored; mount or download them at runtime.
RUN mkdir -p models && touch models/.gitkeep

EXPOSE 5000

CMD ["python", "src/server/app.py", "--host", "0.0.0.0"]