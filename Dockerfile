FROM python:3.11-alpine

ENV PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1 \
     PIP_DISABLE_PIP_VERSION_CHECK=on

# Install ffmpeg, Docker CLI and dependencies
RUN apk --no-cache add ffmpeg docker curl

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir

# Create a dagger directory for cache
RUN mkdir -p /root/.dagger

CMD ["python", "bot/main.py"]