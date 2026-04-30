FROM python:3.14-alpine AS builder

WORKDIR /usr/bot

RUN apk add --no-cache gcc musl-dev libffi-dev build-base

COPY requirements.txt ./
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.14-alpine

WORKDIR /usr/bot

RUN apk add --no-cache ffmpeg

COPY --from=builder /venv /venv
COPY . /usr/bot/

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH="/venv/bin:$PATH"

CMD ["ddtrace-run", "python", "bot/main.py"]
