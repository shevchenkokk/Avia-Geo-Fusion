FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs

RUN pip install --upgrade pip && pip install -e .

CMD ["bash"]
