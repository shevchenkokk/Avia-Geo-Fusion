FROM python:3.11-slim

ARG DINOV2_REF=7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c
ARG XFEAT_REF=e92685f57f8318b18725c5c8c0bd28c7fe188d9a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    TORCH_HOME=/opt/torch

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

RUN mkdir -p /opt/torch/hub \
    && git clone --depth 1 https://github.com/facebookresearch/dinov2.git /opt/torch/hub/facebookresearch_dinov2_main \
    && git -C /opt/torch/hub/facebookresearch_dinov2_main fetch --depth 1 origin "$DINOV2_REF" \
    && git -C /opt/torch/hub/facebookresearch_dinov2_main checkout "$DINOV2_REF" \
    && git clone --depth 1 https://github.com/verlab/accelerated_features.git /opt/torch/hub/verlab_accelerated_features_main \
    && git -C /opt/torch/hub/verlab_accelerated_features_main fetch --depth 1 origin "$XFEAT_REF" \
    && git -C /opt/torch/hub/verlab_accelerated_features_main checkout "$XFEAT_REF" \
    && rm -rf /opt/torch/hub/facebookresearch_dinov2_main/.git /opt/torch/hub/verlab_accelerated_features_main/.git \
    && python -c "import torch; torch.hub.load('/opt/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source='local', verbose=False); torch.hub.load('/opt/torch/hub/verlab_accelerated_features_main', 'XFeat', pretrained=True, top_k=4096, source='local', verbose=False)"

CMD ["bash"]
