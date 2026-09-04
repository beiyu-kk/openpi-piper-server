# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs and opencv-python dynamically loads GLib/OpenGL.
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings.
# This mirror contains the exact python-build-standalone artifacts requested
# by uv and avoids GitHub release download timeouts on mainland networks.
ENV UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
ENV UV_HTTP_TIMEOUT=120

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Some networks terminate GitHub's HTTP/2 connections during long fetches.
# Default to the proxy used by the Piper training environment while allowing
# callers to override it with a trusted internal proxy when available.
ARG GITHUB_PROXY_PREFIX=https://ghfast.top/
RUN git config --global http.version HTTP/1.1 && \
    if [ -n "$GITHUB_PROXY_PREFIX" ]; then \
        git config --global \
            url."${GITHUB_PROXY_PREFIX}https://github.com/".insteadOf \
            "https://github.com/"; \
    fi

# Fail quickly when the configured GitHub route is unavailable instead of
# waiting for uv's Git dependency fetch to hit a multi-minute TCP timeout.
RUN timeout 20 git ls-remote https://github.com/huggingface/lerobot HEAD >/dev/null || \
    (echo "Cannot reach the LeRobot Git repository; configure GITHUB_PROXY_PREFIX" && exit 1)

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# TorchCodec dynamically loads FFmpeg when LeRobot decodes dataset videos.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Validate native image and video dependencies while building the image.
RUN /.venv/bin/python -c "import cv2; from torchcodec.decoders import VideoDecoder; print('OpenCV and TorchCodec imports passed')"

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
