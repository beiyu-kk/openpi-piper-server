# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang

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
# GITHUB_PROXY_PREFIX can be set to a trusted proxy prefix such as
# "https://example-proxy/"; it is prepended to every https://github.com/ URL.
ARG GITHUB_PROXY_PREFIX=
RUN git config --global http.version HTTP/1.1 && \
    if [ -n "$GITHUB_PROXY_PREFIX" ]; then \
        git config --global \
            url."${GITHUB_PROXY_PREFIX}https://github.com/".insteadOf \
            "https://github.com/"; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    for attempt in 1 2 3; do \
        GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev && exit 0; \
        echo "uv sync failed (attempt ${attempt}/3); retrying in 5 seconds"; \
        sleep 5; \
    done; \
    exit 1

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
