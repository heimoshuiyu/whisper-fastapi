#!/bin/bash

podman run -d --name whisper-fastapi \
    --restart unless-stopped \
    --name whisper-fastapi \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --device nvidia.com/gpu=all --security-opt=label=disable \
    --gpus all \
    -p 5000:5000 \
    docker.io/heimoshuiyu/whisper-fastapi:latest \
    --model large-v2
