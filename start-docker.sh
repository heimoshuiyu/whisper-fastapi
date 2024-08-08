#!/bin/bash

docker run -d --name whisper-fastapi \
    --restart unless-stopped \
    --name whisper-fastapi \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --gpus all \
    -p 5000:5000 \
    docker.io/heimoshuiyu/whisper-fastapi:lastet \
    --model large-v2