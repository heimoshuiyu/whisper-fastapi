FROM docker.io/nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y ffmpeg python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# 启动 whisper_fastapi.py
ENTRYPOINT ["python3", "whisper_fastapi.py"]
