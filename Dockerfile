FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.9 python3-pip git wget curl \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install pillow pandas

WORKDIR /app
COPY . /app

CMD ["python", "report_generator.py", "-pt", "fewshot", "-et", "base", "-lt", "base", "-it", "tiff", "-d", "cuda:0"]
