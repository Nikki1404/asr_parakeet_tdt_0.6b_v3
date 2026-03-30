FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${http_proxy}
ENV HTTPS_PROXY=${https_proxy}
ENV PYTHONUNBUFFERED=1

WORKDIR /srv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        ffmpeg \
        wget \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

RUN python3.11 - <<EOF
import nemo.collections.asr as nemo_asr
print("Downloading Parakeet model...")
_ = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    map_location="cpu"
)
print("Parakeet downloaded successfully.")
EOF

COPY app ./app
COPY client.py ./client.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--ws-ping-interval", "20", "--ws-ping-timeout", "120"]