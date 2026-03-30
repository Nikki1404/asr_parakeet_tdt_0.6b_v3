FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

#ENV http_proxy="http://163.116.128.80:8080"
#ENV https_proxy="http://163.116.128.80:8080"
ENV PYTHONUNBUFFERED=1

WORKDIR /srv

# Install Python + system deps
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Download and prepare models
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("Downloading Nemotron from HuggingFace")

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)
model.save_to("/srv/nemotron-speech-streaming-en-0.6b.nemo")
print("Nemotron saved as .nemo")
print("Warm loading Nemotron (CPU)")

model = nemo_asr.models.ASRModel.restore_from(
    restore_path="/srv/nemotron-speech-streaming-en-0.6b.nemo",
    map_location="cpu"
)
print("Nemotron warm load complete")
EOF

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
