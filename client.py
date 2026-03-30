#server.py
import io
import json
import time
import wave

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import nemo.collections.asr as nemo_asr

SAMPLE_RATE = 16000
MODEL_NAME = "nvidia/parakeet-ctc-0.6b"

app = FastAPI()

print("Loading model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name=MODEL_NAME
)
asr_model = asr_model.eval()
print("Model loaded successfully")


def pcm_to_wav_bytes(audio_pcm: bytes, sr: int = SAMPLE_RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_pcm)
    buffer.seek(0)
    return buffer


@app.websocket("/ws/asr")
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    audio_buffer = bytearray()

    try:
        while True:
            chunk = await ws.receive_bytes()

            # EOS signal
            if chunk == b"":
                if len(audio_buffer) == 0:
                    break

                wav_file = pcm_to_wav_bytes(bytes(audio_buffer))

                t0 = time.time()
                text = asr_model.transcribe([wav_file])[0]
                latency = round((time.time() - t0) * 1000, 2)

                await ws.send_text(json.dumps({
                    "type": "final",
                    "text": text,
                    "latency_ms": latency
                }))

                audio_buffer = bytearray()
                continue

    uvicorn.run(app, host="0.0.0.0", port=8000)


#client.py-
import asyncio
import json

import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:8000/ws/asr"
SAMPLE_RATE = 16000
CHUNK_MS = 500
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)


async def stream_audio():
    async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
        print("Connected to ASR server")
        print("Speak now... Ctrl+C to stop")

        def callback(indata, frames, time, status):
            if status:
                print(status)

            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            asyncio.create_task(ws.send(pcm))

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                print(f"[{data['type'].upper()}] {data['text']}")


if __name__ == "__main__":
    asyncio.run(stream_audio())


#requirements.txt
fastapi
uvicorn[standard]
websockets
sounddevice
numpy
nemo_toolkit[asr]
torch
torchaudio


#Dockerfile 
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

COPY server.py .
COPY client.py .

EXPOSE 8000

CMD ["python3", "server.py"]
