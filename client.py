#server.py
import asyncio
import json
import os
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import nemo.collections.asr as nemo_asr

# =========================
# CONFIG
# =========================
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" or "cpu"

# Audio segmentation / endpointing
PARTIAL_EVERY_SEC = float(os.getenv("PARTIAL_EVERY_SEC", "1.5"))
FORCE_FLUSH_SEC = float(os.getenv("FORCE_FLUSH_SEC", "6.0"))
SILENCE_TIMEOUT_MS = int(os.getenv("SILENCE_TIMEOUT_MS", "900"))
MIN_UTT_MS = int(os.getenv("MIN_UTT_MS", "250"))

# Energy VAD
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "0.01"))

app = FastAPI()

print(f"Loading model: {MODEL_NAME}", flush=True)
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

if DEVICE == "cuda":
    asr_model = asr_model.cuda()

asr_model.eval()
print("Model loaded successfully", flush=True)


# =========================
# HELPERS
# =========================
def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert PCM16 mono bytes to float32 [-1, 1]."""
    if not audio_bytes:
        return np.array([], dtype=np.float32)
    x = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return x


def rms_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def transcribe_audio_array(audio_f32: np.ndarray, sample_rate: int) -> str:
    """
    Writes float32 mono audio to a temp wav and runs NeMo transcribe.
    This is simple and reliable across NeMo versions.
    """
    if audio_f32.size == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio_f32, sample_rate, subtype="PCM_16")
        out = asr_model.transcribe([tmp.name])

    if isinstance(out, list) and len(out) > 0:
        text = out[0]
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()

    return ""


class SessionState:
    def __init__(self) -> None:
        self.audio_buffer = bytearray()
        self.in_speech = False

        self.last_speech_time: Optional[float] = None
        self.last_partial_time: float = 0.0
        self.segment_start_time: Optional[float] = None

        self.last_partial_text: str = ""
        self.segment_count: int = 0

        self.bytes_per_frame = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0) * 2)

    def reset_segment(self) -> None:
        self.audio_buffer = bytearray()
        self.in_speech = False
        self.last_speech_time = None
        self.last_partial_time = 0.0
        self.segment_start_time = None
        self.last_partial_text = ""

    def segment_duration_sec(self) -> float:
        return len(self.audio_buffer) / (SAMPLE_RATE * 2.0)

    def enough_audio_for_min_utt(self) -> bool:
        return len(self.audio_buffer) >= int(SAMPLE_RATE * 2 * (MIN_UTT_MS / 1000.0))


async def send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def maybe_send_partial(ws: WebSocket, st: SessionState) -> None:
    now = time.time()
    if not st.in_speech:
        return
    if (now - st.last_partial_time) < PARTIAL_EVERY_SEC:
        return
    if not st.enough_audio_for_min_utt():
        return

    audio_f32 = pcm16_bytes_to_float32(bytes(st.audio_buffer))
    if audio_f32.size == 0:
        return

    try:
        text = transcribe_audio_array(audio_f32, SAMPLE_RATE)
        if text and text != st.last_partial_text:
            st.last_partial_text = text
            st.last_partial_time = now
            await send_json(
                ws,
                {
                    "type": "partial",
                    "text": text,
                    "segment_index": st.segment_count,
                    "audio_sec": round(st.segment_duration_sec(), 2),
                },
            )
    except Exception as e:
        await send_json(ws, {"type": "error", "message": f"partial_failed: {str(e)}"})


async def finalize_segment(ws: WebSocket, st: SessionState, reason: str) -> None:
    if not st.enough_audio_for_min_utt():
        st.reset_segment()
        return

    audio_f32 = pcm16_bytes_to_float32(bytes(st.audio_buffer))
    if audio_f32.size == 0:
        st.reset_segment()
        return

    t0 = time.time()
    try:
        text = transcribe_audio_array(audio_f32, SAMPLE_RATE)
        latency_ms = round((time.time() - t0) * 1000.0, 2)

        if text:
            await send_json(
                ws,
                {
                    "type": "final",
                    "text": text,
                    "reason": reason,
                    "segment_index": st.segment_count,
                    "audio_sec": round(st.segment_duration_sec(), 2),
                    "latency_ms": latency_ms,
                },
            )
            st.segment_count += 1
    except Exception as e:
        await send_json(ws, {"type": "error", "message": f"final_failed: {str(e)}"})

    st.reset_segment()


# =========================
# WEBSOCKET
# =========================
@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    st = SessionState()

    await send_json(
        ws,
        {
            "type": "status",
            "message": "connected",
            "sample_rate": SAMPLE_RATE,
            "model_name": MODEL_NAME,
            "device": DEVICE,
        },
    )

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]

                # EOS from client
                if chunk == b"":
                    await finalize_segment(ws, st, reason="client_eos")
                    continue

                # Add chunk
                st.audio_buffer.extend(chunk)

                # Frame-level speech detection using the latest VAD frame
                if len(st.audio_buffer) >= st.bytes_per_frame:
                    recent = bytes(st.audio_buffer[-st.bytes_per_frame:])
                    recent_f32 = pcm16_bytes_to_float32(recent)
                    current_rms = rms_energy(recent_f32)

                    now = time.time()

                    if current_rms >= VAD_RMS_THRESHOLD:
                        st.last_speech_time = now
                        if not st.in_speech:
                            st.in_speech = True
                            if st.segment_start_time is None:
                                st.segment_start_time = now
                    else:
                        # silence
                        if (
                            st.in_speech
                            and st.last_speech_time is not None
                            and (now - st.last_speech_time) * 1000.0 >= SILENCE_TIMEOUT_MS
                        ):
                            await finalize_segment(ws, st, reason="silence")

                # Force flush very long running utterance
                if st.segment_duration_sec() >= FORCE_FLUSH_SEC:
                    await finalize_segment(ws, st, reason="max_segment")

                # Pseudo partials
                await maybe_send_partial(ws, st)

            elif "text" in message and message["text"] is not None:
                text = message["text"].strip().lower()

                if text == "eos":
                    await finalize_segment(ws, st, reason="client_text_eos")
                else:
                    await send_json(ws, {"type": "status", "message": f"unknown_text_command: {text}"})

            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        try:
            await send_json(ws, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=120,
    )


#client.py-
import asyncio
import json
import signal

import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:8000/ws/asr"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


async def mic_producer(audio_queue: asyncio.Queue, stop_event: asyncio.Event):
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC STATUS] {status}")

        pcm16 = (np.clip(indata[:, 0], -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(audio_queue.put_nowait, pcm16)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        print("Recording started. Press Ctrl+C to stop.")
        while not stop_event.is_set():
            await asyncio.sleep(0.1)


async def sender(ws, audio_queue: asyncio.Queue, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
            await ws.send(chunk)
        except asyncio.TimeoutError:
            continue

    # Send EOS when stopping
    try:
        await ws.send(b"")
    except Exception:
        pass


async def receiver(ws, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            msg = await ws.recv()
            data = json.loads(msg)

            mtype = data.get("type", "unknown")
            if mtype == "partial":
                print(f"[PARTIAL] {data.get('text', '')}")
            elif mtype == "final":
                print(
                    f"[FINAL] {data.get('text', '')} "
                    f"(reason={data.get('reason')}, "
                    f"audio_sec={data.get('audio_sec')}, "
                    f"latency_ms={data.get('latency_ms')})"
                )
            elif mtype == "status":
                print(f"[STATUS] {data}")
            elif mtype == "error":
                print(f"[ERROR] {data.get('message')}")
            else:
                print(f"[MESSAGE] {data}")

        except websockets.ConnectionClosed:
            break
        except Exception as e:
            print(f"[RECV ERROR] {e}")
            break


async def main():
    stop_event = asyncio.Event()
    audio_queue = asyncio.Queue(maxsize=100)

    loop = asyncio.get_running_loop()

    def handle_stop():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_stop)
        except NotImplementedError:
            # Windows fallback
            pass

    print(f"Connecting to {WS_URL} ...")
    async with websockets.connect(
        WS_URL,
        max_size=10_000_000,
        ping_interval=20,
        ping_timeout=120,
    ) as ws:
        print("Connected to server")

        tasks = [
            asyncio.create_task(mic_producer(audio_queue, stop_event)),
            asyncio.create_task(sender(ws, audio_queue, stop_event)),
            asyncio.create_task(receiver(ws, stop_event)),
        ]

        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
        except KeyboardInterrupt:
            stop_event.set()

        await asyncio.sleep(0.5)

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())


#requirements.txt
fastapi==0.116.1
uvicorn[standard]==0.35.0
websockets==15.0.1
sounddevice==0.5.2
soundfile==0.13.1
numpy==1.26.4
nemo_toolkit[asr]


#Dockerfile 
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# =====================================
# INSTALL GPU TORCH FIRST
# =====================================
RUN pip install \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# =====================================
# INSTALL NEMO + APP DEPS
# =====================================
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY server.py .
COPY client.py .

EXPOSE 8000

CMD ["python", "server.py"]


docker run --gpus all -p 8000:8000 -e MODEL_NAME=nvidia/parakeet-tdt-0.6b-v3 -e DEVICE=cuda -e FORCE_FLUSH_SEC=6 -e SILENCE_TIMEOUT_MS=900 -e MIN_UTT_MS=250 nemo-asr-en-es


getting this 
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/asr_parakeet_tdt_0.6b_v3# docker run --gpus all -p 8000:8000 -e MODEL_NAME=nvidia/parakeet-tdt-0.6b-v3 -e DEVICE=cuda -e FORCE_FLUSH_SEC=6 -e SILENCE_TIMEOUT_MS=900 -e MIN_UTT_MS=250 nemo-asr-en-es

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Traceback (most recent call last):
  File "/app/server.py", line 12, in <module>
    import nemo.collections.asr as nemo_asr
  File "/usr/local/lib/python3.11/dist-packages/nemo/collections/asr/__init__.py", line 15, in <module>
    from nemo.collections.asr import data, losses, models, modules
  File "/usr/local/lib/python3.11/dist-packages/nemo/collections/asr/losses/__init__.py", line 15, in <module>
    from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
  File "/usr/local/lib/python3.11/dist-packages/nemo/collections/asr/losses/angularloss.py", line 18, in <module>
    from nemo.core.classes import Loss, Typing, typecheck
  File "/usr/local/lib/python3.11/dist-packages/nemo/core/__init__.py", line 16, in <module>
    from nemo.core.classes import *
  File "/usr/local/lib/python3.11/dist-packages/nemo/core/classes/__init__.py", line 17, in <module>
    import lightning.pytorch
  File "/usr/local/lib/python3.11/dist-packages/lightning/__init__.py", line 20, in <module>
    from lightning.pytorch.callbacks import Callback  # noqa: E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/lightning/pytorch/__init__.py", line 27, in <module>
    from lightning.pytorch.callbacks import Callback  # noqa: E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/lightning/pytorch/callbacks/__init__.py", line 14, in <module>
    from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
  File "/usr/local/lib/python3.11/dist-packages/lightning/pytorch/callbacks/batch_size_finder.py", line 26, in <module>
    from lightning.pytorch.callbacks.callback import Callback
  File "/usr/local/lib/python3.11/dist-packages/lightning/pytorch/callbacks/callback.py", line 22, in <module>
    from lightning.pytorch.utilities.types import STEP_OUTPUT
  File "/usr/local/lib/python3.11/dist-packages/lightning/pytorch/utilities/types.py", line 42, in <module>
    from torchmetrics import Metric
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/__init__.py", line 37, in <module>
    from torchmetrics import functional  # noqa: E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/__init__.py", line 56, in <module>
    from torchmetrics.functional.image._deprecated import (
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/image/__init__.py", line 14, in <module>
    from torchmetrics.functional.image.arniqa import arniqa
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/functional/image/arniqa.py", line 31, in <module>
    from torchvision import transforms
  File "/usr/local/lib/python3.11/dist-packages/torchvision/__init__.py", line 10, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/_meta_registrations.py", line 163, in <module>
    @torch.library.register_fake("torchvision::nms")
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/library.py", line 1087, in register
    use_lib._register_fake(
  File "/usr/local/lib/python3.11/dist-packages/torch/library.py", line 204, in _register_fake
    handle = entry.fake_impl.register(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_library/fake_impl.py", line 50, in register
    if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: operator torchvision::nms does not exist
