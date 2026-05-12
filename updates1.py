#app/asr_engines/base.py-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class EngineCaps:
    streaming: bool
    partials: bool
    ttft_meaningful: bool


class ASRSession(Protocol):
    def accept_pcm16(self, pcm16: bytes) -> None: ...
    def step_if_ready(self) -> Optional[str]: ...
    def finalize(self, pad_ms: int) -> str: ...


class ASREngine(ABC):
    caps: EngineCaps

    @abstractmethod
    def load(self) -> float:
        ...

    @abstractmethod
    def new_session(self, max_buffer_ms: int) -> ASRSession:
        ...

#app/asr_engines/parakeet_asr.py
import io
import os
import time
import wave
import tempfile
from typing import Optional

from app.asr_engines.base import ASREngine, EngineCaps


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and x:
        return safe_text(x[0])
    if hasattr(x, "text"):
        try:
            return x.text or ""
        except Exception:
            return ""
    try:
        return str(x)
    except Exception:
        return ""


class ParakeetASR(ASREngine):

    caps = EngineCaps(
        streaming=False,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, model_name, device, sample_rate):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.model = None

        self.end_silence_ms = 700
        self.min_utt_ms = 300
        self.finalize_pad_ms = 0

        self.partial_interval_sec = 1.5

    def load(self):
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        return time.time() - t0

    def new_session(self, max_buffer_ms):
        return ParakeetSession(self, max_buffer_ms)


class ParakeetSession:
    def __init__(self, engine, max_buffer_ms):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * max_buffer_ms / 1000)

        self.audio = bytearray()
        self.current_text = ""
        self.last_partial_time = 0.0

    def accept_pcm16(self, pcm16):
        self.audio.extend(pcm16)

        max_bytes = self.max_buffer_samples * 2
        if len(self.audio) > max_bytes:
            self.audio = self.audio[-max_bytes:]

    def step_if_ready(self) -> Optional[str]:
        now = time.time()

        if not self.audio:
            return None

        if (now - self.last_partial_time) < self.engine.partial_interval_sec:
            return None

        self.last_partial_time = now

        text = self._transcribe_current_buffer()
        text = text.strip()

        if not text:
            return None

        if text == self.current_text:
            return None

        if self.current_text and self.current_text.startswith(text):
            return None

        self.current_text = text
        return text

    def finalize(self, pad_ms):
        final = self._transcribe_current_buffer().strip()

        if final:
            self.current_text = final

        out = self.current_text.strip()

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0

        return out

    def _transcribe_current_buffer(self) -> str:
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(self._pcm_to_wav(bytes(self.audio)))

            results = self.engine.model.transcribe([tmp_path])

            if not results:
                return ""

            return safe_text(results[0]).strip()

        except Exception:
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _pcm_to_wav(self, pcm):
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.engine.sr)
            wf.writeframes(pcm)

        return buf.getvalue()
    
#app/config.py-
from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class Config:
    asr_backend: str = os.getenv("ASR_BACKEND", "parakeet")
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")

    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "30"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


MODEL_MAP = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
}


def load_config():
    cfg = Config()

    if not cfg.model_name:
        cfg = replace(
            cfg,
            model_name=MODEL_MAP[cfg.asr_backend]
        )

    return cfg

#app/factory.py-
from app.config import Config
from app.asr_engines.parakeet_asr import ParakeetASR


def build_engine(cfg: Config):

    if cfg.asr_backend == "parakeet":
        return ParakeetASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(
        f"Unsupported ASR_BACKEND={cfg.asr_backend}"
    )

#app/main.py-
import asyncio
import json
import logging
import sys

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.config import load_config
from app.factory import build_engine
from app.streaming_session import StreamingSession

cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("parakeet_server")

app = FastAPI()
engine = build_engine(cfg)


@app.on_event("startup")
async def startup():
    load_sec = engine.load()
    log.info(f"Model loaded in {load_sec:.2f}s")


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    client = ws.client  # (host, port)
    log.info(f"[CONNECTION] Client connected: {client}")          # ← NEW

    session = StreamingSession(engine, cfg)
    log.info(f"[SESSION] New session created for {client}")       # ← NEW

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.process_chunk, data)

            for ev in events:
                if len(ev) == 3:
                    ev_type, text, ttfb_ms = ev
                    log.info(f"[{ev_type.upper()}] ttfb={ttfb_ms}ms | {text}")  # ← NEW
                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                        "t_start": ttfb_ms,
                    }))
                else:
                    ev_type, text = ev
                    log.info(f"[{ev_type.upper()}] {text}")                      # ← NEW
                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                    }))

    except WebSocketDisconnect:
        log.info(f"[DISCONNECT] Client disconnected: {client}")  

#app/streaming_session.py-
import logging
import time
from app.vad import AdaptiveEnergyVAD

log = logging.getLogger("parakeet_server")   # ← NEW


class StreamingSession:

    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg

        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )

        self.session = engine.new_session(cfg.max_utt_ms)

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()

        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0

        self.t_utt_start = None
        self.t_first_partial = None

    def process_chunk(self, pcm):
        events = []

        self.raw_buf.extend(pcm)

        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[:self.frame_bytes])
            del self.raw_buf[:self.frame_bytes]

            is_speech, pre = self.vad.push_frame(frame)

            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = 0
                self.silence_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None
                log.info("[UTT_START] Utterance started (VAD triggered)")  # ← NEW

                self.session.accept_pcm16(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials:
                text = self.session.step_if_ready()

                if text:
                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()

                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)
                    events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                final = self.session.finalize(self.engine.finalize_pad_ms)

                if final:
                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial is not None
                        else None
                    )
                    log.info(                                                    # ← NEW
                        f"[UTT_END] Finalized | dur={self.utt_audio_ms}ms "
                        f"ttfb={ttfb_ms}ms | transcript: {final}"
                    )
                    events.append(("transcript", final, ttfb_ms))

                self.reset()

        return events

    def reset(self):
        log.info("[SESSION_RESET] Session state reset after utterance")  # ← NEW
        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None

#app/vad.py-
from collections import deque
import numpy as np


class AdaptiveEnergyVAD:

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        start_margin: float,
        min_noise_rms: float,
        pre_speech_ms: int,
    ):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.start_margin = start_margin
        self.min_noise_rms = min_noise_rms

        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

        self.pre_frames = max(1, int(pre_speech_ms / frame_ms))
        self.ring = deque(maxlen=self.pre_frames)

        self.in_speech = False
        self.noise_rms = min_noise_rms

    def reset(self):
        self.ring.clear()
        self.in_speech = False
        self.noise_rms = self.min_noise_rms

    def _rms(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)

        if not self.in_speech:
            alpha = 0.95
            self.noise_rms = max(
                self.min_noise_rms,
                alpha * self.noise_rms + (1 - alpha) * e
            )

        threshold = self.noise_rms * self.start_margin
        is_speech = e >= threshold

        self.ring.append(frame_pcm16)

        pre_roll = None

        if (not self.in_speech) and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)

        return is_speech, pre_roll
    
#Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Proxy (optional)
ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    git \
    wget \
    curl \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch GPU
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY app ./app

# Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface

# Preload model into image
RUN python -c "\
import nemo.collections.asr as nemo_asr; \
m = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); \
print('Model cached ✓')"

EXPOSE 8001

CMD [
    "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8001",
    "--ws-ping-interval", "20",
    "--ws-ping-timeout", "120"
]

just add logs only don't even change anything else log whenever we are creating the connection, it should get logged, but along with that, we need another log like transcriptions  as the log? Second, you should log the session creation and disconnection. Basically, the complete session lifecycle should be logged in .
