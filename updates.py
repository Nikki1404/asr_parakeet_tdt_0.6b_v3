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

#app/asr_engines/parakeet_asr.py-
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

@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    

@app.websocket("/asr/ml/ws")
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

log = logging.getLogger("parakeet_server")


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
                log.info("[UTT_START] Utterance started (VAD triggered)")

                self.session.accept_pcm16(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials and self.utt_audio_ms >= 800:  # ← CHANGED
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
                    log.info(
                        f"[UTT_END] Finalized | dur={self.utt_audio_ms}ms "
                        f"ttfb={ttfb_ms}ms | transcript: {final}"
                    )
                    events.append(("transcript", final, ttfb_ms))

                self.reset()

        return events

    def reset(self):
        log.info("[SESSION_RESET] Session state reset after utterance")
        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None


#app/vad.py
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


#Dockerfile-
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

CMD ["uvicorn","app.main:app","--host", "0.0.0.0","--port", "8001","--ws-ping-interval", "20","--ws-ping-timeout", "120"]


this is my server code 

and this is client code -
#client.py-
"""
Parakeet ASR – Terminal Microphone / File Streaming Client
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

import numpy as np
import pyaudio
import websockets

# =============================================================================
# CONFIG
# =============================================================================

# CHANGE ONLY THIS WHEN SERVER CHANGES
#WS_URL = "ws://localhost:8001/ws"
WS_URL = "wss://cx-asr.exlservice.com/asr/ml/ws"

SAMPLE_RATE = 16_000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_MS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BANNER = rf"""
╔══════════════════════════════════════════════════════════════╗
║        Parakeet-TDT-0.6B-v3  |  Real-Time ASR Client        ║
║          Auto language detection (EN / ES / more)           ║
║            WebSocket: {WS_URL[:52]:<52}║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
  q         Quit
  d         List available audio input devices

Notes
  - Partial transcripts are shown while speaking.
  - Final transcript arrives after endpointing on the server.
"""

# =============================================================================
# AUDIO CAPTURE
# =============================================================================

class MicCapture:

    def __init__(self, device_index: Optional[int] = None):
        self.queue: Queue[bytes] = Queue(maxsize=512)
        self._stop = threading.Event()
        self._device = device_index
        self._thread = threading.Thread(target=self._capture, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _capture(self):

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self._device,
                frames_per_buffer=CHUNK_SAMPLES,
            )

            while not self._stop.is_set():
                data = stream.read(
                    CHUNK_SAMPLES,
                    exception_on_overflow=False,
                )
                self.queue.put(data)

        except OSError as exc:
            print(f"\n[ERROR] Audio capture failed: {exc}", file=sys.stderr)
            self._stop.set()

        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass

            pa.terminate()


def list_devices():

    pa = pyaudio.PyAudio()

    print("\nAvailable audio input devices:")

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:
            print(
                f"  [{i}] {info['name']}  "
                f"({int(info['defaultSampleRate'])} Hz)"
            )

    pa.terminate()
    print()

# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio_as_16k_pcm(path: str) -> bytes:

    try:
        import soundfile as sf

        data, sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        if data.ndim == 2:
            data = data.mean(axis=1)

        if sr != SAMPLE_RATE:
            data = _resample(data, sr, SAMPLE_RATE)

        return _float32_to_pcm16(data)

    except Exception as sf_err:

        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(path)
            seg = seg.set_channels(1)
            seg = seg.set_frame_rate(SAMPLE_RATE)
            seg = seg.set_sample_width(2)

            return seg.raw_data

        except Exception as pd_err:

            raise RuntimeError(
                f"Cannot load '{path}'.\n"
                f"soundfile : {sf_err}\n"
                f"pydub     : {pd_err}\n"
                f"Tip: install ffmpeg."
            )


def _resample(data: np.ndarray, orig_sr: int, target_sr: int):

    try:
        import librosa

        return librosa.resample(
            data,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)

        return resample_poly(
            data,
            target_sr // g,
            orig_sr // g,
        ).astype(np.float32)

    except ImportError:
        pass

    duration = len(data) / orig_sr

    old_times = np.linspace(
        0,
        duration,
        len(data),
        endpoint=False,
    )

    new_len = int(duration * target_sr)

    new_times = np.linspace(
        0,
        duration,
        new_len,
        endpoint=False,
    )

    return np.interp(
        new_times,
        old_times,
        data,
    ).astype(np.float32)


def _float32_to_pcm16(data: np.ndarray) -> bytes:

    return (
        np.clip(data, -1.0, 1.0) * 32767
    ).astype(np.int16).tobytes()

# =============================================================================
# RECEIVER
# =============================================================================

async def _receiver(ws):

    last_partial = ""

    try:
        async for raw in ws:

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            text = msg.get("text", "")
            t_start = msg.get("t_start")

            ts = time.strftime("%H:%M:%S")

            if msg_type == "partial":

                last_partial = text

                if t_start is not None:
                    print(
                        f"\r  ⏳ [{ts}] ({t_start} ms) {text:<100}",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  ⏳ [{ts}] {text:<100}",
                        end="",
                        flush=True,
                    )

            elif msg_type == "transcript":

                print(f"\r{' ' * 140}\r", end="")

                if t_start is not None:
                    print(f"[{ts}] Final | first partial at {t_start} ms")
                else:
                    print(f"[{ts}] Final")

                print(f"  ✅  {text}")
                print("─" * 72)

                last_partial = ""

    except websockets.exceptions.ConnectionClosed:

        if last_partial:
            print()

        print("[Connection closed]")

# =============================================================================
# STDIN READER
# =============================================================================

def stdin_reader(cmd_queue: Queue):

    while True:
        try:
            line = sys.stdin.readline()

            if line:
                cmd_queue.put(line.strip())

        except Exception:
            break

# =============================================================================
# MIC MODE
# =============================================================================

async def run_mic(device_index: Optional[int]):

    uri = WS_URL

    print(BANNER)
    print(f"Connecting to {uri} …", end=" ", flush=True)

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
        ) as ws:

            print("connected ✓\n")

            print("🎙  Speak now – transcriptions appear below\n")
            print("─" * 72)

            mic = MicCapture(device_index)
            mic.start()

            cmd_queue: Queue[str] = Queue()

            threading.Thread(
                target=stdin_reader,
                args=(cmd_queue,),
                daemon=True,
            ).start()

            send_task = asyncio.create_task(
                _mic_sender(ws, mic, cmd_queue)
            )

            recv_task = asyncio.create_task(
                _receiver(ws)
            )

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            mic.stop()

    except (OSError, websockets.exceptions.WebSocketException) as exc:

        print(f"failed ✗\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


async def _mic_sender(ws, mic: MicCapture, cmd_queue: Queue):

    print(HELP)

    try:
        while True:

            try:
                cmd = cmd_queue.get_nowait()

                if cmd.lower() in ("q", "quit", "exit"):
                    print("\nBye!")
                    return

                elif cmd.lower() == "d":
                    print()
                    list_devices()

            except Empty:
                pass

            sent = 0

            while not mic.queue.empty() and sent < 50:
                frame = mic.queue.get_nowait()
                await ws.send(frame)
                sent += 1

            await asyncio.sleep(0.005)

    except websockets.exceptions.ConnectionClosed:
        print("\n[Connection closed by server]")

# =============================================================================
# FILE MODE
# =============================================================================

async def run_files(files: List[str], speed: float):

    uri = WS_URL

    print(BANNER)

    print(f"  Mode   : FILE streaming at {speed:.1f}×")
    print(f"  Server : {uri}")
    print(f"  Files  : {', '.join(Path(f).name for f in files)}\n")

    for filepath in files:

        p = Path(filepath)

        if not p.exists():
            print(f"  [SKIP] File not found: {filepath}\n")
            continue

        print(f"  Loading {p.name} …", end=" ", flush=True)

        try:
            pcm = load_audio_as_16k_pcm(str(p))
        except RuntimeError as exc:
            print(f"FAILED ✗\n  {exc}\n")
            continue

        duration_sec = len(pcm) / 2 / SAMPLE_RATE

        print(f"{duration_sec:.1f}s ✓")

        print(f"\n{'═' * 72}")
        print(f"  📄 {p.name} ({duration_sec:.1f}s)")
        print(f"{'═' * 72}\n")

        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**23,
            ) as ws:

                recv_task = asyncio.create_task(
                    _receiver(ws)
                )

                await _file_sender(ws, pcm, speed)

                await asyncio.sleep(3.0)

                await ws.close()

                await asyncio.sleep(0.2)

                if not recv_task.done():
                    recv_task.cancel()

        except (OSError, websockets.exceptions.WebSocketException) as exc:
            print(f"\n[ERROR] WebSocket failed: {exc}")

        print()

    print("All files processed.")

# =============================================================================
# FILE SENDER
# =============================================================================

async def _file_sender(ws, pcm: bytes, speed: float):

    if speed <= 0:
        raise ValueError("--speed must be > 0")

    chunk_delay = (CHUNK_MS / 1000.0) / speed

    offset = 0

    while offset + CHUNK_BYTES <= len(pcm):

        chunk = pcm[offset: offset + CHUNK_BYTES]

        offset += CHUNK_BYTES

        await ws.send(chunk)

        await asyncio.sleep(chunk_delay)

    leftover = pcm[offset:]

    if leftover:
        await ws.send(
            leftover + bytes(CHUNK_BYTES - len(leftover))
        )

# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Parakeet ASR client"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="Mic device index",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List audio devices",
    )

    parser.add_argument(
        "--file",
        nargs="+",
        metavar="FILE",
        help="Audio file(s)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed",
    )

    return parser.parse_args()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    if args.list:
        list_devices()
        sys.exit(0)

    if args.file:
        asyncio.run(run_files(args.file, args.speed))
    else:
        asyncio.run(run_mic(args.device))


and I was asked to do this 
1.Check code to identify cause of gibberish sentences and confirm if language parameter affects Parakeet model's responses 
2. Save audio chunks during transcription and store them for testing 



import asyncio
import json
import logging
import websockets
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# CONFIG
WEBSOCKET_ADDRESS = "ws://192.168.4.38:8001/ws"
TARGET_SR = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = TARGET_SR * CHUNK_MS // 1000
CHUNK_BYTES = CHUNK_SAMPLES * 2


# AUDIO LOADER
def load_audio(filepath: str):
    audio, sr = sf.read(filepath, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    pcm = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    return pcm.tobytes()


# MAIN STREAM FUNCTION
async def stream_parakeet(audio_file: str):
    event_queue = asyncio.Queue()

    pcm_audio = load_audio(audio_file)

    async with websockets.connect(
        WEBSOCKET_ADDRESS,
        max_size=None,
        ping_interval=20,
        ping_timeout=20
    ) as ws:

        logger.info("Connected to websocket")

        async def receive_task():
            """
            Listen for partial + final transcripts
            throughout the full audio stream.
            """
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        obj = json.loads(msg)

                        typ = obj.get("type")
                        txt = obj.get("text", "")

                        logger.info(
                            f"Websocket received msg: {txt}, type: {typ}"
                        )

                        if typ == "partial":
                            await event_queue.put({
                                "type": "INTERIM_TRANSCRIPT",
                                "text": txt
                            })

                        elif typ in ["transcript", "final"]:
                            await event_queue.put({
                                "type": "FINAL_TRANSCRIPT",
                                "text": txt
                            })

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server")

        async def send_task():
            """
            Stream complete audio file chunk by chunk.
            """
            try:
                offset = 0
                total_chunks = 0

                while offset < len(pcm_audio):
                    chunk = pcm_audio[
                        offset: offset + CHUNK_BYTES
                    ]
                    offset += CHUNK_BYTES

                    if len(chunk) < CHUNK_BYTES:
                        chunk += bytes(
                            CHUNK_BYTES - len(chunk)
                        )

                    await ws.send(chunk)
                    total_chunks += 1

                    if total_chunks % 1000 == 0:
                        logger.info(
                            f"Sent {total_chunks} chunks"
                        )

                    await asyncio.sleep(
                        CHUNK_MS / 1000
                    )

                logger.info("Finished sending full audio")

                # final flush
                await asyncio.sleep(0.5)

                await ws.send(
                    json.dumps({"cmd": "flush"})
                )

                logger.info("Flush sent")

                # wait for last transcript
                await asyncio.sleep(10)

                await event_queue.put(None)

            except Exception as e:
                logger.info(f"Error occurred: {e}")
                await event_queue.put(None)

        # start background tasks
        stask = asyncio.create_task(send_task())
        rtask = asyncio.create_task(receive_task())

        try:
            while True:
                event = await event_queue.get()

                if event is None:
                    break

                yield event

        finally:
            stask.cancel()
            rtask.cancel()
 
