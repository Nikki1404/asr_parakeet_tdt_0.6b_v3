#app/asr_engines/parakeet_asr.py-
import io
import os
import time
import wave
import tempfile
import logging
import threading
from typing import Optional

import torch

from app.asr_engines.base import ASREngine, EngineCaps

log = logging.getLogger("parakeet_engine")

ACTIVE_INFERENCES = 0
COUNTER_LOCK = threading.Lock()


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
            log.exception("Failed to extract hypothesis.text")
            return ""
    try:
        return str(x)
    except Exception:
        log.exception("Failed to convert ASR output to string")
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

        log.info("Loading model=%s device=%s", self.model_name, self.device)

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        load_sec = time.time() - t0

        if torch.cuda.is_available():
            log.info(
                "Model ready | device=%s load=%.2fs gpu_alloc=%.2fGB gpu_reserved=%.2fGB",
                self.device,
                load_sec,
                torch.cuda.memory_allocated() / (1024 ** 3),
                torch.cuda.memory_reserved() / (1024 ** 3),
            )
        else:
            log.info("Model ready | device=%s load=%.2fs", self.device, load_sec)

        return load_sec

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
            old_sec = len(self.audio) / 2 / self.engine.sr
            self.audio = self.audio[-max_bytes:]
            new_sec = len(self.audio) / 2 / self.engine.sr

            log.warning(
                "AUDIO BUFFER TRIMMED | old=%.2fs new=%.2fs",
                old_sec,
                new_sec,
            )

    def step_if_ready(self) -> Optional[str]:
        now = time.time()

        if not self.audio:
            return None

        if (now - self.last_partial_time) < self.engine.partial_interval_sec:
            return None

        self.last_partial_time = now

        text = self._transcribe_current_buffer(reason="partial").strip()

        if not text:
            return None

        if text == self.current_text:
            return None

        if self.current_text and self.current_text.startswith(text):
            return None

        self.current_text = text
        return text

    def finalize(self, pad_ms):
        final = self._transcribe_current_buffer(reason="final").strip()

        if final:
            self.current_text = final

        out = self.current_text.strip()

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0

        return out

    def _transcribe_current_buffer(self, reason: str) -> str:
        global ACTIVE_INFERENCES

        tmp_path = None
        audio_sec = len(self.audio) / 2 / self.engine.sr

        if not self.audio:
            log.warning("TRANSCRIBE SKIPPED | reason=%s audio_empty=True", reason)
            return ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(self._pcm_to_wav(bytes(self.audio)))

            with COUNTER_LOCK:
                ACTIVE_INFERENCES += 1
                parallel = ACTIVE_INFERENCES

            log.info(
                "INFER START | reason=%s parallel=%s audio=%.2fs",
                reason,
                parallel,
                audio_sec,
            )

            if torch.cuda.is_available():
                log.info(
                    "GPU STATE | parallel=%s alloc=%.2fGB reserved=%.2fGB",
                    parallel,
                    torch.cuda.memory_allocated() / (1024 ** 3),
                    torch.cuda.memory_reserved() / (1024 ** 3),
                )

            t0 = time.perf_counter()
            results = self.engine.model.transcribe([tmp_path])
            infer_sec = time.perf_counter() - t0

            log.info(
                "INFER END | reason=%s parallel=%s audio=%.2fs infer=%.2fs rtf=%.2f",
                reason,
                parallel,
                audio_sec,
                infer_sec,
                infer_sec / max(audio_sec, 0.01),
            )

            if not results:
                log.warning(
                    "TRANSCRIBE EMPTY RESULT | reason=%s audio=%.2fs",
                    reason,
                    audio_sec,
                )
                return ""

            text = safe_text(results[0]).strip()

            log.info(
                "TRANSCRIBE TEXT | reason=%s text=%r",
                reason,
                text,
            )

            return text

        except RuntimeError as exc:
            msg = str(exc)

            if "CUDNN_STATUS_INTERNAL_ERROR" in msg:
                log.error(
                    "\n"
                    "🚨 PARAKET LOAD TEST GPU FAILURE DETECTED\n"
                    "Reason: Multiple parallel users triggered concurrent model.transcribe() calls "
                    "on the same NeMo Parakeet model instance.\n"
                    "Why this fails: Parakeet / NeMo RNNT transcribe() is not thread-safe for "
                    "concurrent GPU execution. CUDA/cuDNN internal state can be corrupted under "
                    "parallel threaded inference.\n"
                    "Observed active parallel inference calls: %s\n"
                    "What this means for load testing: this is the practical per-model concurrency "
                    "limit being reached, not a normal transcription error.\n"
                    "Better production options: use separate model replicas/processes, GPU-aware "
                    "autoscaling, or a true streaming inference backend such as Riva/NIM.\n",
                    ACTIVE_INFERENCES,
                )

            elif "Cannot unfreeze partially" in msg:
                log.error(
                    "\n"
                    "🚨 NEMO MODEL STATE CORRUPTION DETECTED\n"
                    "Reason: Concurrent transcribe() calls mutated NeMo internal model state "
                    "during freeze/unfreeze lifecycle.\n"
                    "Fix: Do not run multiple threaded transcribe() calls on the same model instance.\n"
                )

            log.exception("TRANSCRIBE RUNTIME ERROR | reason=%s audio=%.2fs", reason, audio_sec)
            return ""

        except Exception:
            log.exception("TRANSCRIBE ERROR | reason=%s audio=%.2fs", reason, audio_sec)
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    log.exception("TEMP WAV CLEANUP FAILED | path=%s", tmp_path)

            with COUNTER_LOCK:
                ACTIVE_INFERENCES -= 1

    def _pcm_to_wav(self, pcm):
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.engine.sr)
            wf.writeframes(pcm)

        return buf.getvalue()

#app/main.py-
import asyncio
import json
import logging
import sys
import time
import uuid

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.config import load_config
from app.factory import build_engine
from app.streaming_session import StreamingSession

cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("parakeet_server")

app = FastAPI()
engine = build_engine(cfg)


@app.on_event("startup")
async def startup():
    load_sec = engine.load()
    log.info("MODEL LOADED | time=%.2fs", load_sec)


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    conn_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    await ws.accept()
    log.info("[%s] CONNECTION CREATED", conn_id)

    session = StreamingSession(engine, cfg, conn_id=conn_id)

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()

            events = await loop.run_in_executor(None, session.process_chunk, data)

            process_time = time.perf_counter() - t0
            if process_time > 1.0:
                log.warning(
                    "[%s] SLOW PROCESS_CHUNK | time=%.2fs bytes=%s events=%s",
                    conn_id,
                    process_time,
                    len(data),
                    len(events),
                )

            for ev in events:
                if len(ev) == 3:
                    ev_type, text, ttfb_ms = ev
                    payload = {
                        "type": ev_type,
                        "text": text,
                        "t_start": ttfb_ms,
                    }
                else:
                    ev_type, text = ev
                    payload = {
                        "type": ev_type,
                        "text": text,
                    }

                log.info("[%s] WS SEND | payload=%s", conn_id, payload)
                await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        log.info("[%s] CONNECTION DISCONNECTED BY CLIENT", conn_id)

    except Exception as exc:
        log.exception("[%s] CONNECTION ERROR | error=%s", conn_id, exc)

    finally:
        session.close()
        log.info(
            "[%s] CONNECTION CLOSED | duration=%.2fs",
            conn_id,
            time.time() - start_time,
        )

#app/streaming_session.py-
import time
import logging
from app.vad import AdaptiveEnergyVAD

log = logging.getLogger("streaming_session")


class StreamingSession:

    def __init__(self, engine, cfg, conn_id: str = "unknown"):
        self.engine = engine
        self.cfg = cfg
        self.conn_id = conn_id

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

        log.info(
            "[%s] SESSION CREATED | frame_bytes=%s vad_frame_ms=%s",
            self.conn_id,
            self.frame_bytes,
            self.cfg.vad_frame_ms,
        )

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

                self.session.accept_pcm16(pre)

                log.info("[%s] UTTERANCE START", self.conn_id)

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

                    log.info(
                        "[%s] PARTIAL TRANSCRIPT | ttfb_ms=%s text=%r",
                        self.conn_id,
                        ttfb_ms,
                        text,
                    )

                    events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                log.info(
                    "[%s] ENDPOINT DETECTED | utt_ms=%s silence_ms=%s",
                    self.conn_id,
                    self.utt_audio_ms,
                    self.silence_ms,
                )

                final = self.session.finalize(self.engine.finalize_pad_ms)

                if final:
                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial is not None
                        else None
                    )

                    log.info(
                        "[%s] FINAL TRANSCRIPT | ttfb_ms=%s text=%r",
                        self.conn_id,
                        ttfb_ms,
                        final,
                    )

                    events.append(("transcript", final, ttfb_ms))
                else:
                    log.warning("[%s] FINAL EMPTY", self.conn_id)

                self.reset()

        return events

    def reset(self):
        log.info("[%s] SESSION RESET", self.conn_id)

        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None

    def close(self):
        log.info("[%s] SESSION CLOSED", self.conn_id)


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "30", "--ws-ping-timeout", "300"]

A machine image contains a VM's properties, metadata, permissions and data from all its attached disks. You can use a machine image to create, backup or restore a VM. Learn more 

Name
cx-asr-v2-image
Name is permanent
Description
Source VM instance
cx-asr-v2
Location

Multi-regional

Regional
Select location
us (multiple regions in the United States)
Encryption

Google-managed encryption key
Keys owned by Google

Cloud KMS key
Keys owned by customers
You can now automate creation of Cloud KMS keys using Autokey.

Customer-supplied encryption key (CSEK)
Manage outside of Google Cloud
