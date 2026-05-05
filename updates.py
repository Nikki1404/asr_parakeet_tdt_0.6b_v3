#app/asr_engines/parakeet_asr.py-
import io
import os
import time
import wave
import tempfile
import logging
import threading
import torch

from typing import Optional
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
            return ""
    return str(x)


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
        log.info(f"Loading model: {self.model_name}")

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        log.info("Model ready on %s", self.device)
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

        text = self._transcribe().strip()

        if not text or text == self.current_text:
            return None

        self.current_text = text
        return text

    def finalize(self, pad_ms):
        final = self._transcribe().strip()
        out = final or self.current_text

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0

        return out.strip()

    def _transcribe(self) -> str:
        global ACTIVE_INFERENCES

        tmp_path = None
        audio_sec = len(self.audio) / 2 / self.engine.sr

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(self._pcm_to_wav(bytes(self.audio)))

            with COUNTER_LOCK:
                ACTIVE_INFERENCES += 1
                parallel = ACTIVE_INFERENCES

            log.info(f"INFER START | parallel={parallel} audio={audio_sec:.2f}s")

            t0 = time.perf_counter()
            results = self.engine.model.transcribe([tmp_path])
            dt = time.perf_counter() - t0

            log.info(f"INFER END | parallel={parallel} time={dt:.2f}s")

            if not results:
                return ""

            return safe_text(results[0]).strip()

        except RuntimeError as e:
            if "CUDNN_STATUS_INTERNAL_ERROR" in str(e):
                log.error(
                    f"\n🚨 GPU PARALLEL FAILURE\n"
                    f"Active parallel calls: {ACTIVE_INFERENCES}\n"
                    f"Reason: NeMo model is NOT thread-safe\n"
                    f"Limit reached\n"
                )

            log.exception("TRANSCRIBE ERROR")
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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
    t = engine.load()
    log.info(f"MODEL LOADED in {t:.2f}s")


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):

    conn_id = str(uuid.uuid4())[:8]
    start = time.time()

    await ws.accept()
    log.info(f"[{conn_id}] CONNECTED")

    session = StreamingSession(engine, cfg)

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.process_chunk, data)

            for ev in events:
                if len(ev) == 3:
                    typ, txt, ttfb = ev

                    log.info(f"[{conn_id}] {typ.upper()} → {txt}")

                    await ws.send_text(json.dumps({
                        "type": typ,
                        "text": txt,
                        "t_start": ttfb,
                    }))
                else:
                    typ, txt = ev

                    log.info(f"[{conn_id}] {typ.upper()} → {txt}")

                    await ws.send_text(json.dumps({
                        "type": typ,
                        "text": txt,
                    }))

    except WebSocketDisconnect:
        log.info(f"[{conn_id}] DISCONNECTED")

    finally:
        log.info(f"[{conn_id}] CLOSED | duration={time.time()-start:.2f}s")

#app/streaming_session.py-
import time
import logging
from app.vad import AdaptiveEnergyVAD

log = logging.getLogger("streaming_session")


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

        log.info("SESSION CREATED")

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
                log.info("UTTERANCE START")

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

                    log.info(f"PARTIAL → {text}")

                    events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                log.info(
                    f"ENDPOINT | utt={self.utt_audio_ms}ms silence={self.silence_ms}ms"
                )

                final = self.session.finalize(self.engine.finalize_pad_ms)

                if final:
                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial
                        else None
                    )

                    log.info(f"FINAL → {final}")

                    events.append(("transcript", final, ttfb_ms))

                self.reset()

        return events

    def reset(self):
        log.info("SESSION RESET")

        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--ws-ping-interval", "30", "--ws-ping-timeout", "300"]

: 'INTERIM_TRANSCRIPT', 'text': 'Yeah.'}
INFO:__main__:Websocket received msg: Porque si estuvieran escritos en lenguaje de todo el mundo, type: partial
{'type': 'INTERIM_TRANSCRIPT', 'text': 'Porque si estuvieran escritos en lenguaje de todo el mundo'}
INFO:__main__:Websocket received msg: aunque estuvieran escritos en lenguaje de todo el mundo, maybe, pero., type: partial
{'type': 'INTERIM_TRANSCRIPT', 'text': 'aunque estuvieran escritos en lenguaje de todo el mundo, maybe, pero.'}
INFO:__main__:Connection closed by server
INFO:__main__:Error occurred: no close frame received or sent

why it is stopping at this not even final transcript came and in the mid connection closed by server i recieved ?
