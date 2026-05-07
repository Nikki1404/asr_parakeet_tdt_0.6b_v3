#app/asr_engines/parakeet_asr.py-
import io
import os
import time
import wave
import tempfile
import logging
from typing import Optional

from app.asr_engines.base import ASREngine, EngineCaps

log = logging.getLogger("parakeet_engine")

ACTIVE_INFERENCES = 0


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

        log.info(f"Loading model: {self.model_name}")

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        log.info(f"Model loaded on device={self.device}")

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

        log.info("ASR SESSION CREATED")

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

        log.info(f"PARTIAL TRANSCRIPT | text={text}")

        return text

    def finalize(self, pad_ms):
        final = self._transcribe_current_buffer().strip()

        if final:
            self.current_text = final

        out = self.current_text.strip()

        if out:
            log.info(f"FINAL TRANSCRIPT | text={out}")

        self.audio.clear()
        self.current_text = ""
        self.last_partial_time = 0.0

        log.info("ASR SESSION FINALIZED")

        return out

    def _transcribe_current_buffer(self) -> str:
        global ACTIVE_INFERENCES

        tmp_path = None
        audio_sec = len(self.audio) / 2 / self.engine.sr
        reason = "transcribe"

        try:
            ACTIVE_INFERENCES += 1

            log.info(
                "TRANSCRIBE START | audio_sec=%.2f active_parallel=%s",
                audio_sec,
                ACTIVE_INFERENCES,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(self._pcm_to_wav(bytes(self.audio)))

            t0 = time.perf_counter()

            results = self.engine.model.transcribe([tmp_path])

            infer_sec = time.perf_counter() - t0

            log.info(
                "TRANSCRIBE END | audio_sec=%.2f infer_sec=%.2f rtf=%.2f",
                audio_sec,
                infer_sec,
                infer_sec / max(audio_sec, 0.01),
            )

            if not results:
                return ""

            return safe_text(results[0]).strip()

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

            log.exception(
                "TRANSCRIBE RUNTIME ERROR | reason=%s audio=%.2fs",
                reason,
                audio_sec,
            )
            return ""

        except Exception:
            log.exception(
                "TRANSCRIBE ERROR | reason=%s audio=%.2fs",
                reason,
                audio_sec,
            )
            return ""

        finally:
            ACTIVE_INFERENCES -= 1

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


#app/main.py-
import asyncio
import json
import logging
import sys
import uuid
import time

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
    log.info(f"Model loaded in {load_sec:.2f}s")


@app.websocket("/ws")
async def ws_asr(ws: WebSocket):

    conn_id = str(uuid.uuid4())[:8]
    conn_start = time.time()

    await ws.accept()

    log.info(f"[{conn_id}] CONNECTION CREATED")

    session = StreamingSession(engine, cfg)

    try:
        while True:
            data = await ws.receive_bytes()

            loop = asyncio.get_running_loop()

            events = await loop.run_in_executor(
                None,
                session.process_chunk,
                data,
            )

            for ev in events:
                if len(ev) == 3:
                    ev_type, text, ttfb_ms = ev

                    log.info(
                        f"[{conn_id}] TRANSCRIPT EVENT | type={ev_type} text={text}"
                    )

                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                        "t_start": ttfb_ms,
                    }))
                else:
                    ev_type, text = ev

                    log.info(
                        f"[{conn_id}] TRANSCRIPT EVENT | type={ev_type} text={text}"
                    )

                    await ws.send_text(json.dumps({
                        "type": ev_type,
                        "text": text,
                    }))

    except WebSocketDisconnect:
        log.info(f"[{conn_id}] CLIENT DISCONNECTED")

    finally:
        duration = time.time() - conn_start

        log.info(
            f"[{conn_id}] CONNECTION CLOSED | duration_sec={duration:.2f}"
        )

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

        log.info("STREAMING SESSION CREATED")

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

                log.info("UTTERANCE STARTED")

                self.utt_audio_ms = 0
                self.silence_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None

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

                    log.info(f"PARTIAL GENERATED | text={text}")

                    events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.engine.min_utt_ms
                and self.silence_ms >= self.engine.end_silence_ms
            ):
                log.info(
                    "ENDPOINT DETECTED | utt_ms=%s silence_ms=%s",
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

                    log.info(f"FINAL GENERATED | text={final}")

                    events.append(("transcript", final, ttfb_ms))

                self.reset()

        return events

    def reset(self):
        log.info("STREAMING SESSION RESET")

        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None


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


──────────────────────────────────────────────────────────────────────
  ⏳ [19:27:44] (1590 ms) Maybe. No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the the o  ⏳ [19:27:45] (1590 ms) Maybe. No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the the o  ⏳ [19:27:47] (1590 ms) Maybe. No,'cause it has a sensor and the sensor is on on the dashboard. So I guess it takes And the the one  ⏳ [19:27:49] (1590 ms) Maybe. No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes a and the one  ⏳ [19:27:50] (1590 ms) No because it has a sensor and the sensor is on the dashboard. So I guess it takes a one thing I I I mean t  ⏳ [19:27:52] (1590 ms) Maybe on the dashboard. So I guess it takes a the one thing I I I mean this car provides heat, pero no es c  ⏳ [19:27:53] (1590 ms) Maybe the one thing I I I mean this car provides heat pero no es como los carros normales que si tu quieres  ⏳ [19:27:55] (1590 ms) Maybe. No, because it has a sensor and the sensor is on the dashboard. So I guess it takes a and the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only w  ⏳ [19:27:56] (1590 ms) Maybe on the dashboard. So I guess it takes a and the one thing I I I mean this car provides heat, pero no   ⏳ [19:27:58] (1590 ms) Maybe. No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The o  ⏳ [19:27:59] (1590 ms) Maybe. No, because it has a sensor and the sensor is on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only  ⏳ [19:28:01] (1590 ms) Maybe. No, because it has a sensor and the sensor is on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only  ⏳ [19:28:02] (1590 ms) Maybe on the dashboard. So I guess it takes a and the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves al rojito. The only way I can turn the heat is I have to tell it au  ⏳ [19:28:04] (1590 ms) Maybe on the dashboard. So I guess it takes a the one thing I I I mean this car provides heat, pero no es c  ⏳ [19:28:06] (1590 ms) Maybe on the dashboard. So I guess it takes a and the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only way I can turn the heat is I have to tell it aut  ⏳ [19:28:07] (1590 ms) Maybe on the dashboard. So I guess it takes a and the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only way I can turn the heat is I have to tell it aut  ⏳ [19:28:08] (1590 ms) Maybe. No, because it has a sensor and the sensor is on the dashboard. So I guess it takes and the the one other thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the sensor will know that it's col  ⏳ [19:28:10] (1590 ms) Maybe. No, because it has a sensor and the sensor is on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat pero no es como los carros normales que si tu quieres calor tu lo mueves al rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the sensor will know that it's cold outsi  ⏳ [19:28:12] (1590 ms) Maybe. No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat pero no es como los carros normales que si tu quieres calor tu lo mueves a rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the sensor will know that it's cold out  ⏳ [19:28:14] (1590 ms) No, because it has a sensor and the sensor is on on the dashboard. So I guess it takes and the the one thing I I I mean this car provides heat, pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the sensor will know that it's cold outside   ⏳ [19:28:15] (1590 ms) And the sensor is on the dashboard. So I guess it takes and the the one other thing I I I mean this car provides heat, but no carrots normally a rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the sensor will know that it's cold outside and it throws hot air. Like you don't you can have and it takes time. Versu  ⏳ [19:28:16] (1590 ms) The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the se  ⏳ [19:28:18] (1590 ms) So I guess it takes and the the one thing I I I mean this car provides heat pero no es como los carros normales que si tu quieres calor tu los mueves a rojito. The only way I can turn the heat is I have to tell it auto and turn the temperat  ⏳ [19:28:19] (1590 ms) The only way I can turn the heat is I have to tell it auto and turn the temperature up. Because then the se                                                                                                                                     [19:28:20] Final  | first partial at 1590 ms
  ✅  And the one thing I I mean this car provides heat pero no es como los carros normales que si tú quieres calor tu los mueves al rojito. The only way I can turn the heat is I have to tell it auto and turn the temperature up because then the sensor will know that it's cold outside and it throws hot air.
